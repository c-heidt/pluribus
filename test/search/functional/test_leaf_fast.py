"""Gate: the FastState per-combo leaf rollout (``leaf_fast.continuation_value_vector_fast``).

The rollout is **equilibrium-gated** (it draws its own board runout), so byte-identity
is not asserted end-to-end; the MCCFR equilibrium oracle (run under
``PLURIBUS_SEARCH_CORE=1``) is the acceptance gate.  Here we unit-gate the two *new*
building blocks the rollout adds — every other component (betting make/undo,
``vector_payout_concrete``) is already proven byte-identical to ``PokerEnv``:

* ``FastState.refresh_clusters`` recomputes the per-(seat, street) LUT clusters for a
  drawn board **identically** to what ``PokerEnv._compute_info_set`` would look up, so a
  ``BlueprintPolicy`` reads a correct info-set.
* ``leaf_fast._policy_state`` builds a ``PolicyState`` matching ``PokerEnv.policy_state``
  at the same node (info_set / valid_mask / legal).

Plus a smoke that the rollout runs (finite, zero-sum) and falls back on an off-tree
frontier.  Skips when the core is not built.
"""

import collections
import copy

import numpy as np
import pytest

from poker_ai import _core

pytestmark = pytest.mark.skipif(
    not _core.CORE_AVAILABLE, reason="compiled core extension not built"
)

if _core.CORE_AVAILABLE:
    from environment.player import Player
    from environment.poker_env import (
        PokerEnv, _ACTION_BYTE, _STAGE_ID, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND,
    )
    from poker_ai._core import _state as _cystate
    from poker_ai.search.context import SubgameContext
    from poker_ai.search.leaf import LeafConfig, continuation_value_vector
    from poker_ai.search.leaf_fast import (
        continuation_value_vector_fast, _policy_state,
    )
    from poker_ai.search.mccfr import _BIAS_CLASSES
    from test.search._helpers import UniformPolicy

    _cystate.configure(_STAGE_ID, _ACTION_BYTE, RAISE_SIZES_BY_STAGE, MAX_RAISES_PER_ROUND)
    FastState = _cystate.FastState


class _ClusterStage:
    """Deterministic non-trivial pseudo-cluster per (hole+board) key."""
    def __getitem__(self, key):
        acc = 0
        for c in key:
            acc = (acc * 1000003 + int(c)) & 0xFFFFFFFF
        return acc % 50000


class _RealishLUT(dict):
    def __missing__(self, stage):
        s = _ClusterStage()
        self[stage] = s
        return s


def _flop_frontier(seed, n=3, lut=None):
    np.random.seed(seed)
    env = PokerEnv(players=[Player(i, 200) for i in range(n)],
                   low_card_rank=9, high_card_rank=14, small_blind=25, big_blind=50)
    env.card_info_lut = lut if lut is not None else collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0))
    g = 0
    while env.betting_round < 1 and not env.is_terminal and g < 20:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        g += 1
    return env


def _ctx_for(env, rng_seed=3, policy_cls=UniformPolicy):
    n = env.n_players
    leaf = LeafConfig(policies={c: policy_cls() for c in _BIAS_CLASSES})
    return SubgameContext.from_runtime(
        env=env, my_seat=0, my_hole=tuple(int(c) for c in env.players[0].cards),
        ranges={s: np.ones(env.n_combos, np.float32) / env.n_combos for s in range(n)},
        folded_ranges={}, leaf=leaf, rng=np.random.default_rng(rng_seed),
    )


@pytest.mark.parametrize("seed", range(20))
def test_refresh_clusters_matches_lut(seed):
    """After ``set_board(drawn) + refresh_clusters(lut)``, every (seat, street)
    cluster equals the LUT lookup ``PokerEnv`` would do for that board."""
    lut = _RealishLUT()
    env = _flop_frontier(seed, lut=lut)
    fs = FastState.from_poker_env(env)
    n = env.n_players
    # A drawn board: keep the flop prefix, invent turn+river from unused cards.
    prefix = [int(c) for c in env.community_cards]
    used = set(prefix)
    for p in env.players:
        used.update(int(c) for c in p.cards)
    deck = [int(c) for c in np.unique(env.combo_cards)]
    undealt = [c for c in deck if c not in used]
    rng = np.random.RandomState(seed)
    completion = list(rng.choice(undealt, size=5 - len(prefix), replace=False))
    board = prefix + [int(c) for c in completion]
    fs.set_board(board)
    fs.refresh_clusters(lut)

    holes = [[int(c) for c in p.cards] for p in env.players]
    board_len = (0, 3, 4, 5)
    names = ("pre_flop", "flop", "turn", "river")
    for si in range(4):
        blen = board_len[si]
        for seat in range(n):
            key = tuple(sorted(holes[seat]) + sorted(board[:blen]))
            expected = int(lut[names[si]][key])
            cid, has = fs.cluster_at(seat, si)
            assert has == 1 and cid == expected, (
                f"seed={seed} seat={seat} street={si}: cluster {cid} != {expected}"
            )


def test_policy_state_matches_env_at_frontier():
    """At the frontier node (board == frontier board), ``_policy_state(fs)`` matches
    ``PokerEnv.policy_state`` — same info_set (current actor), valid_mask, legal."""
    lut = _RealishLUT()
    for seed in range(30):
        env = _flop_frontier(seed, lut=lut)
        if env.is_terminal:
            continue
        # advance a little flop betting so the history/legal set is non-trivial
        if "call" in env.legal_actions:
            env.step_in_place("call")
        if env.is_terminal:
            continue
        fs = FastState.from_poker_env(env)
        fs.refresh_clusters(lut)  # frontier board == fs board → clusters == env's
        legal = [a for a in fs.legal_actions() if a is not None]
        ps = _policy_state(fs, legal)
        env_ps = env.policy_state           # env's current-actor PolicyState
        assert ps.player_i == env_ps.player_i
        assert ps.betting_round == env_ps.betting_round
        assert ps.legal_actions == env_ps.legal_actions
        assert np.array_equal(ps.valid_mask, env_ps.valid_mask)
        assert ps.info_set == env_ps.info_set, f"info_set seed={seed}"


def test_policy_state_matches_env_across_streets():
    """``_policy_state(fs)`` matches ``PokerEnv.policy_state`` at EVERY node of a
    multi-street rollout — not just the frontier.

    ``test_policy_state_matches_env_at_frontier`` only reaches the flop (+1 call), and
    the unbiasedness value gate uses ``UniformPolicy`` (blind to ``info_set``), so a
    wrong-but-valid info-set at a **turn/river** rollout node would slip through both.
    Here we drive a full check-down (so all five board cards are dealt and the
    FastState board set below equals ``env``'s community at every street) and assert
    ``info_set`` / ``valid_mask`` / ``legal`` agree at each flop-, turn- and
    river-street decision node.  This is the composition
    (``refresh_clusters`` on a drawn board + deep-street history) the value gate
    cannot see — the one that matters once a real ``BlueprintPolicy`` (Phase 4c) reads
    ``info_set``.
    """
    lut = _RealishLUT()
    saw_turn = saw_river = False
    for seed in range(25):
        env = _flop_frontier(seed, lut=lut)
        if env.is_terminal:
            continue
        # Record the env's per-node PolicyState down a pure check/call line (which
        # runs flop -> turn -> river -> showdown, dealing the whole board).
        roll = copy.deepcopy(env)
        line, nodes = [], []
        g = 0
        while not roll.is_terminal and g < 60:
            ps = roll.policy_state
            nodes.append((ps.player_i, ps.betting_round, ps.info_set,
                          ps.valid_mask.copy(), ps.legal_actions))
            a = "check" if "check" in roll.legal_actions else "call"
            line.append(a)
            roll.step_in_place(a)
            g += 1
        board = [int(c) for c in roll.community_cards]
        if len(board) != 5:
            continue  # a short (folded) line can't pin fs's per-street board
        fs = FastState.from_poker_env(env)
        fs.set_board(board)          # board[:blen] == env's community at each street
        fs.refresh_clusters(lut)
        for a, (p_i, rnd, iset, vmask, legal_env) in zip(line, nodes):
            legal_fs = [x for x in fs.legal_actions() if x is not None]
            ps = _policy_state(fs, legal_fs)
            assert ps.player_i == p_i
            assert ps.betting_round == rnd
            assert ps.legal_actions == legal_env
            assert np.array_equal(ps.valid_mask, vmask)
            assert ps.info_set == iset, f"info_set seed={seed} round={rnd}"
            saw_turn = saw_turn or rnd == 2
            saw_river = saw_river or rnd == 3
            fs.step_in_place(a)
    assert saw_turn and saw_river, "never reached turn/river rollout nodes — vacuous"


def test_vector_rollout_runs_and_falls_back():
    """The per-combo rollout returns a finite ``(n_combos,)`` vector; an off-tree
    (overlay) frontier falls back to the Python per-combo rollout."""
    env = _flop_frontier(1)
    if "call" in env.legal_actions:
        env.step_in_place("call")
    active = [s for s in range(env.n_players) if env.players[s].is_active]
    profile = {s: "none" for s in active}
    traverser = active[0]
    v = continuation_value_vector_fast(
        copy.deepcopy(env), profile, _ctx_for(env), traverser)
    assert v.shape == (env.n_combos,) and np.all(np.isfinite(v))

    env2 = copy.deepcopy(env)
    env2._extra_legal_actions[env2.public_key] = frozenset({"raise:1.5"})
    vfb = continuation_value_vector_fast(
        env2, profile, _ctx_for(env2), traverser)
    assert np.all(np.isfinite(vfb))


def test_rollout_vector_unbiased_vs_python():
    """The FastState per-combo leaf rollout is an **unbiased** estimator of the
    Python ``continuation_value_vector`` — the Phase-2 analogue of
    ``test_rollout_unbiased_vs_python``.

    Both sweep the traverser's whole range at the reached terminal (settled
    byte-identically by ``vector_payout_concrete``); they differ only in the board
    draw + RNG stream, so over many independent single-rollout calls the two
    per-combo means must agree within combined Monte-Carlo error.  A persistent
    per-combo gap would betray a scoring / policy / board-draw bias in the
    vectorized rollout.
    """
    env = _flop_frontier(7)
    if "call" in env.legal_actions:
        env.step_in_place("call")
    active = [s for s in range(env.n_players) if env.players[s].is_active]
    profile = {s: "none" for s in active}
    traverser = active[0]
    N = 1000  # independent single-rollout calls averaged per arm

    def estimate(fast, base_seed):
        vals = []
        for i in range(N):
            ctx = _ctx_for(env, rng_seed=base_seed * 100_000 + i)
            fn = continuation_value_vector_fast if fast else continuation_value_vector
            vals.append(fn(copy.deepcopy(env), profile, ctx, traverser))
        arr = np.array(vals)                       # (N, n_combos)
        return arr.mean(0), arr.std(0, ddof=1) / np.sqrt(N)

    m_fast, se_fast = estimate(True, 1)
    m_py, se_py = estimate(False, 2)
    sem = np.sqrt(se_fast ** 2 + se_py ** 2)
    gap = np.abs(m_fast - m_py)
    # Infeasible combos are exactly 0 in both (gap 0); feasible combos must match
    # within combined SE.  Self-calibrating slack, as the scalar twin.
    assert np.all(gap < 4.0 * sem + 0.5), (
        f"vector rollout biased vs python: max gap={float(gap.max()):.3f} "
        f"at combo {int(gap.argmax())} (sem there={float(sem[gap.argmax()]):.3f})"
    )


def test_vector_rollout_terminal_delegates_to_python():
    """An already-terminal frontier delegates to the Python per-combo reference."""
    env = _flop_frontier(2)
    ctx = _ctx_for(env)
    term = copy.deepcopy(env)
    g = 0
    while not term.is_terminal and g < 40:
        term.step_in_place("all_in" if "all_in" in term.legal_actions else "call")
        g += 1
    assert term.is_terminal
    profile = {s: "none" for s in range(term.n_players)}
    a = continuation_value_vector_fast(copy.deepcopy(term), profile, ctx, 0)
    b = continuation_value_vector(copy.deepcopy(term), profile, ctx, 0)
    assert np.array_equal(a, b)
