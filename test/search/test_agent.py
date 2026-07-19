"""Tests for the search-aware play agent (``SearchAgent``, §6.6, row 7).

These are **wiring / orchestration** tests: they assert the agent drives the
already-tested collaborators (`RangeTracker`, `solve`, the policy readers) in the
right order with the right arguments, not that those collaborators are internally
correct (covered by their own suites).  Small decks, ``UniformPolicy`` stand-ins
for the leaf fleet and blueprint (no real blueprint artifact), and a tiny
``max_iterations`` keep every case fast; the autouse ``_seeded`` fixture
(conftest.py) runs each across five RNG seeds for card-layout coverage.
"""

import copy
from unittest import mock

import numpy as np
import pytest

from environment.player import Player
from environment.poker_env import PokerEnv
import poker_ai.search.agent as agent_mod
from poker_ai.search.agent import SearchAgent
from poker_ai.search.leaf import LeafConfig
from poker_ai.search.solver import SolverConfig

from test.search.test_solver import UniformPolicy, _policies, _stub_lut


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _env(n_players=2, stacks=None, low=11, high=14) -> PokerEnv:
    stacks = stacks or [200] * n_players
    env = PokerEnv(
        players=[Player(i, s) for i, s in enumerate(stacks)],
        low_card_rank=low,
        high_card_rank=high,
    )
    _stub_lut(env)
    return env


def _cfg(leaf, iters=20, kappa=0.0) -> SolverConfig:
    # Default kappa=0 so the toy-search agent tests exercise the *pure search* read
    # path (a 20-iter toy search leaves most rows starved, so the production default
    # kappa would shrink nearly every read toward the blueprint and mask what these
    # tests check).  The shrinkage itself is covered by test_blueprint_prior_shrinkage.
    return SolverConfig(
        leaf=leaf, max_iterations=iters, max_wall_seconds=30.0, discount_interval=20,
        workers=1, blueprint_prior_kappa=kappa,
    )


def _agent(blueprint=None, kappa=0.0, **kw) -> SearchAgent:
    leaf = LeafConfig(policies=_policies(), n_rollouts=2)
    return SearchAgent(
        leaf_policies=leaf.policies,
        blueprint_policy=blueprint or UniformPolicy(),
        solver_cfg=_cfg(leaf, kappa=kappa),
        rng=np.random.default_rng(0),
        **kw,
    )


def _to_flop(env, agent=None, buffer=False):
    """Drive a heads-up env to the flop (calls/checks); return the new flop cards.

    With ``buffer=True`` each action is announced to ``agent.on_observed_action``
    (deepcopying the pre-action env, as a runner would) before being applied.
    """
    pre = {int(c) for c in env.community_cards}
    while env.betting_round < 1 and not env.is_terminal:
        action = "call" if "call" in env.legal_actions else "check"
        if buffer and agent is not None:
            agent.on_observed_action(copy.deepcopy(env), env.player_i, action)
        env.step_in_place(action)
    return [c for c in env.community_cards if int(c) not in pre]


def _advance_to_my_turn(env, seat, max_round):
    """Call/check opponents until ``seat`` is to act (or the round/hand ends)."""
    guard = 0
    while (
        env.player_i != seat
        and not env.is_terminal
        and env.betting_round == max_round
        and guard < 12
    ):
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1


def _inject_off_canonical(env) -> str:
    """Inject a *playable* raise whose fraction is off the canonical grid (as the
    runtime would for an off-abstraction opponent raise); return the action string."""
    canonical = {f"raise:{f}" for f in env.canonical_raise_fractions()}
    for x in (0.6, 0.61, 0.62, 0.137, 1.234, 1.7, 2.6):
        a = f"raise:{x}"
        if a not in canonical and env.inject_action(a):
            return a
    raise AssertionError("no playable off-canonical raise available at this node")


def _count_solves(monkeypatch):
    """Wrap ``agent.solve`` with a counter that delegates to the real solver;
    returns the per-call ``warm_start`` log (asserts orchestration without
    re-implementing the solver)."""
    real = agent_mod.solve
    log = []

    def wrapper(root_env, ctx, cfg, warm_start=None):
        log.append(warm_start)
        return real(root_env, ctx, cfg, warm_start=warm_start)

    monkeypatch.setattr(agent_mod, "solve", wrapper)
    return log


# --------------------------------------------------------------------------- #
# A. Lifecycle wiring
# --------------------------------------------------------------------------- #

def test_on_hand_start_resets_state(_seeded):
    env = _env()
    env.inject_action("raise:0.137")  # stray overlay from a "previous hand"
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)

    assert agent.my_seat == 0
    assert agent.my_hole == tuple(sorted(int(c) for c in env.players[0].cards))
    assert agent.last_search is None
    assert agent.pending_actions == []
    assert agent._searched_this_round is False
    assert agent._root_env is not None and agent._root_env.betting_round == 0
    assert env.has_overlay_at_current_node is False  # reset_overlay ran


def test_round1_act_returns_legal_blueprint_action(_seeded):
    env = _env()
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    _advance_to_my_turn(env, 0, 0)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act pre-flop this layout")

    action = agent.act(env)
    assert action in [a for a in env.legal_actions if a is not None]
    assert agent.last_search is None  # no search on round 1


def test_board_update_runs_search_then_act_freezes(_seeded):
    env = _env()
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    new = _to_flop(env)
    agent.on_board_update(env, new)

    assert agent.last_search is not None
    assert agent._root_env is not None and agent._ctx is not None
    assert agent._searched_this_round is True

    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")
    pk, hr = env.public_key, agent._hand_row(env)
    action = agent.act(env)
    assert action in [a for a in env.legal_actions if a is not None]
    assert (pk, hr) in agent.last_search.state.frozen


def test_act_rounds2plus_reads_search_not_blueprint(_seeded):
    blueprint = UniformPolicy()
    env = _env()
    agent = _agent(blueprint=blueprint)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")

    with mock.patch.object(
        agent.last_search.policy, "strategy_for", wraps=agent.last_search.policy.strategy_for
    ) as spy_policy, mock.patch.object(blueprint, "strategy", wraps=blueprint.strategy) as spy_bp:
        agent.act(env)
    assert spy_policy.called
    assert not spy_bp.called


def test_act_falls_back_to_blueprint_when_node_unsolved(_seeded):
    # A search ran, but the played node is NOT in the solved tree (a decision past a
    # depth-limit leaf, or an off-tree line the tree does not contain). The agent
    # must play the BLUEPRINT here, never a uniform guess over the legal actions.
    blueprint = UniformPolicy()
    env = _env()
    agent = _agent(blueprint=blueprint)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")
    assert agent.last_search is not None
    # Simulate "the search does not cover this node": drop every registered node so
    # the played public_key is absent from the solved tree.
    agent.last_search.state.legal_at.clear()

    legal, prob, searched, bp_w = agent.play_distribution(env)
    assert searched is False                       # blueprint fallback, not a search read
    assert bp_w == 1.0                             # a pure-blueprint play
    assert len(prob) == len(legal) and abs(float(np.sum(prob)) - 1.0) < 1e-6

    frozen_before = len(agent.last_search.state.frozen)
    with mock.patch.object(blueprint, "strategy", wraps=blueprint.strategy) as spy_bp:
        action = agent.act(env)
    assert spy_bp.called                           # played the blueprint
    assert action in [a for a in env.legal_actions if a is not None]
    # A fallback play pins nothing — there is no solved row to freeze.
    assert len(agent.last_search.state.frozen) == frozen_before


def _boom(*a, **k):
    raise RuntimeError("solve blew up")


def test_solve_failure_falls_back_to_blueprint(_seeded, monkeypatch):
    # A solve that raises for ANY reason must not crash the hand: the agent clears
    # last_search and plays the blueprint for the round.
    blueprint = UniformPolicy()
    env = _env()
    agent = _agent(blueprint=blueprint)
    agent.on_hand_start(env, my_seat=0)
    monkeypatch.setattr(agent_mod, "solve", _boom)
    agent.on_board_update(env, _to_flop(env))      # must not propagate the error
    assert agent.last_search is None

    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")
    _, _, searched, _ = agent.play_distribution(env)
    assert searched is False
    with mock.patch.object(blueprint, "strategy", wraps=blueprint.strategy) as spy_bp:
        action = agent.act(env)
    assert spy_bp.called
    assert action in [a for a in env.legal_actions if a is not None]


# --------------------------------------------------------------------------- #
# B. Boundary belief update
# --------------------------------------------------------------------------- #

def test_boundary_replay_calls_tracker_on_action(_seeded):
    env = _env()
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    with mock.patch.object(
        agent.tracker, "on_action", wraps=agent.tracker.on_action
    ) as spy_action, mock.patch.object(
        agent.tracker, "on_board_update", wraps=agent.tracker.on_board_update
    ) as spy_board:
        new = _to_flop(env, agent, buffer=True)
        n_buffered = len(agent.pending_actions)
        agent.on_board_update(env, new)

    assert spy_action.call_count == n_buffered
    assert spy_board.call_count == 1
    assert agent.pending_actions == []  # cleared after the boundary


def test_boundary_blueprint_branch_uses_blueprint(_seeded):
    blueprint = UniformPolicy()
    env = _env()
    agent = _agent(blueprint=blueprint)
    agent.on_hand_start(env, my_seat=0)
    new = _to_flop(env, agent, buffer=True)
    if not agent.pending_actions:
        pytest.skip("no pre-flop actions buffered this layout")
    with mock.patch.object(blueprint, "strategy", wraps=blueprint.strategy) as spy_bp:
        agent.on_board_update(env, new)
    # Round-1→2 boundary: no search ran, so the sigma closure is the blueprint.
    assert spy_bp.called


def test_boundary_search_branch_uses_average_policy(_seeded):
    blueprint = UniformPolicy()
    env = _env()
    agent = _agent(blueprint=blueprint)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))  # flop search now exists

    # Buffer a flop action, then cross to the turn.
    pre = {int(c) for c in env.community_cards}
    while env.betting_round == 1 and not env.is_terminal:
        action = "call" if "call" in env.legal_actions else "check"
        agent.on_observed_action(copy.deepcopy(env), env.player_i, action)
        env.step_in_place(action)
    if env.betting_round != 2 or not agent.pending_actions:
        pytest.skip("did not reach the turn with a buffered flop action")
    turn_new = [c for c in env.community_cards if int(c) not in pre]

    avg = agent.last_search.average_policy
    with mock.patch.object(
        avg, "strategy_for", wraps=avg.strategy_for
    ) as spy_avg, mock.patch.object(blueprint, "strategy", wraps=blueprint.strategy) as spy_bp:
        agent.on_board_update(env, turn_new)
    assert spy_avg.called
    assert not spy_bp.called


# --------------------------------------------------------------------------- #
# B'. Blueprint-prior shrinkage (§6.6) — low-mass rows fall back to the blueprint
# --------------------------------------------------------------------------- #

def test_blueprint_prior_shrinkage_blends_by_mass(_seeded):
    # The played σ = (mass·search + kappa·blueprint)/(mass + kappa): pure blueprint at
    # mass 0, pure search as mass ⇒ ∞, an even mix at mass == kappa.
    agent = _agent(kappa=4.0)
    search = np.array([1.0, 0.0])          # search fully commits to action 0
    bp = np.array([0.0, 1.0])              # blueprint fully commits to action 1
    calls = {"n": 0}

    def bp_fn():
        calls["n"] += 1
        return bp

    out_lo, w_lo = agent._shrink_to_blueprint(search, 0.0, bp_fn)
    assert w_lo == pytest.approx(1.0)                 # kappa/(0+kappa) = 1
    assert out_lo == pytest.approx([0.0, 1.0])        # → the blueprint

    out_mid, w_mid = agent._shrink_to_blueprint(search, 4.0, bp_fn)
    assert w_mid == pytest.approx(0.5)                # kappa/(kappa+kappa)
    assert out_mid == pytest.approx([0.5, 0.5])

    # A well-trained row: blueprint weight < 0.1% → skipped entirely (lazy: no fetch).
    fetched_before = calls["n"]
    out_hi, w_hi = agent._shrink_to_blueprint(search, 40_000.0, bp_fn)
    assert w_hi == 0.0
    assert out_hi == pytest.approx([1.0, 0.0])        # → the search, untouched
    assert calls["n"] == fetched_before               # blueprint thunk not called


def test_blueprint_prior_kappa_zero_disables_shrinkage(_seeded):
    agent = _agent(kappa=0.0)
    search = np.array([1.0, 0.0])
    out, w = agent._shrink_to_blueprint(
        search, 0.0, lambda: (_ for _ in ()).throw(AssertionError("must not fetch"))
    )
    assert w == 0.0
    assert out == pytest.approx([1.0, 0.0])


def test_play_distribution_reports_blueprint_weight(_seeded):
    # A covered (searched) decision reports the exact shrinkage weight from the node's
    # mass, so the evaluation can log how much of the play was blueprint prior.
    env = _env()
    agent = _agent(kappa=5.0)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")

    legal, prob, searched, bp_w = agent.play_distribution(env)
    assert searched is True
    assert 0.0 <= bp_w <= 1.0
    assert abs(float(prob.sum()) - 1.0) < 1e-6
    # The reported weight is exactly kappa/(mass+kappa) (or 0 when negligible).
    pk = agent._solved_public_key(env)
    hr = agent._hand_row(env)
    mass = agent.last_search.policy.mass(pk, hr)
    w_expected = 5.0 / (mass + 5.0)
    assert bp_w == pytest.approx(w_expected if w_expected >= 1e-3 else 0.0)


def test_mass_zero_for_unseen_key(_seeded):
    env = _env()
    agent = _agent(kappa=5.0)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    pol = agent.last_search.policy
    assert pol.mass(("no-such-public-key",), 0) == 0.0


def test_fold_moves_seat_to_folded_after_replay(_seeded):
    env = _env(n_players=3, stacks=[300, 300, 300])
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)

    # Find an opponent to act pre-flop and have them fold.
    _advance_to_my_turn(env, 1, 0)
    if env.is_terminal or env.player_i != 1 or "fold" not in env.legal_actions:
        pytest.skip("seat 1 not able to fold pre-flop this layout")
    folder = 1
    calls = []
    real_on_action = agent.tracker.on_action
    real_on_folded = agent.tracker.on_seat_folded

    def rec_action(*a, **k):
        calls.append("action")
        return real_on_action(*a, **k)

    def rec_folded(*a, **k):
        calls.append("folded")
        return real_on_folded(*a, **k)

    agent.on_observed_action(copy.deepcopy(env), folder, "fold")
    env.step_in_place("fold")
    new = _to_flop(env)
    with mock.patch.object(agent.tracker, "on_action", side_effect=rec_action), \
         mock.patch.object(agent.tracker, "on_seat_folded", side_effect=rec_folded):
        agent.on_board_update(env, new)

    assert folder not in agent.tracker.snapshot()
    assert folder in agent.tracker.folded_snapshot()
    # on_action (replay the fold) precedes on_seat_folded → post-fold posterior.
    assert calls.index("action") < calls.index("folded")


# --------------------------------------------------------------------------- #
# C. Freezing across a re-search
# --------------------------------------------------------------------------- #

def test_act_writes_frozen_row_aligned(_seeded):
    env = _env()
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")

    pk, hr = env.public_key, agent._hand_row(env)
    legal = [a for a in env.legal_actions if a is not None]
    agent.act(env)
    row = agent.last_search.state.frozen[(pk, hr)]
    assert row.shape == (len(legal),)
    assert abs(float(row.sum()) - 1.0) < 1e-6


def test_frozen_survives_off_tree_research(_seeded, monkeypatch):
    env = _env(stacks=[1000, 1000])
    agent = _agent(offtree_threshold=0.0)  # isolate the re-search mechanism from the gap gate
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    _advance_to_my_turn(env, 0, 1)
    if env.is_terminal or env.player_i != 0:
        pytest.skip("bot not to act on the flop this layout")
    pk, hr = env.public_key, agent._hand_row(env)
    agent.act(env)
    frozen_before = agent.last_search.state.frozen[(pk, hr)].copy()
    state_before = agent.last_search.state

    log = _count_solves(monkeypatch)
    # An opponent makes a genuine off-canonical raise (injected into the tree).
    eb = copy.deepcopy(env)
    off = _inject_off_canonical(eb)
    agent.on_observed_action(eb, 1, off)

    assert len(log) == 1                       # exactly one re-search
    assert log[0] is state_before              # warm-started with the prior state
    assert agent.last_search.state is state_before
    # The freeze survives: the originally-pinned probabilities are intact. (If the
    # injection widened the bot's own node, the row just gains zero-mass columns
    # for the new action.)
    after = agent.last_search.state.frozen[(pk, hr)]
    assert np.array_equal(after[: len(frozen_before)], frozen_before)
    assert np.all(after[len(frozen_before):] == 0.0)


# --------------------------------------------------------------------------- #
# D. Off-tree re-search
# --------------------------------------------------------------------------- #

def test_is_off_tree_classification(_seeded):
    env = _env(stacks=[1000, 1000])
    agent = _agent()
    # Advance to a node that has a non-empty canonical raise set.
    guard = 0
    while not env.canonical_raise_fractions() and not env.is_terminal and guard < 8:
        env.step_in_place("call" if "call" in env.legal_actions else "check")
        guard += 1
    # An off-canonical raise is off-tree; canonical raises and the always-legal
    # actions are on-tree.
    canonical = {f"raise:{f}" for f in env.canonical_raise_fractions()}
    off = next(f"raise:{x}" for x in (0.137, 0.61, 1.234) if f"raise:{x}" not in canonical)
    assert agent._is_off_tree(env, off) is True
    for on_tree in ("fold", "call", "check", "all_in", "raise"):  # bare "raise": not off-tree
        assert agent._is_off_tree(env, on_tree) is False
    for a in canonical:
        assert agent._is_off_tree(env, a) is False


def test_off_tree_triggers_research_on_tree_does_not(_seeded, monkeypatch):
    # 1000 stacks gives a non-empty canonical raise set.
    log = _count_solves(monkeypatch)
    env = _env(stacks=[1000, 1000])
    agent = _agent(offtree_threshold=0.0)  # any off-tree raise re-searches (gate off)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))   # boundary solve
    base = len(log)

    # On-tree opponent action → no re-search.
    agent.on_observed_action(copy.deepcopy(env), 1, "call")
    assert len(log) == base
    # An off-canonical raise that was NOT injected (not in the tree) → no re-search.
    agent.on_observed_action(copy.deepcopy(env), 1, "raise:0.6")
    assert len(log) == base
    # A genuine off-canonical raise the runtime injected → exactly one re-search.
    eb = copy.deepcopy(env)
    off = _inject_off_canonical(eb)
    agent.on_observed_action(eb, 1, off)
    assert len(log) == base + 1
    assert log[-1] is agent.last_search.state


@pytest.mark.parametrize("threshold,expect_research", [(10.0, 0), (0.0, 1)])
def test_offtree_gap_gate(_seeded, monkeypatch, threshold, expect_research):
    """Rounds 2-4: the *same* injected off-canonical raise is translated (no
    re-search) under a large ``offtree_threshold`` and re-searched under a zero
    one — the pot-fraction gap gate deciding inject-vs-translate, independent of
    the specific canonical grid at this layout.
    """
    log = _count_solves(monkeypatch)
    env = _env(stacks=[1000, 1000])
    agent = _agent(offtree_threshold=threshold)
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))   # boundary solve
    base = len(log)
    eb = copy.deepcopy(env)
    off = _inject_off_canonical(eb)
    agent.on_observed_action(eb, 1, off)
    assert len(log) == base + expect_research
    # A translated raise leaves the prior search in place (nothing to warm-start).
    if not expect_research:
        assert agent.last_search is not None


# --------------------------------------------------------------------------- #
# E. act() always returns a legal action (smoke, HU + multiway)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_players", [2, 3])
def test_act_always_legal(_seeded, n_players):
    # End-to-end act()-legality across HU + multiway, with the real solver.
    env = _env(n_players=n_players, stacks=[300] * n_players)
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)

    # Round 1.
    _advance_to_my_turn(env, 0, 0)
    if not env.is_terminal and env.player_i == 0 and env.betting_round == 0:
        assert agent.act(env) in [a for a in env.legal_actions if a is not None]

    # Drive to the flop and search.
    new = _to_flop(env)
    if env.is_terminal:
        pytest.skip("hand ended pre-flop this layout")
    agent.on_board_update(env, new)
    _advance_to_my_turn(env, 0, 1)
    if not env.is_terminal and env.player_i == 0 and env.betting_round == 1:
        assert agent.act(env) in [a for a in env.legal_actions if a is not None]


# --------------------------------------------------------------------------- #
# F. Round-1 trigger (pot-relative fraction-gap)
# --------------------------------------------------------------------------- #

def test_round1_trigger_fires_far_off_tree(_seeded):
    env = _env(stacks=[1000, 1000])
    agent = _agent(round1_offtree_threshold=0.25, round1_max_players=4)
    agent.on_hand_start(env, my_seat=0)
    # An opponent makes a wildly off-abstraction pre-flop raise (far from every
    # canonical size).
    env_before = copy.deepcopy(env)
    agent.on_observed_action(env_before, 1, "raise:9.0")

    assert agent.last_search is not None
    assert agent._searched_this_round is True
    assert agent._ctx.depth_limit.street_at_root == 0


def test_round1_trigger_skips_when_too_many_players(_seeded):
    env = _env(n_players=5, stacks=[1000] * 5)
    agent = _agent(round1_max_players=4)
    agent.on_hand_start(env, my_seat=0)
    agent.on_observed_action(copy.deepcopy(env), 1, "raise:9.0")
    assert agent.last_search is None   # > max_players live → no round-1 search


def test_round1_trigger_skips_near_abstraction(_seeded):
    env = _env(stacks=[1000, 1000])
    agent = _agent(round1_offtree_threshold=5.0)  # nothing is "far" under this τ
    agent.on_hand_start(env, my_seat=0)
    agent.on_observed_action(copy.deepcopy(env), 1, "raise:0.51")
    assert agent.last_search is None


def test_search_round1_mechanism(_seeded):
    env = _env(stacks=[1000, 1000])
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    agent.search_round1()
    assert agent.last_search is not None
    assert agent._searched_this_round is True
    assert agent._ctx.depth_limit.street_at_root == 0


# --------------------------------------------------------------------------- #
# G. Own-action / fold guards (bugs 2 & 3, fold dormancy)
# --------------------------------------------------------------------------- #

def test_bot_own_off_tree_action_does_not_research(_seeded, monkeypatch):
    env = _env(stacks=[1000, 1000])
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    agent.on_board_update(env, _to_flop(env))
    log = _count_solves(monkeypatch)
    # Even a (hypothetical) off-canonical raise attributed to the bot's own seat
    # must not trigger a re-search — re-search reacts to opponents only.
    eb = copy.deepcopy(env)
    off = _inject_off_canonical(eb)
    agent.on_observed_action(eb, agent.my_seat, off)
    assert len(log) == 0


def test_bot_fold_makes_agent_dormant(_seeded, monkeypatch):
    env = _env(n_players=3, stacks=[300, 300, 300])
    agent = _agent()
    agent.on_hand_start(env, my_seat=0)
    log = _count_solves(monkeypatch)
    # Bot folds; the agent goes dormant — later board updates run no search.
    agent.on_observed_action(copy.deepcopy(env), agent.my_seat, "fold")
    assert agent._folded is True
    agent.on_board_update(env, _to_flop(env))
    assert len(log) == 0
    assert agent.last_search is None
