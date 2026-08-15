# cython: language_level=3
"""Compiled betting-state engine (Phase 2) — Cython port of
``poker_ai._core._state_ref.FastStateRef`` (which is itself the audited reduction
of ``environment.poker_env.PokerEnv``'s blueprint-training betting engine).

``FastState`` is a ``cdef class`` holding the whole betting state in **flat C
arrays** (no ``Player`` / ``Pot`` / ``Deck`` / history-dict Python objects on the
hot path) with make/undo driven by an explicit **C POD stack** (``memcpy`` of a
fixed struct, not a dataclass).  It is byte-for-byte equivalent to ``PokerEnv`` /
``FastStateRef`` on the blueprint contract — ``player_i`` / ``legal_actions`` /
``is_terminal`` / ``info_set`` bytes / ``payout`` at **every** node — which the
Phase-2 differential fuzz proves against both oracles.

Design notes (why this is Phase-3 ready, not throwaway):

* **History is stored as resolved action *byte-codes*** in flat C arrays
  (``hist[stage][k]`` + ``hist_n[stage]``), not Python strings.  The code for an
  action is resolved **once, at step time**, against the dumped ``_ACTION_BYTE``
  alphabet (one dict lookup per *step* — cheap), so :meth:`info_set` is a pure-C
  ``memcpy`` walk with **no per-token dict lookup per node** (the old
  ``encode_info_set`` did an ``O(history)`` re-encode *per node*).  This is the
  exact byte stream ``encode_info_set`` produces, so the key is identical.
* **The alphabet / raise grid are DUMPED from ``poker_env`` via :func:`configure`,
  never hard-coded** — the same anti-drift discipline as the Phase-1 kernels (a
  hard-coded copy would silently diverge when ``RAISE_SIZES_BY_STAGE`` changes →
  keys that hash differently from what the tables were written under).
* **make/undo restores counts, not history data.**  Within one ``_apply`` only the
  *entry* stage's history grows (by ``skip_counter + 1`` entries); the POD frame
  snapshots ``hist_n`` (append cursor) — the stale data past the cursor is simply
  overwritten by the next append, so no data copy is needed.
* Settlement (:meth:`payout`) is a **separate value layer** (Python evaluator +
  ``Pot.compute_utility``, identical to ``FastStateRef.payout``); it is only
  touched at terminal leaves and is already covered by the Phase-1d/1e kernels, so
  Phase 2 leaves it in Python and Phase 3 may swap in the kernels if it profiles.
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy
from libc.math cimport ceil
from cpython.bytes cimport PyBytes_FromStringAndSize

import itertools

import numpy as np

from environment import range_showdown
from environment.evaluator import default_evaluator
from environment.player import Player
from environment.pot import Pot
from environment.poker_env import _as_runout, _n_choose_k, _settle_runout, _settle_traverser
from environment.utils import enumerate_combos
from information_abstraction.lookup import MemmapLookup


# ---------------------------------------------------------------------------
# Compile-time bounds (loud RuntimeError on overflow — never silent truncation)
# ---------------------------------------------------------------------------
DEF MAX_PLAYERS = 32
DEF MAX_STAGE_ACTIONS = 2048   # history entries per betting round (skips + actions)
DEF MAX_FRACS = 8              # raise fractions per stage in the grid
DEF N_DECISION_STAGES = 4      # pre_flop, flop, turn, river (rounds 0..3)

# Internal stage enumeration (canonical play order), mirroring the PokerEnv
# ``_betting_stage`` strings.  0..3 are decision rounds; 4/5 are terminal.
DEF ST_PRE = 0
DEF ST_FLOP = 1
DEF ST_TURN = 2
DEF ST_RIVER = 3
DEF ST_SHOWDOWN = 4
DEF ST_TERMINAL = 5

# Community cards visible at each stage (board is external / precomputed).
cdef int _BOARD_LEN[6]
_BOARD_LEN[:] = [0, 3, 4, 5, 5, 5]


# ---------------------------------------------------------------------------
# Process-wide config dumped from poker_env (see configure()).  The alphabet and
# raise grid are constant per process, so they live at module scope shared by all
# FastState instances — never hard-coded here.
# ---------------------------------------------------------------------------
cdef bint _configured = False
cdef int _MAX_RAISES = 0
cdef int _STAGE_BYTE[N_DECISION_STAGES]      # internal stage idx -> _STAGE_ID byte
cdef object _ACTION_CODE = None              # list[4] of {token_str: int code}
cdef object _CODE_ACTION = None              # list[4] of {int code: token_str} (public_key)
cdef int _FIRST_N[N_DECISION_STAGES]
cdef int _SUB_N[N_DECISION_STAGES]
cdef double _FIRST_FRAC[N_DECISION_STAGES][MAX_FRACS]
cdef double _SUB_FRAC[N_DECISION_STAGES][MAX_FRACS]
cdef object _FIRST_STR = None                # list[4] of list[str] "raise:<f>"
cdef object _SUB_STR = None

_STAGE_NAMES = ("pre_flop", "flop", "turn", "river")


def configure(stage_id, action_byte, raise_sizes_by_stage, max_raises):
    """Install the encoding alphabet + raise grid dumped from ``poker_env``.

    Call once at wire time with the live ``_STAGE_ID`` / ``_ACTION_BYTE`` /
    ``RAISE_SIZES_BY_STAGE`` / ``MAX_RAISES_PER_ROUND`` — never a hard-coded copy
    (they derive from the raise grid and would silently drift otherwise).
    """
    global _configured, _MAX_RAISES, _ACTION_CODE, _CODE_ACTION, _FIRST_STR, _SUB_STR
    cdef int si, k
    cdef double f
    _MAX_RAISES = int(max_raises)
    _ACTION_CODE = [dict(action_byte[name]) for name in _STAGE_NAMES]
    # Inverse map (code -> token) for public_key reconstruction — the byte-codes
    # in ``hist`` are the same ones PokerEnv appends to ``_history`` (skip=0), so
    # inverting the alphabet rebuilds the exact string history public_key needs.
    _CODE_ACTION = [
        {code: token for token, code in (<dict>tbl).items()} for tbl in _ACTION_CODE
    ]
    _FIRST_STR = [[] for _ in range(N_DECISION_STAGES)]
    _SUB_STR = [[] for _ in range(N_DECISION_STAGES)]
    for si in range(N_DECISION_STAGES):
        name = _STAGE_NAMES[si]
        _STAGE_BYTE[si] = int(stage_id[name])
        cfg = raise_sizes_by_stage.get(name, {})
        first = list(cfg.get("first_raise", [1.0]))
        sub = list(cfg.get("subsequent_raise", [1.0]))
        if len(first) > MAX_FRACS or len(sub) > MAX_FRACS:
            raise RuntimeError("raise grid exceeds MAX_FRACS — raise the bound")
        _FIRST_N[si] = len(first)
        _SUB_N[si] = len(sub)
        for k in range(len(first)):
            f = float(first[k])
            _FIRST_FRAC[si][k] = f
            # f"raise:{f}" — the identical formatting poker_env / FastStateRef use.
            (<list>_FIRST_STR[si]).append("raise:{}".format(f))
        for k in range(len(sub)):
            f = float(sub[k])
            _SUB_FRAC[si][k] = f
            (<list>_SUB_STR[si]).append("raise:{}".format(f))
    _configured = True


def is_configured():
    return _configured


# ---------------------------------------------------------------------------
# make/undo POD frame — the full mutable betting state (fixed struct, memcpy'd).
# hist DATA is NOT snapshotted (append-only; restored via the hist_n cursor).
# ---------------------------------------------------------------------------
ctypedef struct FrameT:
    int betting_stage
    int player_i_index
    int n_raises
    int n_actions
    int skip_counter
    long last_raise_amount
    int n_players_started_round
    int terminal_board_len
    long n_chips[MAX_PLAYERS]
    long n_bet_chips[MAX_PLAYERS]
    int is_active[MAX_PLAYERS]
    long pot_chips[MAX_PLAYERS]
    int hist_n[N_DECISION_STAGES]


cdef class FastState:
    """Flat-array betting engine equivalent to ``FastStateRef`` / ``PokerEnv``.

    Construct via :meth:`from_poker_env` (Phase-2 differential harness) — the deal
    and per-(seat, round) cluster precompute stay in Python; this engine owns only
    the betting transitions.
    """

    # config (immutable after construction)
    # ``readonly`` so the search FastEnvAdapter can read the seat count (Phase 3).
    cdef readonly int n_players
    cdef long small_blind
    cdef long big_blind
    cdef int player_i_lut[6][MAX_PLAYERS]
    cdef int hole[MAX_PLAYERS][2]
    cdef int board[5]
    cdef int order[MAX_PLAYERS]
    cdef int clusters[MAX_PLAYERS][N_DECISION_STAGES]
    cdef int has_cluster[MAX_PLAYERS][N_DECISION_STAGES]

    # mutable betting state
    cdef long n_chips[MAX_PLAYERS]
    cdef long n_bet_chips[MAX_PLAYERS]
    cdef int is_active[MAX_PLAYERS]
    cdef long pot_chips[MAX_PLAYERS]
    cdef int betting_stage
    cdef int player_i_index
    cdef int n_raises
    cdef int n_actions
    cdef int skip_counter
    cdef long last_raise_amount
    # ``readonly`` so the search DepthLimit.classify / FastEnvAdapter can read it
    # as a plain Python attribute (Phase 3); still written at C level internally.
    cdef readonly int n_players_started_round
    # Board length (community-card count) when the hand became terminal, before
    # any force-deal — the search vector_payout fold path masks against it.  -1
    # (== Python ``None``) until terminal.  Snapshotted in FrameT for undo.
    cdef int _terminal_board_len
    # Deck bounds (search vector_payout: removal / ranked_board / board mask).
    cdef int _low_card_rank
    cdef int _high_card_rank

    # history: resolved action byte-codes, per decision round (append-only cursor)
    cdef unsigned char hist[N_DECISION_STAGES][MAX_STAGE_ACTIONS]
    cdef int hist_n[N_DECISION_STAGES]

    # make/undo POD stack
    cdef FrameT* _stack
    cdef int _stack_top
    cdef int _stack_cap

    def __cinit__(self):
        if not _configured:
            raise RuntimeError(
                "poker_ai._core._state.FastState used before configure() — the "
                "action alphabet + raise grid must be dumped from poker_env first."
            )
        # Start small so the realloc growth path is exercised by ordinary play
        # (a root-to-leaf line is tens of steps deep) rather than being dead code
        # only a pathological >64-deep 6-max line would ever reach.  The stack
        # reaches its steady-state depth after a few doublings, one-time cost.
        self._stack_cap = 8
        self._stack = <FrameT*>malloc(self._stack_cap * sizeof(FrameT))
        if self._stack == NULL:
            raise MemoryError()
        self._stack_top = 0
        self._terminal_board_len = -1
        self._low_card_rank = 0
        self._high_card_rank = 0

    def __dealloc__(self):
        if self._stack != NULL:
            free(self._stack)

    # ------------------------------------------------------------------
    # Construction from a live PokerEnv (Python owns the deal + clusters)
    # ------------------------------------------------------------------
    @staticmethod
    def from_poker_env(env):
        """Snapshot a live ``PokerEnv`` into an equivalent flat ``FastState``.

        Mirrors ``FastStateRef.from_poker_env`` exactly: board is the
        line-independent ``deck[2n:2n+5]``; clusters are precomputed per
        (seat, round) from ``env.card_info_lut``.
        """
        cdef FastState s = FastState()
        cdef int n = env.n_players
        if n > MAX_PLAYERS:
            raise RuntimeError("n_players exceeds MAX_PLAYERS")
        s.n_players = n
        s.small_blind = int(env.small_blind)
        s.big_blind = int(env.big_blind)
        # Kernel API boundary: read via PokerEnv's public properties, never its
        # private attributes directly (the adapter marshals values in; a private
        # attribute can gain normalization/clamping logic that a private read would
        # silently bypass, drifting from PokerEnv without any byte-identity test
        # catching it since a private-vs-private comparison would still agree).
        s._low_card_rank = <int>env.low_card_rank
        s._high_card_rank = <int>env.high_card_rank
        # ``None`` at a live decision-node root (the only place a solve constructs
        # a FastState); mirror PokerEnv's captured value defensively otherwise.
        tbl = env.terminal_board_len
        s._terminal_board_len = -1 if tbl is None else <int>tbl

        cdef int seat, k, si
        # Per-stage seat permutation (mirrors PokerEnv._player_i_lut / FastStateRef).
        base = list(range(n))
        postflop = base[::-1] if n == 2 else base
        preflop = base[2:] + base[:2]
        lut_by_stage = [preflop, postflop, postflop, postflop, postflop, postflop]
        for si in range(6):
            perm = lut_by_stage[si]
            for k in range(n):
                s.player_i_lut[si][k] = <int>perm[k]

        # Hole cards + board (the 5-card runout, read via the deck's layout
        # accessor rather than a hand-rolled 2n offset; line-independent).
        board = [int(c) for c in env.deck.board_runout(5)]
        for k in range(5):
            s.board[k] = <int>board[k]
        holes = [[int(c) for c in p.cards] for p in env.players]
        for seat in range(n):
            s.hole[seat][0] = <int>holes[seat][0]
            s.hole[seat][1] = <int>holes[seat][1]
            s.order[seat] = <int>env.players[seat].order

        # Clusters per (seat, round) — same key PokerEnv._compute_info_set uses.
        for seat in range(n):
            for si in range(N_DECISION_STAGES):
                s.clusters[seat][si] = 0
                s.has_cluster[seat][si] = 0
        for si in range(N_DECISION_STAGES):
            name = _STAGE_NAMES[si]
            blen = _BOARD_LEN[si]
            try:
                stage_lut = env.card_info_lut[name]
            except (KeyError, TypeError):
                continue
            for seat in range(n):
                key = tuple(sorted(holes[seat]) + sorted(board[:blen]))
                try:
                    s.clusters[seat][si] = <int>stage_lut[key]
                    s.has_cluster[seat][si] = 1
                except (KeyError, IndexError):
                    pass

        # Mutable betting state.
        for seat in range(n):
            s.n_chips[seat] = <long>env.players[seat].n_chips
            s.n_bet_chips[seat] = <long>env.players[seat].n_bet_chips
            s.is_active[seat] = 1 if env.players[seat].is_active else 0
            s.pot_chips[seat] = <long>env.pot._chips[seat]
        stage_name = env._betting_stage
        s.betting_stage = _STAGE_NAMES.index(stage_name) if stage_name in _STAGE_NAMES else (
            ST_SHOWDOWN if stage_name == "show_down" else ST_TERMINAL)
        s.player_i_index = <int>env._player_i_index
        s.n_raises = <int>env._n_raises
        s.n_actions = <int>env._n_actions
        s.skip_counter = <int>env._skip_counter
        s.last_raise_amount = <long>env._last_raise_amount
        s.n_players_started_round = <int>env._n_players_started_round

        # History: resolve each existing token to its byte-code (rounds 0..3).
        for si in range(N_DECISION_STAGES):
            s.hist_n[si] = 0
        for name, actions in env._history.items():
            if name not in _STAGE_NAMES:
                continue
            si = _STAGE_NAMES.index(name)
            table = <dict>_ACTION_CODE[si]
            for token in actions:
                s._push_hist_code(si, s._resolve_code(table, token))
        return s

    def clone(self):
        """Return an independent deep copy of this betting state (search Phase 3).

        Copies every config + mutable betting field + resolved history into a fresh
        ``FastState`` with its **own empty make/undo stack**, so a leaf rollout (or a
        parallel branch) can make/undo-walk the copy without disturbing this one.
        The clone shares no mutable state with ``self``.  Used by the MCCFR walk to
        hand the depth-limit leaf a private frontier it can draw boards on and
        walk to terminals independently of the parent traversal.
        """
        cdef FastState c = FastState.__new__(FastState)
        cdef int i, j, s
        c.n_players = self.n_players
        c.small_blind = self.small_blind
        c.big_blind = self.big_blind
        c._low_card_rank = self._low_card_rank
        c._high_card_rank = self._high_card_rank
        c.betting_stage = self.betting_stage
        c.player_i_index = self.player_i_index
        c.n_raises = self.n_raises
        c.n_actions = self.n_actions
        c.skip_counter = self.skip_counter
        c.last_raise_amount = self.last_raise_amount
        c.n_players_started_round = self.n_players_started_round
        c._terminal_board_len = self._terminal_board_len
        for i in range(6):
            for j in range(MAX_PLAYERS):
                c.player_i_lut[i][j] = self.player_i_lut[i][j]
        for s in range(MAX_PLAYERS):
            c.hole[s][0] = self.hole[s][0]
            c.hole[s][1] = self.hole[s][1]
            c.order[s] = self.order[s]
            c.n_chips[s] = self.n_chips[s]
            c.n_bet_chips[s] = self.n_bet_chips[s]
            c.is_active[s] = self.is_active[s]
            c.pot_chips[s] = self.pot_chips[s]
            for j in range(N_DECISION_STAGES):
                c.clusters[s][j] = self.clusters[s][j]
                c.has_cluster[s][j] = self.has_cluster[s][j]
        for i in range(5):
            c.board[i] = self.board[i]
        for i in range(N_DECISION_STAGES):
            c.hist_n[i] = self.hist_n[i]
            for j in range(self.hist_n[i]):
                c.hist[i][j] = self.hist[i][j]
        return c

    def set_board(self, board):
        """Overwrite the 5-card board (Phase 4b leaf rollout).

        The betting engine is board-independent, so a rollout builds the FastState
        once from the frontier and draws a fresh 5-card runout per call — this
        installs it.  ``board`` is the full 5 cards (prefix + drawn completion);
        only ``payout`` / ``runout_equity`` ranking read it.  Cluster caches
        (``clusters``) are NOT recomputed here — a UniformPolicy rollout ignores
        them; the in-core BlueprintPolicy path (4c) recomputes per drawn board.
        """
        cdef int k
        if len(board) != 5:
            raise ValueError("set_board expects exactly 5 cards")
        for k in range(5):
            self.board[k] = <int>board[k]

    def refresh_clusters(self, card_info_lut):
        """Recompute per-(seat, street) LUT clusters from the CURRENT board.

        The Phase-4b leaf rollout draws a fresh board per call (``set_board``), so
        the clusters ``info_set`` embeds must be re-derived for that board — same
        key ``PokerEnv._compute_info_set`` / ``from_poker_env`` use
        (``sorted(hole) + sorted(board[:street_len])``).  A missing key leaves the
        (seat, street) cluster unset (``has_cluster == 0``), exactly as
        ``from_poker_env`` swallows a precompute miss; ``info_set`` then raises loud
        at a decision node with no cluster (never a silent cluster-0).
        """
        cdef int seat, si, blen
        cdef Py_ssize_t n = self.n_players
        board = [self.board[seat] for seat in range(5)]
        for seat in range(self.n_players):
            for si in range(N_DECISION_STAGES):
                self.clusters[seat][si] = 0
                self.has_cluster[seat][si] = 0
        holes = np.array(
            [[self.hole[seat][0], self.hole[seat][1]] for seat in range(n)],
            dtype=np.int64,
        )
        for si in range(N_DECISION_STAGES):
            name = _STAGE_NAMES[si]
            blen = _BOARD_LEN[si]
            try:
                stage_lut = card_info_lut[name]
            except (KeyError, TypeError):
                continue
            if isinstance(stage_lut, MemmapLookup):
                board_arr = np.array(board[:blen], dtype=np.int64)
                ids = stage_lut.seat_batch_lookup(holes, board_arr)
                for seat in range(n):
                    cid = ids[seat]
                    if cid >= 0:
                        self.clusters[seat][si] = <int>cid
                        self.has_cluster[seat][si] = 1
            else:
                for seat in range(self.n_players):
                    key = tuple(
                        sorted([self.hole[seat][0], self.hole[seat][1]])
                        + sorted(board[:blen])
                    )
                    try:
                        self.clusters[seat][si] = <int>stage_lut[key]
                        self.has_cluster[seat][si] = 1
                    except (KeyError, IndexError):
                        pass

    # ------------------------------------------------------------------
    # Derived read-only contract (mirrors FastStateRef / PokerEnv)
    # ------------------------------------------------------------------
    cdef inline int _cur_seat(self):
        return self.player_i_lut[self.betting_stage][self.player_i_index]

    @property
    def player_i(self):
        return self._cur_seat()

    @property
    def is_terminal(self):
        return self.betting_stage == ST_SHOWDOWN or self.betting_stage == ST_TERMINAL

    cpdef bint is_seat_active(self, int seat):
        """True iff *seat* is still active (has not folded).

        The Phase-3 in-core traverse needs this to reproduce the Python
        ``is_terminal(state, i)`` short-circuit (``not players[i].is_active``)
        exactly — that early return decides how many opponent nodes are
        visited below a folded traversing player, and therefore how many
        replay/sample draws the traversal consumes.  ``cpdef`` so the pure-C
        traverse and Python differential tests share one implementation.
        """
        return self.is_active[seat] != 0

    @property
    def betting_round(self):
        return self.betting_stage

    def hole_cards(self, int seat):
        """The two concrete hole cards dealt to ``seat`` (search Phase 3 adapter)."""
        return (self.hole[seat][0], self.hole[seat][1])

    def community_cards(self):
        """Board cards public at the current street — the prefix of the 5-card board.

        ``_BOARD_LEN[betting_stage]`` gives the public count (0 pre-flop, 3 flop,
        4 turn, 5 river/terminal), so this equals ``PokerEnv.community_cards`` at
        every node — the MCCFR adapter reads it for the leaf frontier prefix and
        card-removal masks, and it returns the full five at a terminal (as the
        concrete settlement expects).
        """
        cdef int blen = _BOARD_LEN[self.betting_stage]
        cdef int k
        return [self.board[k] for k in range(blen)]

    # ------------------------------------------------------------------
    # Search public-state surface (Phase 3) — mirrors PokerEnv exactly so the
    # subgame solver keys SolverState nodes identically off a FastState.
    # ------------------------------------------------------------------
    @property
    def public_key(self):
        """``(stage_name, ((stage_name, (action_str, ...)), ...))`` — byte-for-byte
        identical to ``PokerEnv.public_key`` for canonical (on-tree) histories.

        Rebuilt from the resolved ``hist`` byte-codes: iterating ``si=0..3`` where
        ``hist_n[si] > 0`` reproduces ``_history.items()`` play order (stages are
        only ever inserted ascending), and inverting the dumped action alphabet
        (skip=0) recovers the exact token strings PokerEnv appended (including the
        ``"skip"`` padding).  Off-tree injected raises have no alphabet code and are
        NOT representable here — the solver falls back to the Python walk when the
        overlay is non-empty (see FastEnvAdapter), so this only ever runs on
        canonical histories.
        """
        cdef int si, k
        cdef int st = self.betting_stage
        if st < N_DECISION_STAGES:
            stage_name = _STAGE_NAMES[st]
        elif st == ST_SHOWDOWN:
            stage_name = "show_down"
        else:
            stage_name = "terminal"
        items = []
        cdef dict inv
        for si in range(N_DECISION_STAGES):
            if self.hist_n[si] > 0:
                inv = <dict>_CODE_ACTION[si]
                actions = []
                for k in range(self.hist_n[si]):
                    actions.append(inv[self.hist[si][k]])
                items.append((_STAGE_NAMES[si], tuple(actions)))
        return (stage_name, tuple(items))

    @property
    def n_raises_this_round(self):
        return self.n_raises

    @property
    def terminal_board_len(self):
        """Community-card count when betting ended, or ``None`` (PokerEnv parity)."""
        return None if self._terminal_board_len < 0 else self._terminal_board_len

    @property
    def pot_size(self):
        return self._pot_size()

    cdef inline long _pot_size(self):
        cdef long tot = 0
        cdef int s
        for s in range(self.n_players):
            tot += self.pot_chips[s]
        return tot

    cdef inline long _biggest_bet(self):
        cdef long mx = self.n_bet_chips[0]
        cdef int s
        for s in range(1, self.n_players):
            if self.n_bet_chips[s] > mx:
                mx = self.n_bet_chips[s]
        return mx

    cdef inline bint _is_all_in(self, int seat):
        return self.is_active[seat] != 0 and self.n_chips[seat] == 0

    cdef int _n_players_with_moves(self):
        cdef int c = 0
        cdef int s
        for s in range(self.n_players):
            if self.is_active[s] != 0 and self.n_chips[s] != 0:
                c += 1
        return c

    def n_players_with_moves(self):
        return self._n_players_with_moves()

    cdef int _n_active_players(self):
        cdef int c = 0
        cdef int s
        for s in range(self.n_players):
            if self.is_active[s] != 0:
                c += 1
        return c

    cdef bint _more_betting_needed(self):
        # True iff some live (active, non-all-in) player has not matched the
        # largest bet among ALL active players (all-in included) — mirrors
        # dynamics.more_betting_needed / FastStateRef.more_betting_needed.
        cdef long max_bet = 0
        cdef bint any_live = False
        cdef int s
        for s in range(self.n_players):
            if self.is_active[s] != 0:
                if self.n_bet_chips[s] > max_bet:
                    max_bet = self.n_bet_chips[s]
                if self.n_chips[s] != 0:
                    any_live = True
        if not any_live:
            return False
        for s in range(self.n_players):
            if self.is_active[s] != 0 and self.n_chips[s] != 0:
                if self.n_bet_chips[s] < max_bet:
                    return True
        return False

    cdef inline bint _all_players_have_actioned(self):
        return self.n_actions >= self.n_players_started_round

    cdef bint _hand_over(self):
        # Terminal when <=1 active, or all remaining active are all-in, or the
        # sole live player has matched the largest bet.  A lone live player who
        # still owes a call is NOT terminal.  Mirrors FastStateRef._hand_over.
        cdef int n_active = 0
        cdef int n_live = 0
        cdef int live_seat = -1
        cdef long max_bet = 0
        cdef int s
        for s in range(self.n_players):
            if self.is_active[s] != 0:
                n_active += 1
                if self.n_bet_chips[s] > max_bet:
                    max_bet = self.n_bet_chips[s]
                if self.n_chips[s] != 0:
                    n_live += 1
                    live_seat = s
        if n_active <= 1:
            return True
        if n_live == 0:
            return True
        if n_live == 1:
            return self.n_bet_chips[live_seat] >= max_bet
        return False

    # ------------------------------------------------------------------
    # legal_actions (canonical, no overlay) — returns Python action strings
    # ------------------------------------------------------------------
    def legal_actions(self):
        cdef int seat = self._cur_seat()
        if self.is_active[seat] == 0:
            return [None]
        cdef long biggest_bet = self._biggest_bet()
        cdef long n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        cdef long chips_available = self.n_chips[seat]
        actions = ["fold"]
        if n_chips_to_call >= chips_available:
            if chips_available > 0:
                actions.append("all_in")
        else:
            actions.append("call")
            if self.n_raises < _MAX_RAISES and self._n_players_with_moves() >= 2:
                actions += self._get_available_raise_sizes()
        return actions

    cdef long _compute_raise_chip_amount(self, double pot_fraction, bint enforce_minimum):
        cdef int seat = self._cur_seat()
        cdef long biggest_bet = self._biggest_bet()
        cdef long n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        cdef long n_chips_to_add = <long>ceil(<double>self._pot_size() * pot_fraction)
        cdef long floor_amt
        if enforce_minimum:
            floor_amt = n_chips_to_call + self.last_raise_amount
            if floor_amt > n_chips_to_add:
                n_chips_to_add = floor_amt
        return n_chips_to_add

    def _get_available_raise_sizes(self):
        cdef int st = self.betting_stage
        if st >= ST_SHOWDOWN:
            return []
        cdef int seat = self._cur_seat()
        cdef long biggest_bet = self._biggest_bet()
        cdef long n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        cdef long chips_available = self.n_chips[seat]
        cdef int n
        cdef int k
        cdef double frac
        cdef long chips_raw, actual_raise, chips
        cdef long added[MAX_FRACS]
        cdef int n_added = 0
        cdef int j
        cdef bint dup
        raise_actions = []
        if self.n_raises == 0:
            n = _FIRST_N[st]
            frac_strs = <list>_FIRST_STR[st]
        else:
            n = _SUB_N[st]
            frac_strs = <list>_SUB_STR[st]
        for k in range(n):
            if self.n_raises == 0:
                frac = _FIRST_FRAC[st][k]
            else:
                frac = _SUB_FRAC[st][k]
            chips_raw = self._compute_raise_chip_amount(frac, False)
            actual_raise = chips_raw - n_chips_to_call
            if actual_raise < self.last_raise_amount:
                continue
            chips = self._compute_raise_chip_amount(frac, True)
            if chips > chips_available or chips >= chips_available - 1:
                continue
            dup = False
            for j in range(n_added):
                if added[j] == chips:
                    dup = True
                    break
            if dup:
                continue
            added[n_added] = chips
            n_added += 1
            raise_actions.append(frac_strs[k])
        if chips_available > 0 and chips_available >= n_chips_to_call:
            dup = False
            for j in range(n_added):
                if added[j] == chips_available:
                    dup = True
                    break
            if not dup:
                raise_actions.append("all_in")
        return raise_actions

    # ------------------------------------------------------------------
    # info_set — pure-C encode over resolved history byte-codes
    # ------------------------------------------------------------------
    def info_set(self):
        cdef int seat = self._cur_seat()
        cdef int rnd = self.betting_stage
        # Match PokerEnv._compute_info_set's LOUD contract: a decision-stage
        # cluster missing from card_info_lut raises ValueError there (only
        # terminal/show_down falls back to a default).  info_set() is only called
        # at decision nodes (rnd 0..3), so a missing cluster here means a genuinely
        # incomplete LUT — without this guard the precompute's swallow (see
        # from_poker_env) would leave clusters[seat][rnd]==0 and the core would
        # SILENTLY train on a wrong (cluster-0) info-set while every PokerEnv path
        # crashes.  No-op on a complete LUT (has_cluster==1 at every used node).
        if rnd < N_DECISION_STAGES and self.has_cluster[seat][rnd] == 0:
            raise ValueError(
                "FastState.info_set: no cluster for seat %d at decision stage %d "
                "— cards missing from card_info_lut (load it correctly). Mirrors "
                "PokerEnv._compute_info_set; the compiled core must never silently "
                "fall back to cluster 0." % (seat, rnd)
            )
        return self._encode_info_set(self.clusters[seat][rnd])

    def info_set_for(self, int cluster):
        """``info_set`` for a **hypothetical** cluster at the current node.

        The info-set key is ``varint(cluster) ++ history``, and the history is
        public — so the only card-dependent term is the cluster.  Supplying it
        explicitly yields the exact key any holding in that LUT cluster would
        produce, without reseating the state.

        This is what lets the opponent-model clamp
        (:func:`poker_ai.search.vform.apply_model_clamp`) run on the compiled walk:
        it must query the model per hypothetical combo, which on a ``PokerEnv``
        goes through :meth:`~environment.poker_env.PokerEnv.policy_state_for`.
        Since combos sharing a cluster share an info-set, one call per cluster
        covers them all.

        No ``has_cluster`` guard: the caller supplies the cluster (from the LUT via
        ``ClusterMapper``), so the silent-cluster-0 failure :meth:`info_set` guards
        against cannot arise here.

        Byte-identical to ``PokerEnv.policy_state_for(combo).info_set`` for any
        combo in ``cluster``, on the canonical histories this engine represents
        (the adapters are only built when the overlay is empty, so history
        canonicalisation is a no-op).
        """
        return self._encode_info_set(cluster)

    cdef bytes _encode_info_set(self, int cluster):
        # Size bound (never under-allocate → no silent heap overflow): a uLEB128
        # varint is at most 10 bytes for a full uint64, so allow 10 for the
        # cluster and, per stage, 1 stage byte + 10 for varint(count) + the codes.
        cdef Py_ssize_t need = 10
        cdef int si
        for si in range(N_DECISION_STAGES):
            if self.hist_n[si] > 0:
                need += 11 + self.hist_n[si]
        cdef unsigned char* buf = <unsigned char*>malloc(need)
        if buf == NULL:
            raise MemoryError()
        cdef Py_ssize_t pos = 0
        cdef int k
        try:
            _put_uvarint(buf, &pos, <unsigned long long>cluster)
            for si in range(N_DECISION_STAGES):
                if self.hist_n[si] > 0:
                    buf[pos] = <unsigned char>_STAGE_BYTE[si]
                    pos += 1
                    _put_uvarint(buf, &pos, <unsigned long long>self.hist_n[si])
                    for k in range(self.hist_n[si]):
                        buf[pos] = self.hist[si][k]
                        pos += 1
            return PyBytes_FromStringAndSize(<char*>buf, pos)
        finally:
            free(buf)

    # ------------------------------------------------------------------
    # step / undo (C POD stack)
    # ------------------------------------------------------------------
    cdef void _push_frame(self) except *:
        # ``except *`` is REQUIRED: a bare ``cdef void`` swallows any raise at the
        # C call boundary (Cython prints "Exception ignored" and continues with
        # corrupted state).  This method can raise MemoryError on realloc failure.
        cdef FrameT* f
        cdef int s
        if self._stack_top >= self._stack_cap:
            self._stack_cap *= 2
            self._stack = <FrameT*>realloc(self._stack, self._stack_cap * sizeof(FrameT))
            if self._stack == NULL:
                raise MemoryError()
        f = &self._stack[self._stack_top]
        f.betting_stage = self.betting_stage
        f.player_i_index = self.player_i_index
        f.n_raises = self.n_raises
        f.n_actions = self.n_actions
        f.skip_counter = self.skip_counter
        f.last_raise_amount = self.last_raise_amount
        f.n_players_started_round = self.n_players_started_round
        f.terminal_board_len = self._terminal_board_len
        for s in range(self.n_players):
            f.n_chips[s] = self.n_chips[s]
            f.n_bet_chips[s] = self.n_bet_chips[s]
            f.is_active[s] = self.is_active[s]
            f.pot_chips[s] = self.pot_chips[s]
        for s in range(N_DECISION_STAGES):
            f.hist_n[s] = self.hist_n[s]
        self._stack_top += 1

    cdef void _restore_frame(self, int idx):
        cdef FrameT* f = &self._stack[idx]
        cdef int s
        self.betting_stage = f.betting_stage
        self.player_i_index = f.player_i_index
        self.n_raises = f.n_raises
        self.n_actions = f.n_actions
        self.skip_counter = f.skip_counter
        self.last_raise_amount = f.last_raise_amount
        self.n_players_started_round = f.n_players_started_round
        self._terminal_board_len = f.terminal_board_len
        for s in range(self.n_players):
            self.n_chips[s] = f.n_chips[s]
            self.n_bet_chips[s] = f.n_bet_chips[s]
            self.is_active[s] = f.is_active[s]
            self.pot_chips[s] = f.pot_chips[s]
        for s in range(N_DECISION_STAGES):
            self.hist_n[s] = f.hist_n[s]

    def step_in_place(self, action_str):
        """Apply ``action_str`` in place; return an int undo token (stack index)."""
        cdef int token = self._stack_top
        self._push_frame()
        self._apply(action_str)
        return token

    def undo(self, token):
        """Restore the state pushed by the matching :meth:`step_in_place` (LIFO)."""
        cdef int idx = <int>token
        if idx != self._stack_top - 1:
            raise RuntimeError(
                "FastState.undo out of LIFO order (expected top=%d, got %d)"
                % (self._stack_top - 1, idx)
            )
        self._restore_frame(idx)
        self._stack_top -= 1

    # ------------------------------------------------------------------
    # Internal transitions (mirror FastStateRef._apply / PokerEnv)
    # ------------------------------------------------------------------
    cdef inline int _resolve_code(self, dict table, token) except -1:
        code = table.get(token)
        if code is None:
            raise RuntimeError(
                "FastState: token %r has no alphabet code in this stage — off-tree "
                "raw-token path is not represented in the compiled history."
                % (token,)
            )
        return <int>code

    cdef void _push_hist_code(self, int stage, int code) except *:
        # ``except *`` REQUIRED (see _push_frame): the overflow guard below raises.
        if self.hist_n[stage] >= MAX_STAGE_ACTIONS:
            raise RuntimeError("FastState history overflow — raise MAX_STAGE_ACTIONS")
        self.hist[stage][self.hist_n[stage]] = <unsigned char>code
        self.hist_n[stage] += 1

    cdef void _add_to_pot(self, int seat, long n_chips):
        cdef long actual = n_chips
        if self.n_chips[seat] < actual:
            actual = self.n_chips[seat]
        self.pot_chips[seat] += actual
        self.n_chips[seat] -= actual
        self.n_bet_chips[seat] += actual

    cdef void _move_to_next_player(self):
        self.player_i_index += 1
        if self.player_i_index >= self.n_players:
            self.player_i_index = 0

    cdef void _reset_betting_round_state(self):
        self.n_actions = 0
        self.n_raises = 0
        self.last_raise_amount = self.big_blind
        self.player_i_index = 0
        # Count only players who can still act (active AND not all-in).
        self.n_players_started_round = self._n_players_with_moves()
        while self.is_active[self._cur_seat()] == 0:
            self.skip_counter += 1
            self.player_i_index += 1

    cdef void _increment_stage(self):
        cdef int st = self.betting_stage
        cdef int s
        if st == ST_PRE:
            self.betting_stage = ST_FLOP
        elif st == ST_FLOP:
            self.betting_stage = ST_TURN
        elif st == ST_TURN:
            self.betting_stage = ST_RIVER
        elif st == ST_RIVER:
            self.betting_stage = ST_SHOWDOWN
        # show_down / terminal: no change
        for s in range(self.n_players):
            self.n_bet_chips[s] = 0

    cdef void _apply(self, action_str) except *:
        # ``except *`` REQUIRED (see _push_frame): this raises ValueError on an
        # unrecognised action and propagates RuntimeError from _resolve_code /
        # _push_hist_code.  Without it those raises are silently swallowed and the
        # state is left half-mutated — the invalid action appears to "succeed".
        cdef int seat = self._cur_seat()
        cdef int entry_stage = self.betting_stage
        cdef long biggest_bet, n_chips_to_call, n_chips_to_add, actual_raise_amount
        cdef double pot_fraction
        cdef int cur, k

        if action_str is None:
            if self.is_active[seat] != 0:
                raise AssertionError("Active player cannot do nothing!")
        elif action_str == "call":
            if not self._is_all_in(seat):
                biggest_bet = self._biggest_bet()
                self._add_to_pot(seat, biggest_bet - self.n_bet_chips[seat])
        elif action_str == "fold":
            self.is_active[seat] = 0
        elif action_str == "all_in":
            n_chips_to_add = self.n_chips[seat]
            biggest_bet = self._biggest_bet()
            n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self.last_raise_amount:
                self.last_raise_amount = actual_raise_amount
                self.n_raises += 1
            self._add_to_pot(seat, n_chips_to_add)
        elif isinstance(action_str, str) and (<str>action_str).startswith("raise:"):
            pot_fraction = float((<str>action_str).split(":")[1])
            n_chips_to_add = self._compute_raise_chip_amount(pot_fraction, True)
            biggest_bet = self._biggest_bet()
            n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self.last_raise_amount:
                self.last_raise_amount = actual_raise_amount
            self._add_to_pot(seat, n_chips_to_add)
            self.n_raises += 1
        else:
            raise ValueError("Unrecognised action '%r'." % (action_str,))

        # Append skip padding + the action token, as resolved byte-codes.
        cdef dict table = <dict>_ACTION_CODE[entry_stage]
        for k in range(self.skip_counter):
            self._push_hist_code(entry_stage, 0)  # "skip" == 0 in every stage
        self._push_hist_code(entry_stage, self._resolve_code(table, action_str))
        self.n_actions += 1
        self.skip_counter = 0

        # Advance loop (mirrors the corrected PokerEnv._apply_action_in_place):
        # detect terminal / advance the stage only — never deal or settle.
        # Board length on the street the just-applied action was made on — the
        # value PokerEnv captures as ``board_len_at_action`` BEFORE any
        # round-closing stage advance, then stores as ``_terminal_board_len`` in
        # ``_settle_terminal`` (only-if-None).  The search vector_payout fold path
        # masks against it; ``_BOARD_LEN[entry_stage] == len(community_cards)``.
        while True:
            self._move_to_next_player()
            if self._hand_over():
                if self.betting_stage < ST_SHOWDOWN:
                    self.betting_stage = ST_TERMINAL
                if self._terminal_board_len < 0:
                    self._terminal_board_len = _BOARD_LEN[entry_stage]
                break
            if not self._more_betting_needed() and self._all_players_have_actioned():
                self._increment_stage()
                self._reset_betting_round_state()
                if self.betting_stage == ST_SHOWDOWN:
                    if self._terminal_board_len < 0:
                        self._terminal_board_len = _BOARD_LEN[entry_stage]
                    break
            cur = self._cur_seat()
            if self.is_active[cur] == 0:
                self.skip_counter += 1
                continue
            if self._is_all_in(cur):
                self.skip_counter += 1
                continue
            break

    # ------------------------------------------------------------------
    # Value layer (separate; Python evaluator + Pot — identical to FastStateRef)
    # ------------------------------------------------------------------
    def payout(self):
        """Net chip delta per seat at a terminal: ``won[i] - contrib[i]``."""
        cdef int seat
        board = [self.board[k] for k in range(5)]
        grouped = {}
        for seat in range(self.n_players):
            if self.is_active[seat] != 0:
                rank = default_evaluator.evaluate(
                    [self.hole[seat][0], self.hole[seat][1]], board
                )
                grouped.setdefault(rank, []).append(seat)
        stub = [Player(s) for s in range(self.n_players)]
        for seat in range(self.n_players):
            stub[seat].order = self.order[seat]
        ranked = [[stub[s] for s in grouped[r]] for r in sorted(grouped)]
        pot = Pot(self.n_players)
        pot._chips = [self.pot_chips[s] for s in range(self.n_players)]
        won = pot.compute_utility(stub, ranked)
        return {s: won[s] - self.pot_chips[s] for s in range(self.n_players)}

    def vector_payout(self, int seat, int opp_seat, opp_reach, runout, combo_cards):
        """Range-vs-range terminal value to ``seat`` — Phase-3b, byte-identical to
        ``PokerEnv.vector_payout``.

        The search vector regime's terminal settlement moved in-core: the caller
        (the compiled/adapter walk) supplies the CFR quantities it owns
        (``opp_seat``'s reach, the sampled ``runout``, the shared ``combo_cards``);
        everything else reads FastState fields.  ``pot_chips`` are the per-seat
        contributions PokerEnv captures as ``_terminal_contributions`` (same source
        ``payout`` uses).  It calls the **module-level** ``range_showdown``
        functions (``showdown_cfv``/``reach_after_removal`` are core-backed under the
        ``showdown`` flag), exactly as ``PokerEnv.vector_payout`` does — so the whole
        method is byte-identical to the env by construction; only the field source
        differs.  Precondition (as in the env): a terminal with exactly two
        contesting seats.

        ``runout`` is the sampled board completion — one card for a turn root,
        two for a flop root, ``None`` for an already-complete board.  A bare int
        is accepted as the one-card form (see ``poker_env._as_runout``).
        """
        if not self.is_terminal:
            raise ValueError("vector_payout is only defined at a terminal node.")
        cdef int s
        cdef long tc_seat = self.pot_chips[seat]
        cdef long tc_opp = self.pot_chips[opp_seat]
        cdef long tc_sum = 0
        for s in range(self.n_players):
            tc_sum += self.pot_chips[s]
        # Matched stake (winner-takes; larger stack's excess uncalled) + dead money
        # (every other, folded, seat's contribution) — float, mirroring the env.
        stake = float(tc_seat if tc_seat < tc_opp else tc_opp)
        dead = float(tc_sum - tc_seat - tc_opp)
        low = self._low_card_rank
        high = self._high_card_rank
        removal = range_showdown.removal_for(low, high)
        community = [self.board[s] for s in range(5)]
        comp = _as_runout(runout)
        # Board cards already public at the subgame root; the runout is the rest.
        prefix_len = 5 - len(comp) if comp is not None else len(community)

        if self.is_active[seat] != 0 and self.is_active[opp_seat] != 0:
            # Showdown: complete the board with the search's sampled runout (or the
            # already-complete board for a river subgame), then settle ranges.
            board = community[:prefix_len] + list(comp) if comp is not None else community
            ranks, valid = range_showdown.ranked_board(low, high, board)
            return range_showdown.showdown_cfv(
                ranks, valid, combo_cards, opp_reach, stake, dead=dead, removal=removal
            )

        # Fold: the still-active contesting seat wins; value is rank-independent.
        # A fold sees only the runout cards dealt by the time it happened — from a
        # flop root that is three cases (flop-/turn-/river-side), not two.
        winner = seat if self.is_active[seat] != 0 else opp_seat
        sign = 1.0 if winner == seat else -1.0
        real_len = self._terminal_board_len
        if real_len < 0:
            real_len = len(community)
        if comp is None or real_len <= prefix_len:
            board = community[:real_len]
        else:
            board = community[:prefix_len] + list(comp[: real_len - prefix_len])
        valid = range_showdown.board_valid_mask(low, high, board)
        gain = stake + dead if winner == seat else stake
        return range_showdown.fold_cfv(
            valid, combo_cards, opp_reach, sign * gain, removal
        )

    def vector_payout_concrete(self, int traverser_seat, combo_cards, feasible=None):
        """Per-combo chip delta to ``traverser_seat`` at a terminal with **concrete**
        opponents — byte-identical to ``PokerEnv.vector_payout_concrete`` (search
        Phase 2).

        The traverser-vectorized MCCFR terminal / leaf settlement moved in-core: the
        caller supplies the shared ``combo_cards``; everything else reads FastState
        fields (per-seat holes, the complete board, contributions, active mask, bet
        order — the engine owns them).  Every *other* seat holds a single concrete
        sampled hand, so only the traverser's own hand varies per combo.  Returns a
        ``(n_combos,)`` float64 chip delta (``0`` on combos the traverser cannot hold
        given the board + other seats' cards).  Settles via the module-level
        ``_settle_traverser`` (core-backed under the ``settle_concrete`` flag), so the
        method is byte-identical to the env by construction — only the field source
        differs.  The board is complete (5 cards) at every terminal a leaf rollout or
        MCCFR walk reaches.

        ``feasible``, optional: precomputed ``(n_combos,)`` bool feasibility mask —
        a caller settling many terminals against the SAME frozen board + opponent
        holes in one MCCFR iteration (the main walk) computes this once and passes
        it through, skipping the exclusion-set rebuild below.  ``None`` (default)
        computes it fresh, as before — required whenever the board/opponents can
        differ per call (e.g. a leaf rollout's own independently-drawn board).
        """
        if not self.is_terminal:
            raise ValueError(
                "vector_payout_concrete is only defined at a terminal node."
            )
        n = self.n_players
        seat = traverser_seat
        contribution = float(self.pot_chips[seat])
        n_combos = combo_cards.shape[0]
        community = [self.board[k] for k in range(5)]

        low, high = self._low_card_rank, self._high_card_rank
        removal = range_showdown.removal_for(low, high)
        deck_uniq = range_showdown.deck_slots(low, high)
        if feasible is None:
            # Feasibility: the traverser's combo shares no card with the board or
            # any OTHER seat's concrete hole.  Impossible combos return 0.  Card
            # ints are Cactus-Kev encoded (not ``0..51``), so membership is tested
            # via the densified slot index (``removal_index``/``deck_slots``)
            # rather than ``np.isin`` against the tiny exclusion set — a dense
            # boolean lookup over ``deck_size`` (<=52) slots beats a general
            # set-membership scan, and this runs at every terminal node in the
            # compiled walk.
            excluded = set(community)
            for s in range(n):
                if s != seat:
                    excluded.add(self.hole[s][0])
                    excluded.add(self.hole[s][1])
            s0, s1, deck_size = removal
            excl_slots = np.searchsorted(deck_uniq, list(excluded))
            excl_mask = np.zeros(deck_size, dtype=bool)
            excl_mask[excl_slots] = True
            feasible = ~(excl_mask[s0] | excl_mask[s1])

        # (i) traverser folded → forfeits its contribution regardless of hand.
        if self.is_active[seat] == 0:
            return np.where(feasible, -contribution, 0.0)

        # Rank the traverser's combos once; each opponent ranks to a scalar.  Build
        # ``rank_mat`` (n_combos, n_active) column-aligned to the active seats.
        active_seats = [s for s in range(n) if self.is_active[s] != 0]
        # Reuse the deck densification already computed above (this fires on a
        # FRESH, uncached board most terminals — a distinct MCCFR-sampled runout
        # each time — so ranked_board's board-keyed LRU cache doesn't help here;
        # passing removal/deck_uniq through skips a second np.isin pass).
        trav_ranks, _ = range_showdown.rank_combos_on_board(
            combo_cards, community, removal=removal, deck_uniq=deck_uniq,
        )
        trav_ranks = trav_ranks.copy()
        trav_ranks[~feasible] = range_showdown._SENTINEL_RANK
        opp_hands = [
            [self.hole[s][0], self.hole[s][1]] + community
            for s in active_seats if s != seat
        ]
        opp_ranks = (
            default_evaluator.evaluate_batch(np.asarray(opp_hands, dtype=np.int64))
            if opp_hands else np.empty(0, dtype=np.int64)
        )
        rank_mat = np.empty((n_combos, len(active_seats)), dtype=np.int64)
        oi = 0
        for a, s in enumerate(active_seats):
            if s == seat:
                rank_mat[:, a] = trav_ranks
            else:
                rank_mat[:, a] = opp_ranks[oi]
                oi += 1

        # Player stubs carry the plain ``order`` / ``player_i`` the settlement wants
        # (as ``payout`` builds them); ``_settle_traverser`` marshals ints from these.
        stub = [Player(s) for s in range(n)]
        for s in range(n):
            stub[s].order = self.order[s]
            stub[s].is_active = self.is_active[s] != 0
        active_players = [stub[s] for s in active_seats]
        pot_chips = [self.pot_chips[s] for s in range(n)]
        won = _settle_traverser(
            active_players, stub, pot_chips, rank_mat, n_combos, n, seat
        )
        cfv = won - contribution
        cfv[~feasible] = 0.0
        return cfv

    # ------------------------------------------------------------------
    # Decision-free all-in runout (Phase 4a) — the shared terminal the MCCFR
    # walk AND the leaf rollout score exactly, ported off PokerEnv so both can
    # settle from a FastState.  Byte-identical to PokerEnv (the completion
    # average is an integer chip sum → order-independent, so the completion
    # enumeration order need not match the deck's).
    # ------------------------------------------------------------------
    @property
    def is_decision_free(self):
        """True iff this is a terminal all-in showdown over an incomplete board.

        Mirrors ``PokerEnv.is_decision_free`` (``_runout_info is not None``, set in
        ``_settle_terminal`` when ``5-len(community) > 0`` and ``n_active >= 2``):
        terminal, ``0 <= terminal_board_len < 5``, and >=2 seats still active.
        """
        if not (self.betting_stage == ST_SHOWDOWN or self.betting_stage == ST_TERMINAL):
            return False
        return (0 <= self._terminal_board_len < 5) and self._n_active_players() >= 2

    @property
    def runout_key(self):
        """``(board_prefix, pot_contributions, active_mask)`` — ``None`` if not a
        decision-free runout.  Byte-identical to ``PokerEnv.runout_key``
        (``_runout_info``), so a caller memoising ``runout_equity`` integrations
        (e.g. AIVAT's terminal chance correction) keys identically regardless of
        which engine reached the terminal."""
        if not self.is_decision_free:
            return None
        cdef int tbl = self._terminal_board_len
        cdef int s
        prefix = tuple(self.board[s] for s in range(tbl))
        pot = tuple(self.pot_chips[s] for s in range(self.n_players))
        active = tuple(bool(self.is_active[s]) for s in range(self.n_players))
        return (prefix, pot, active)

    def runout_equity(self, rng=None, cap=5000):
        """Exact expected per-seat chip delta over every board completion.

        Byte-identical drop-in for ``PokerEnv.runout_equity``: reconstructs the
        pre-runout snapshot from FastState fields (prefix = ``board[:terminal_board_len]``,
        contributions = ``pot_chips``, active = ``is_active``), enumerates the
        completion deck (all deck cards minus every dealt hole and the prefix),
        batch-ranks each (completion x active seat) 7-card hand with the shared
        evaluator, and settles via the SAME module-level ``_settle_runout`` PokerEnv
        uses (core-backed under the ``runout`` flag) over Player stubs carrying
        ``player_i``/``order``.  The completion average is an integer chip sum, so
        the result is independent of completion order (the deck order need not match).
        """
        if not self.is_decision_free:
            raise ValueError(
                "runout_equity requires a decision-free all-in runout state "
                "(is_decision_free is False); nothing to integrate."
            )
        cdef int n = self.n_players
        cdef int tbl = self._terminal_board_len
        cdef int s, kk = 5 - tbl
        prefix = [self.board[s] for s in range(tbl)]
        used = set(prefix)
        for s in range(n):
            used.add(self.hole[s][0])
            used.add(self.hole[s][1])
        combo_cards, _ = enumerate_combos(self._low_card_rank, self._high_card_rank)
        full_deck = [int(c) for c in np.unique(combo_cards)]
        available = [c for c in full_deck if c not in used]

        if kk <= 0:
            completions = [()]
        else:
            n_combos = _n_choose_k(len(available), kk)
            if n_combos <= cap:
                completions = itertools.combinations(available, kk)
            else:
                gen = rng if rng is not None else np.random.default_rng()
                completions = (
                    tuple(gen.choice(available, size=kk, replace=False))
                    for _ in range(cap)
                )
        prefix_list = list(prefix)
        completions = [prefix_list + [int(c) for c in comp] for comp in completions]
        count = len(completions)

        # Player stubs carrying player_i (== seat) + order for the side-pot settle.
        stub = [Player(s) for s in range(n)]
        for s in range(n):
            stub[s].order = self.order[s]
        active_players = [stub[s] for s in range(n) if self.is_active[s] != 0]
        pot_chips = [self.pot_chips[s] for s in range(n)]

        accum = [0.0] * n
        if count and active_players:
            boards = np.asarray(completions, dtype=np.int64)          # (count, 5)
            holes = np.asarray(
                [[self.hole[s][0], self.hole[s][1]]
                 for s in range(n) if self.is_active[s] != 0],
                dtype=np.int64,
            )                                                          # (n_active, 2)
            n_active = len(active_players)
            hands = np.empty((count, n_active, 7), dtype=np.int64)
            hands[:, :, :5] = boards[:, None, :]
            hands[:, :, 5:] = holes[None, :, :]
            rank_mat = default_evaluator.evaluate_batch(
                hands.reshape(count * n_active, 7)
            ).reshape(count, n_active)
            accum = _settle_runout(active_players, stub, pot_chips, rank_mat, count, n)

        if count == 0:
            return {i: float(-pot_chips[i]) for i in range(n)}
        return {i: accum[i] / count - pot_chips[i] for i in range(n)}

    # ------------------------------------------------------------------
    # Debug / differential-test accessors (not on the hot path)
    # ------------------------------------------------------------------
    def cluster_at(self, int seat, int si):
        """``(cluster_id, has_cluster)`` for ``(seat, street)`` — Phase-4b gate for
        ``refresh_clusters`` (not on the hot path)."""
        return (self.clusters[seat][si], self.has_cluster[seat][si])

    def snapshot(self):
        """Full mutable-state tuple for differential comparison against the ref."""
        cdef int s, k
        return (
            self.betting_stage, self.player_i_index, self.n_raises,
            self.n_actions, self.skip_counter, self.last_raise_amount,
            self.n_players_started_round,
            tuple(self.n_chips[s] for s in range(self.n_players)),
            tuple(self.n_bet_chips[s] for s in range(self.n_players)),
            tuple(self.is_active[s] for s in range(self.n_players)),
            tuple(self.pot_chips[s] for s in range(self.n_players)),
            tuple(
                tuple(self.hist[si][k] for k in range(self.hist_n[si]))
                for si in range(N_DECISION_STAGES)
            ),
        )


cdef inline void _put_uvarint(unsigned char* buf, Py_ssize_t* pos, unsigned long long value):
    # Unsigned LEB128 — identical to poker_env._put_uvarint / _infoset.put_varint.
    cdef unsigned int byte
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buf[pos[0]] = <unsigned char>(byte | 0x80)
            pos[0] += 1
        else:
            buf[pos[0]] = <unsigned char>byte
            pos[0] += 1
            return
