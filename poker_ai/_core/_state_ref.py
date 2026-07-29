"""Pure-Python reduced betting-state reference for the Cython core (Phase 2).

``FastStateRef`` is the *spec and oracle* for the compiled ``_state.pyx``: a
faithful reduction of :class:`environment.poker_env.PokerEnv`'s blueprint-training
betting engine, with everything the scalar external-sampling traversal never
touches stripped out, and ``Player``/``Pot``/``Deck`` objects flattened to plain
arrays.  It is byte-for-byte equivalent to ``PokerEnv`` on the blueprint contract
— ``legal_actions`` / ``is_terminal`` / ``payout`` / ``info_set`` / current seat at
every node — which the differential fuzz proves; the Cython port then transliterates
*this* (simpler, readable) reference rather than the full ``PokerEnv``.

What is dropped vs ``PokerEnv`` (each verified off the blueprint path):

* the real-time-search / vector regime (``vector_payout``, ``runout_equity``,
  ``policy_state``, ``public_state`` cache, the ``settle_winners=False`` branch,
  ``_runout_info``, ``_terminal_board_len``, ``_terminal_contributions``);
* the off-tree action **overlay** (``_extra_legal_actions`` / ``inject_action``) —
  training uses only canonical actions;
* the **deck and community dealing** — the board is line-independent
  (``deck[2n:2n+5]``) and precomputed, so the state carries only the *stage*;
* ``is_turn`` and the ``name`` / blind / dealer display flags.

**Regime-agnostic factoring** (so the future search port reuses this engine): the
state owns only pure betting; the value/settlement is a *separate* layer
(:meth:`payout` composes the evaluator + side-pot settlement).  Search will later
plug a range-vector value layer onto the same ``step`` / ``undo`` / ``legal_actions``
/ ``is_terminal`` engine.

Field names and the ``_apply`` control flow mirror ``PokerEnv`` deliberately, so
the reduction is auditable line-against-line.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from environment.poker_env import (
    RAISE_SIZES_BY_STAGE,
    encode_info_set,
    max_raises_per_round,
)

# Stage names in play order, mirroring PokerEnv._betting_stage strings.
_STAGES = ("pre_flop", "flop", "turn", "river", "show_down", "terminal")
_STAGE_TO_ROUND = {"pre_flop": 0, "flop": 1, "turn": 2, "river": 3, "show_down": 4}
# Community cards visible at each stage (the board is external / precomputed).
_BOARD_LEN = {
    "pre_flop": 0, "flop": 3, "turn": 4, "river": 5, "show_down": 5, "terminal": 5,
}
_TERMINAL_STAGES = frozenset({"show_down", "terminal"})


# The mutable snapshot restored by :meth:`FastStateRef.undo` — the reduced
# counterpart of ``PokerEnv``'s ``UndoToken`` (search-only fields dropped).
UndoToken = Tuple


class FastStateRef:
    """Reduced, flat-array betting engine equivalent to ``PokerEnv`` on the
    blueprint contract.  Board and per-(seat, round) clusters are fixed
    precomputed inputs; the value layer (:meth:`payout`) is composed separately.
    """

    __slots__ = (
        # config (immutable)
        "n_players", "small_blind", "big_blind", "initial_chips",
        "player_i_lut", "hole", "board", "clusters", "order",
        # mutable betting state
        "n_chips", "n_bet_chips", "is_active", "pot_chips",
        "betting_stage", "player_i_index", "n_raises", "n_actions",
        "skip_counter", "last_raise_amount",
        "n_players_started_round", "history",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        n_players: int,
        small_blind: int,
        big_blind: int,
        initial_chips: int,
        hole: Sequence[Sequence[int]],
        board: Sequence[int],
        clusters: Dict[Tuple[int, int], int],
        n_chips: Sequence[int],
        n_bet_chips: Sequence[int],
        is_active: Sequence[bool],
        pot_chips: Sequence[int],
        betting_stage: str,
        player_i_index: int,
        n_raises: int,
        n_actions: int,
        skip_counter: int,
        last_raise_amount: int,
        n_players_started_round: int,
        history: Dict[str, List[str]],
        order: Sequence[int],
    ) -> None:
        self.n_players = n_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_chips = initial_chips
        self.hole = [tuple(h) for h in hole]
        self.board = tuple(board)
        # clusters[(seat, round)] -> cluster id, for rounds 0..3 (decision nodes).
        self.clusters = dict(clusters)
        self.order = list(order)
        # Per-stage seat permutation: pre_flop starts left of the BB, others at
        # seat 0 (SB) — mirrors PokerEnv._player_i_lut.
        base = list(range(n_players))
        # Heads-up: the button (seat 0) is the SB and acts LAST post-flop, so
        # the BB (seat 1) leads — post-flop order is the seat order reversed.
        # For 3+ players the SB (seat 0) leads post-flop, so plain seat order is
        # correct.  Mirrors PokerEnv._player_i_lut.
        postflop = base[::-1] if n_players == 2 else base
        self.player_i_lut = {
            "pre_flop": base[2:] + base[:2],
            "flop": postflop, "turn": postflop, "river": postflop,
            "show_down": postflop, "terminal": postflop,
        }
        # mutable
        self.n_chips = list(n_chips)
        self.n_bet_chips = list(n_bet_chips)
        self.is_active = [bool(a) for a in is_active]
        self.pot_chips = list(pot_chips)
        self.betting_stage = betting_stage
        self.player_i_index = player_i_index
        self.n_raises = n_raises
        self.n_actions = n_actions
        self.skip_counter = skip_counter
        self.last_raise_amount = last_raise_amount
        self.n_players_started_round = n_players_started_round
        self.history = {k: list(v) for k, v in history.items()}

    @classmethod
    def from_poker_env(cls, env) -> "FastStateRef":
        """Snapshot a live ``PokerEnv`` into an equivalent reduced state.

        Reads the env's current betting state verbatim and precomputes the
        board (``deck[2n:2n+5]`` — line-independent) and the per-(seat, round)
        clusters from the env's ``card_info_lut``, so both start field-identical.
        """
        n = env.n_players
        board = tuple(int(c) for c in env.deck.board_runout(5))
        hole = [tuple(p._cards) for p in env.players]
        # Precompute clusters for every (seat, round) decision node the same way
        # PokerEnv._compute_info_set does: LUT[stage][sorted(hole)+sorted(board_prefix)].
        clusters: Dict[Tuple[int, int], int] = {}
        for stage, rnd in (("pre_flop", 0), ("flop", 1), ("turn", 2), ("river", 3)):
            blen = _BOARD_LEN[stage]
            try:
                stage_lut = env.card_info_lut[stage]
            except (KeyError, TypeError):
                continue
            for seat in range(n):
                # Same key PokerEnv._compute_info_set uses; the LUT maps every
                # valid combo (combinadic index), so look it up directly.
                key = tuple(sorted(hole[seat]) + sorted(board[:blen]))
                try:
                    clusters[(seat, rnd)] = stage_lut[key]
                except (KeyError, IndexError):
                    pass
        return cls(
            n_players=n,
            small_blind=env.small_blind,
            big_blind=env.big_blind,
            initial_chips=env._initial_n_chips,
            hole=hole,
            board=board,
            clusters=clusters,
            n_chips=[p.n_chips for p in env.players],
            n_bet_chips=[p.n_bet_chips for p in env.players],
            is_active=[p.is_active for p in env.players],
            pot_chips=list(env.pot._chips),
            betting_stage=env._betting_stage,
            player_i_index=env._player_i_index,
            n_raises=env._n_raises,
            n_actions=env._n_actions,
            skip_counter=env._skip_counter,
            last_raise_amount=env._last_raise_amount,
            n_players_started_round=env._n_players_started_round,
            history={k: list(v) for k, v in env._history.items()},
            order=[p.order for p in env.players],
        )

    # ------------------------------------------------------------------
    # Derived read-only contract (mirrors the PokerEnv properties)
    # ------------------------------------------------------------------

    @property
    def player_i(self) -> int:
        """Seat index of the current actor (via the per-stage permutation)."""
        return self.player_i_lut[self.betting_stage][self.player_i_index]

    @property
    def is_terminal(self) -> bool:
        return self.betting_stage in _TERMINAL_STAGES

    @property
    def betting_round(self) -> int:
        return _STAGE_TO_ROUND[self.betting_stage]

    @property
    def pot_size(self) -> int:
        return sum(self.pot_chips)

    def _is_all_in(self, seat: int) -> bool:
        return self.is_active[seat] and self.n_chips[seat] == 0

    def n_active_players(self) -> int:
        return sum(1 for a in self.is_active if a)

    def n_players_with_moves(self) -> int:
        return sum(
            1 for s in range(self.n_players)
            if self.is_active[s] and not self._is_all_in(s)
        )

    def more_betting_needed(self) -> bool:
        # True iff some live (active, non-all-in) player has not matched the
        # largest bet among ALL active players — including all-in players, so an
        # over-the-top all-in still leaves live players owing a decision.
        # Mirrors PokerEnv/dynamics.more_betting_needed.
        active = [s for s in range(self.n_players) if self.is_active[s]]
        live = [s for s in active if not self._is_all_in(s)]
        if not live:
            return False
        max_bet = max(self.n_bet_chips[s] for s in active)
        return any(self.n_bet_chips[s] < max_bet for s in live)

    @property
    def all_players_have_actioned(self) -> bool:
        return self.n_actions >= self.n_players_started_round

    def legal_actions(self) -> List[Optional[str]]:
        """Canonical legal actions for the current actor (no overlay)."""
        seat = self.player_i
        if not self.is_active[seat]:
            return [None]
        biggest_bet = max(self.n_bet_chips)
        n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        chips_available = self.n_chips[seat]
        actions: List[Optional[str]] = ["fold"]
        if n_chips_to_call >= chips_available:
            if chips_available > 0:
                actions.append("all_in")
        else:
            actions.append("call")
            # Raises are only meaningful when another live player could call
            # them; facing a lone all-in the only responses are call/fold.
            if (self.n_raises < max_raises_per_round(self.n_players)
                    and self.n_players_with_moves() >= 2):
                actions += self._get_available_raise_sizes()
        return actions

    def info_set(self) -> bytes:
        """Compact info-set key for the current actor (encode_info_set)."""
        seat = self.player_i
        cluster = self.clusters[(seat, self.betting_round)]
        return encode_info_set(cluster, self.history.items())

    # ------------------------------------------------------------------
    # Raise-size enumeration (mirrors PokerEnv)
    # ------------------------------------------------------------------

    def _compute_raise_chip_amount(self, pot_fraction: float,
                                   enforce_minimum: bool = True) -> int:
        seat = self.player_i
        biggest_bet = max(self.n_bet_chips)
        n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        n_chips_to_add = math.ceil(self.pot_size * pot_fraction)
        if enforce_minimum:
            n_chips_to_add = max(
                n_chips_to_add, n_chips_to_call + self.last_raise_amount
            )
        return n_chips_to_add

    def _get_available_raise_sizes(self) -> List[str]:
        if self.betting_stage in _TERMINAL_STAGES:
            return []
        stage_config = RAISE_SIZES_BY_STAGE.get(self.betting_stage, {})
        fractions = (
            stage_config.get("first_raise", [1.0])
            if self.n_raises == 0
            else stage_config.get("subsequent_raise", [1.0])
        )
        seat = self.player_i
        biggest_bet = max(self.n_bet_chips)
        n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
        chips_available = self.n_chips[seat]
        added: set = set()
        raise_actions: List[str] = []
        for fraction in fractions:
            chips_raw = self._compute_raise_chip_amount(fraction, enforce_minimum=False)
            actual_raise = chips_raw - n_chips_to_call
            if actual_raise < self.last_raise_amount:
                continue
            chips = self._compute_raise_chip_amount(fraction, enforce_minimum=True)
            if chips > chips_available or chips >= chips_available - 1 or chips in added:
                continue
            added.add(chips)
            raise_actions.append(f"raise:{fraction}")
        if chips_available > 0 and chips_available >= n_chips_to_call:
            if chips_available not in added:
                raise_actions.append("all_in")
        return raise_actions

    # ------------------------------------------------------------------
    # Chip primitives (mirror Player.add_to_pot / call / raise_to)
    # ------------------------------------------------------------------

    def _add_to_pot(self, seat: int, n_chips: int) -> int:
        actual = min(n_chips, self.n_chips[seat])
        self.pot_chips[seat] += actual
        self.n_chips[seat] -= actual
        self.n_bet_chips[seat] += actual
        return actual

    # ------------------------------------------------------------------
    # step / undo
    # ------------------------------------------------------------------

    def _capture(self) -> UndoToken:
        return (
            self.betting_stage, self.player_i_index, self.n_raises,
            self.n_actions, self.skip_counter, self.last_raise_amount,
            self.n_players_started_round,
            list(self.n_chips), list(self.n_bet_chips), list(self.is_active),
            list(self.pot_chips),
            {k: list(v) for k, v in self.history.items()},
        )

    def step_in_place(self, action_str: Optional[str]) -> UndoToken:
        token = self._capture()
        self._apply(action_str)
        return token

    def undo(self, token: UndoToken) -> None:
        (self.betting_stage, self.player_i_index, self.n_raises,
         self.n_actions, self.skip_counter, self.last_raise_amount,
         self.n_players_started_round,
         n_chips, n_bet_chips, is_active, pot_chips, history) = token
        self.n_chips = list(n_chips)
        self.n_bet_chips = list(n_bet_chips)
        self.is_active = list(is_active)
        self.pot_chips = list(pot_chips)
        self.history = {k: list(v) for k, v in history.items()}

    # ------------------------------------------------------------------
    # Internal transitions (mirror PokerEnv._apply_action_in_place)
    # ------------------------------------------------------------------

    def _move_to_next_player(self) -> None:
        self.player_i_index += 1
        if self.player_i_index >= self.n_players:
            self.player_i_index = 0

    def _reset_betting_round_state(self) -> None:
        self.n_actions = 0
        self.n_raises = 0
        self.last_raise_amount = self.big_blind
        self.player_i_index = 0
        # Count only players who can still act (active AND not all-in); an
        # already-all-in player never acts this round, so counting them would
        # make ``all_players_have_actioned`` unreachable and re-poll the live
        # players.  Mirrors PokerEnv._reset_betting_round_state.
        self.n_players_started_round = self.n_players_with_moves()
        while not self.is_active[self.player_i]:
            self.skip_counter += 1
            self.player_i_index += 1

    def _increment_stage(self) -> None:
        stage = self.betting_stage
        if stage == "pre_flop":
            self.betting_stage = "flop"
        elif stage == "flop":
            self.betting_stage = "turn"
        elif stage == "turn":
            self.betting_stage = "river"
        elif stage == "river":
            self.betting_stage = "show_down"
        elif stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {stage}")
        for s in range(self.n_players):
            self.n_bet_chips[s] = 0

    def _apply(self, action_str: Optional[str]) -> None:
        seat = self.player_i

        if action_str is None:
            assert not self.is_active[seat], "Active player cannot do nothing!"
        elif action_str == "call":
            if not self._is_all_in(seat):
                biggest_bet = max(self.n_bet_chips)
                self._add_to_pot(seat, biggest_bet - self.n_bet_chips[seat])
        elif action_str == "fold":
            self.is_active[seat] = False
        elif action_str == "all_in":
            n_chips_to_add = self.n_chips[seat]
            biggest_bet = max(self.n_bet_chips)
            n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self.last_raise_amount:
                self.last_raise_amount = actual_raise_amount
                self.n_raises += 1
            self._add_to_pot(seat, n_chips_to_add)
        elif action_str.startswith("raise:"):
            pot_fraction = float(action_str.split(":")[1])
            n_chips_to_add = self._compute_raise_chip_amount(
                pot_fraction, enforce_minimum=True
            )
            biggest_bet = max(self.n_bet_chips)
            n_chips_to_call = biggest_bet - self.n_bet_chips[seat]
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self.last_raise_amount:
                self.last_raise_amount = actual_raise_amount
            self._add_to_pot(seat, n_chips_to_add)
            self.n_raises += 1
        else:
            raise ValueError(f"Unrecognised action '{action_str}'.")

        skip_actions = ["skip"] * self.skip_counter
        stage_hist = self.history.setdefault(self.betting_stage, [])
        stage_hist += skip_actions
        stage_hist.append(action_str)
        self.n_actions += 1
        self.skip_counter = 0

        # Advance loop, mirroring the corrected ``PokerEnv._apply_action_in_place``.
        # Settlement is deferred to :meth:`payout` (the separate value layer), so
        # this only detects the terminal / advances the stage — it never deals a
        # board or settles.  A player facing an all-in they can still act on is
        # NOT terminal (:meth:`_hand_over`); they get to call or fold first.
        while True:
            self._move_to_next_player()
            if self._hand_over():
                if self.betting_stage not in {"show_down", "terminal"}:
                    self.betting_stage = "terminal"
                break
            finished_betting = not self.more_betting_needed()
            if finished_betting and self.all_players_have_actioned:
                self._increment_stage()
                self._reset_betting_round_state()
                if self.betting_stage == "show_down":
                    break
            cur = self.player_i
            if not self.is_active[cur]:
                self.skip_counter += 1
                continue
            if self._is_all_in(cur):
                # An all-in player has no betting decision; advance past them.
                self.skip_counter += 1
                continue
            break

    def _hand_over(self) -> bool:
        """True when no active player has a further betting decision (mirrors
        ``PokerEnv._hand_over``): all but one folded, all remaining all-in, or the
        sole live player has matched the largest bet.  A lone live player who
        still owes an outstanding call is NOT terminal — they must act first."""
        active = [s for s in range(self.n_players) if self.is_active[s]]
        if len(active) <= 1:
            return True
        live = [s for s in active if not self._is_all_in(s)]
        if not live:
            return True
        if len(live) == 1:
            max_bet = max(self.n_bet_chips[s] for s in active)
            return self.n_bet_chips[live[0]] >= max_bet
        return False

    # ------------------------------------------------------------------
    # Value layer (separate from the betting engine; blueprint-concrete)
    # ------------------------------------------------------------------

    def payout(self) -> Dict[int, int]:
        """Net chip delta per seat at a terminal: ``won[i] - contrib[i]``.

        Ranks active seats over the full board with the shared evaluator, splits
        the side pots (mirroring ``Pot.compute_utility``), and nets against each
        seat's contribution — equal to ``PokerEnv.payout`` (``n_chips - initial``)
        by construction.  Kept out of the betting engine so search can later plug
        a range-vector value layer onto the same ``step``/``undo``.
        """
        from environment.evaluator import default_evaluator
        from environment.pot import Pot
        from environment.player import Player

        board = list(self.board)
        grouped: Dict[int, List[int]] = {}
        for seat in range(self.n_players):
            if self.is_active[seat]:
                rank = default_evaluator.evaluate(list(self.hole[seat]), board)
                grouped.setdefault(rank, []).append(seat)
        # Build the ranked groups (best rank first) and settle via the real Pot.
        stub = [Player(s) for s in range(self.n_players)]
        for s in range(self.n_players):
            stub[s].order = self.order[s]
        ranked = [[stub[s] for s in grouped[r]] for r in sorted(grouped)]
        pot = Pot(self.n_players)
        pot._chips = list(self.pot_chips)
        won = pot.compute_utility(stub, ranked)
        return {s: won[s] - self.pot_chips[s] for s in range(self.n_players)}
