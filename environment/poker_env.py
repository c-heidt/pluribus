"""Poker game environment for CFR/MCCFR training.

The ``PokerEnv`` class is the central state object. It holds all live
game data (players, pot, deck, community cards) and provides the CFR
interface: ``step_in_place`` / ``undo``, ``legal_actions``, ``info_set``,
and ``is_terminal``.

Deterministic game-logic functions are in ``dynamics.py``; stochastic
(dealing) operations are in ``chance.py``.
"""

from __future__ import annotations

import collections
import copy
import itertools
import json
import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from environment import dynamics
from environment import range_showdown
from environment.chance import Deck
from environment.evaluator import default_evaluator
from environment.player import Player
from environment.pot import Pot
from environment.utils import enumerate_combos


@dataclass(frozen=True)
class PolicyState:
    """Decoupled view of the env fields a :class:`Policy.strategy` needs.

    A :class:`PolicyState` is the value object the search package's
    :class:`poker_ai.search.policy.Policy` consumes instead of a raw
    :class:`PokerEnv`.  It carries exactly the public fields a policy
    or its instrumentation reads — current actor index, betting
    round, info-set key, canonical-width valid mask, and the
    legal-action list (non-``None`` entries, including overlay) —
    none of which depend on opponents' hole cards.

    Constructed by :attr:`PokerEnv.policy_state` (current actor at
    the current env state) or :meth:`PokerEnv.policy_state_for`
    (current actor with a hypothetical hole, for leak-free
    ``sigma_for_combo`` queries).

    Attributes
    ----------
    player_i : int
        Seat index of the current actor (public state; same value
        as :attr:`PokerEnv.player_i`).
    betting_round : int
        0=pre_flop, 1=flop, 2=turn, 3=river.
    info_set : bytes
        Compact binary info-set key (cluster + history); see
        :func:`encode_info_set`.  Opaque — hashed by the index/cache.
    valid_mask : numpy.ndarray
        Boolean mask over the canonical action set; immutable.
    legal_actions : tuple[str, ...]
        Legal actions for the current actor, in
        :attr:`PokerEnv.legal_actions` order, with ``None`` filtered.
    """

    player_i: int
    betting_round: int
    info_set: bytes
    valid_mask: np.ndarray
    legal_actions: Tuple[str, ...]


class PublicPolicyFields(NamedTuple):
    """The combo-independent part of a :class:`PolicyState`.

    Every field of a :class:`PolicyState` except ``info_set`` is a pure function of
    the public state — independent of the actor's hypothetical hole.  A caller that
    builds many :class:`PolicyState` objects for the *same* env state but different
    combos (e.g. the belief-update ``sigma_for_combo`` sweep over ~1326 combos) can
    compute these **once** via :meth:`PokerEnv.policy_public_fields` and pass them to
    :meth:`PokerEnv.policy_state_for` as ``public=`` so only the per-combo ``info_set``
    is rebuilt per combo.
    """

    player_i: int
    betting_round: int
    legal_actions: Tuple[str, ...]
    valid_mask: np.ndarray


@dataclass
class UndoToken:
    """Snapshot of the mutable per-hand state taken by
    :meth:`PokerEnv.step_in_place` and consumed by :meth:`PokerEnv.undo`.

    Covers exactly the field set :meth:`PokerEnv.__deepcopy__` deep-copies
    (the mutable per-hand state); restoring it returns the env to its
    pre-step state, field-identical to a pre-step ``deepcopy``.  All
    mutable containers are copied at capture time so the token is
    independent of the subsequent in-place mutation.
    """

    betting_stage: str
    skip_counter: int
    last_raise_amount: int
    n_actions: int
    n_raises: int
    player_i_index: int
    n_players_started_round: int
    community_cards: Tuple[int, ...]
    deck_cursor: int
    pot_chips: List[int]
    player_states: List[tuple]
    history: Dict[str, List[str]]
    runout_info: Optional[Tuple]
    terminal_contributions: Optional[Tuple]
    terminal_board_len: Optional[int]


logger = logging.getLogger("environment.poker_env")


def _n_choose_k(n: int, k: int) -> int:
    """Binomial coefficient C(n, k) (``math.comb`` is Python 3.8+; we target 3.7)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


class _NumpyJSONEncoder(json.JSONEncoder):
    """Handle those pesky numpy arrays on serialisation."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

# ---------------------------------------------------------------------------
# Action abstraction configuration
# ---------------------------------------------------------------------------
# Raise sizes as fractions of the current pot, by betting stage.
# Based on the Pluribus blueprint strategy design.
#
# Deliberately coarse: the number of distinct betting histories — and with
# it the infoset count and the per-traversal branching of CFR — grows
# multiplicatively in the per-node raise count, so every extra size is
# paid for across the whole tree.  A 6-max run with the previous 9/7
# preflop grid allocated ~1B infosets at only ~10M iterations per player
# (≈5M betting contexts per 200-bucket street), leaving the vast majority
# of regret rows with a handful of noisy updates.  Off-tree sizes at play
# time are handled by pseudo-harmonic action translation
# (:meth:`PokerEnv._translate_fraction`), and real-time search re-adds
# granularity where it matters, so blueprint coarseness here is the cheap
# axis to give up.
RAISE_SIZES_BY_STAGE: Dict[str, Dict[str, List[float]]] = {
    "pre_flop": {
        "first_raise":      [1.0, 2.0, 3.0],
        "subsequent_raise": [1.0],
    },
    "flop": {
        "first_raise":      [0.5, 1.0, 1.5],
        "subsequent_raise": [1.0],
    },
    "turn": {
        "first_raise":      [0.5, 1.0],
        "subsequent_raise": [1.0],
    },
    "river": {
        "first_raise":      [0.5, 1.0],
        "subsequent_raise": [1.0],
    },
}

MAX_RAISES_PER_ROUND: int = 3


# ---------------------------------------------------------------------------
# Compact info-set key encoding (v2)
# ---------------------------------------------------------------------------
# The info-set key is an opaque hash pre-image everywhere it is consumed
# (it is immediately fed to xxh3-128 by the LMDB index / shm cache; no
# consumer parses it in the CFR or search hot loop).  Historically it was
# ``json.dumps({"cards_cluster": C, "history": H})``, rebuilt from scratch
# at every visited node — an O(history) list rebuild + JSON encode per node,
# i.e. O(depth^2) per traversal and one of the two dominant inner-loop
# costs.  ``encode_info_set`` replaces that with an injective, compact
# ``bytes`` encoding built with single-byte appends.
#
# Injectivity: ``varint(cluster)`` then, per stage in play order,
# ``STAGE_ID`` byte + ``varint(len(actions))`` (frames the block) + one
# alphabet byte per action token.  The length prefix makes each stage block
# uniquely decodable, so distinct ``(cluster, history)`` map to distinct
# bytes.
#
# Canonicalisation equivalence (the load-bearing inference invariant): an
# off-tree ``raise:<f>`` and its on-tree pseudo-harmonic neighbour must map
# to the SAME key so a blueprint lookup on an off-tree line hits the row
# on-tree play wrote.  This holds by construction because
# ``get_canonical_actions``, ``_get_available_raise_sizes`` and
# ``_canonicalize_history`` all format raises via the identical
# ``f"raise:{f}"`` over the same ``RAISE_SIZES_BY_STAGE`` floats — so the
# canonicalised token string equals the on-tree token string and hits the
# same alphabet byte.  ``_blueprint_info_set`` therefore differs from
# ``_compute_info_set`` only in passing ``_canonicalize_history(...)``.
INFO_SET_ENCODING: str = "v2-compact-bytes"
"""Version tag for the info-set key encoding.

Persisted in ``server_state.pkl`` and checked as a structural key on
resume / warm-start so a blueprint written under a different encoding can
never be silently opened under this one (every lookup would miss → uniform
strategy everywhere, a silent catastrophe that looks like "untrained").
"""

_STAGE_ID: Dict[str, int] = {"pre_flop": 0, "flop": 1, "turn": 2, "river": 3}
_RAW_TOKEN_MARK: int = 0xFF
"""Prefix for a token absent from the per-stage alphabet — a length-framed
raw-bytes fallback that keeps the encoding injective (and crash-free) if the
action grid ever drifts.  Never fires for a valid canonicalised token."""

_INFO_SET_DEFAULT: bytes = b"\x00__default__"
"""Sentinel key returned when the hole+board is missing from the LUT at a
terminal / show-down node (mirrors the old JSON sentinel string)."""


def _canonical_action_tokens(stage: str) -> List[str]:
    """Canonical action tokens for ``stage`` (mirrors ``get_canonical_actions``).

    Built from :data:`RAISE_SIZES_BY_STAGE` with the identical ordering and
    ``f"raise:{f}"`` formatting used by
    :meth:`PokerEnv.get_canonical_actions`; a unit test asserts the two agree
    so the alphabet can never silently drift from the canonical action set.
    """
    cfg = RAISE_SIZES_BY_STAGE[stage]
    fracs = sorted(
        set(cfg.get("first_raise", [])) | set(cfg.get("subsequent_raise", []))
    )
    return ["fold", "call", "all_in"] + [f"raise:{f}" for f in fracs]


# Per-stage ``token -> 1-byte code``.  ``"skip"`` reserved as 0; canonical
# actions occupy 1..k (k <= 3 + len(grid), always one byte).
_ACTION_BYTE: Dict[str, Dict[str, int]] = {
    stage: {"skip": 0, **{a: i + 1 for i, a in enumerate(_canonical_action_tokens(stage))}}
    for stage in _STAGE_ID
}


def _put_uvarint(buf: bytearray, value: int) -> None:
    """Append ``value`` (a non-negative int) to ``buf`` as unsigned LEB128."""
    value = int(value)
    if value < 0:
        raise ValueError(f"info-set varint requires non-negative, got {value}")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buf.append(byte | 0x80)
        else:
            buf.append(byte)
            return


def encode_info_set(cluster: int, history_items) -> bytes:
    """Return the compact injective ``bytes`` key for ``(cluster, history)``.

    Parameters
    ----------
    cluster : int
        Card-cluster id for the acting hand (from the LUT).
    history_items : Iterable[tuple[str, Sequence[str]]]
        ``(stage, actions)`` pairs in play order — typically
        ``self._history.items()`` (raw) or
        ``self._canonicalize_history(self._history)`` (blueprint lookups).
    """
    buf = bytearray()
    _put_uvarint(buf, int(cluster))
    for stage, actions in history_items:
        buf.append(_STAGE_ID[stage])
        _put_uvarint(buf, len(actions))
        table = _ACTION_BYTE[stage]
        for token in actions:
            code = table.get(token)
            if code is None:
                raw = str(token).encode("utf-8")
                buf.append(_RAW_TOKEN_MARK)
                _put_uvarint(buf, len(raw))
                buf += raw
            else:
                buf.append(code)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Optional compiled-core encoder (Phase 1b)
# ---------------------------------------------------------------------------
# Keep the pure-Python encoder as the byte-exact oracle, then swap the public
# ``encode_info_set`` for the Cython kernel when it is built AND enabled
# (``PLURIBUS_CORE_KERNELS`` includes ``info_set``).  ``configure()`` dumps the
# LIVE ``_STAGE_ID`` / ``_ACTION_BYTE`` (derived here from
# ``RAISE_SIZES_BY_STAGE``) into the kernel so its alphabet can never drift from
# the grid the tables were written under.  ``_compute_info_set`` /
# ``_blueprint_info_set`` look this name up as a module global at call time, so
# the swap is transparent; tests monkeypatch it directly.
_encode_info_set_py = encode_info_set

try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    if _CORE_AVAILABLE and _kernel_enabled("info_set"):
        from poker_ai._core._infoset import (
            configure as _core_infoset_configure,
            encode_info_set as _core_encode_info_set,
        )
        _core_infoset_configure(_STAGE_ID, _ACTION_BYTE, _RAW_TOKEN_MARK)
        encode_info_set = _core_encode_info_set
except ImportError:
    pass  # extension / flags unavailable -> keep the pure-Python reference


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def new_game(
    n_players: int,
    card_info_lut: Optional[dict] = None,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
) -> PokerEnv:
    """Create a new poker game.

    The deck is determined automatically from ``card_info_lut``: the rank
    bounds are read from the pre-flop keys so the game always uses the same
    card set the LUT was built for.  When no LUT is provided (e.g. in tests),
    a full 52-card deck is used.

    Loading a LUT from disk is the caller's responsibility — use
    :func:`information_abstraction.load_info_set_lut` and pass the result
    via ``card_info_lut``.

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut : dict, optional
        Pre-loaded card cluster lookup table.  When ``None`` or empty,
        a full 52-card deck is used.
    small_blind : int
        Small blind amount.
    big_blind : int
        Big blind amount.
    initial_chips : int
        Starting chip count per player.

    Returns
    -------
    PokerEnv
        Initial game state with the deck matching the supplied LUT.
    """
    if card_info_lut is None:
        card_info_lut = {}

    low_card_rank, high_card_rank = 2, 14  # default: full deck
    if card_info_lut:
        from environment.utils import card_rank_int as _rank
        preflop = card_info_lut.get("pre_flop", {})
        if isinstance(preflop, dict) and preflop:
            all_ranks = [_rank(int(c)) for combo in preflop for c in combo]
            low_card_rank, high_card_rank = min(all_ranks), max(all_ranks)

    players = [
        Player(player_i=i, initial_chips=initial_chips)
        for i in range(n_players)
    ]
    env = PokerEnv(
        players=players,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    env.card_info_lut = card_info_lut
    return env


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PokerEnv:
    """Poker game environment for CFR/MCCFR training.

    Supports any deck size via ``low_card_rank`` and ``high_card_rank``.
    Cards are represented as 32-bit integers throughout (see
    ``utils.py`` for the encoding).

    The env is advanced with the make/undo pair ``step_in_place`` /
    ``undo``: ``step_in_place`` mutates the env in place and returns an
    ``UndoToken``; ``undo`` reverses it.  A caller that needs both the
    pre- and post-action state ``copy.deepcopy`` the env first.

    Attributes
    ----------
    players : list[Player]
        All players at the table, in seating order.
    pot : Pot
        Shared pot tracking per-player chip contributions.
    deck : Deck
        The shuffled deck for this hand.
    community_cards : tuple[int, ...]
        Board cards dealt so far, as card integers.
    small_blind : int
        Small blind amount.
    big_blind : int
        Big blind amount.
    card_info_lut : dict
        Card cluster lookup table used to compute ``info_set``.
    """

    def __init__(
        self,
        players: List[Player],
        small_blind: int = 50,
        big_blind: int = 100,
        low_card_rank: int = 2,
        high_card_rank: int = 14,
    ):
        """Initialise the environment and deal the first hand.

        The caller owns LUT loading — assign ``env.card_info_lut`` after
        construction (or use :func:`new_game` as the factory).

        Parameters
        ----------
        players : list[Player]
            Pre-constructed player objects.
        small_blind : int
            Small blind amount.
        big_blind : int
            Big blind amount.
        low_card_rank : int
            Lowest rank in the deck (2=Two, ..., 14=Ace).
        high_card_rank : int
            Highest rank in the deck.
        """
        n_players = len(players)
        if n_players < 2:
            raise ValueError(
                f"At least 2 players required, got {n_players}."
            )
        if low_card_rank < 2 or high_card_rank > 14:
            raise ValueError(
                f"Card ranks must be in [2, 14], got "
                f"low={low_card_rank}, high={high_card_rank}."
            )
        if low_card_rank > high_card_rank:
            raise ValueError(
                f"low_card_rank ({low_card_rank}) must be <= "
                f"high_card_rank ({high_card_rank})."
            )
        n_ranks = high_card_rank - low_card_rank + 1
        n_cards = n_ranks * 4
        min_cards = n_players * 2 + 5
        if n_cards < min_cards:
            raise ValueError(
                f"Deck has {n_cards} cards but need at least {min_cards} "
                f"({n_players} players × 2 + 5 community). "
                f"Use fewer players or a larger deck."
            )

        # Config (immutable after init — skipped in __deepcopy__)
        self._low_card_rank: int = low_card_rank
        self._high_card_rank: int = high_card_rank
        self._initial_n_chips: int = players[0].n_chips
        # Per-seat starting stack, so ``payout`` nets each seat against its OWN
        # start rather than seat 0's.  Identical to ``_initial_n_chips`` for the
        # equal-stack hands every current caller deals (``new_game``), but the
        # only correct source once stacks can differ (unequal effective stacks in
        # a search subgame rebuilt from remaining chips, tournament/session
        # carryover): netting everyone against seat 0's stack yields a non
        # zero-sum ``payout`` — see ``payout``.  Player.n_chips is mutated during
        # the hand, so the per-seat initial must be captured here at construction.
        self._initial_chips_by_seat: Tuple[int, ...] = tuple(
            p.n_chips for p in players
        )
        self.small_blind: int = small_blind
        self.big_blind: int = big_blind
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0, "flop": 1, "turn": 2,
            "river": 3, "show_down": 4,
        }

        # LUT is populated by the caller (see `new_game`); default empty
        # for tests that don't need cluster-id lookups.
        self.card_info_lut: dict = {}

        # Off-tree action injections (private — callers use inject_action /
        # reset_overlay / legal_actions). Keyed by public state
        # (betting_stage, history-tuple) so the overlay is the same for
        # every actor that reaches a given public node.
        self._extra_legal_actions: Dict[
            Tuple[str, Tuple[str, ...]], FrozenSet[str]
        ] = {}
        # Monotone version of the overlay above, held in a 1-element list so the
        # whole deepcopy lineage shares ONE counter by reference (exactly like
        # ``_extra_legal_actions`` itself).  ``inject_action`` / ``reset_overlay``
        # bump it; the per-env ``_legal_actions_cache`` stamps each entry with the
        # version it was built at and treats a mismatch as a miss.  This is what
        # makes an inject/reset on ANY env in the lineage invalidate every
        # lineage env's cache — the overlay is shared, so its invalidation must be
        # too (a bare per-env ``pop``/``clear`` left sibling caches stale).
        self._overlay_version: List[int] = [0]

        # Live game state (deep-copied by __deepcopy__)
        self.players: List[Player] = players
        self.pot: Pot = Pot(n_players)
        self.deck: Deck = Deck(low_card_rank, high_card_rank)
        self.community_cards: tuple = ()
        # Snapshot of a decision-free (all-in) board runout, recorded when a
        # hand force-resolves to showdown over an *incomplete* board (§6.4
        # decision-free runout equity).  ``None`` unless the hand ended that
        # way.  Layout: ``(prefix_board, pot_contributions, active_mask)`` —
        # all immutable — captured *before* ``compute_winners`` resets the pot,
        # so :meth:`runout_equity` can integrate over every board completion.
        self._runout_info: Optional[Tuple] = None
        # Per-seat pot contributions captured at the terminal *before*
        # ``compute_winners`` resets the pot (``None`` until the hand ends).  The
        # authoritative source for the matched/contested stake at a terminal — the
        # smaller of two contributions in heads-up is the winner-takes amount (the
        # bigger stack's uncalled excess is the difference).  Consumed by the
        # range-vs-range showdown evaluator so the search need not reconstruct it.
        self._terminal_contributions: Optional[Tuple] = None
        # Number of community cards that were actually on the board when betting
        # ended, captured *before* the force-deal that completes the runout on a
        # fold/all-in (``None`` until the hand ends).  The engine deals the board
        # out to five at every terminal, so ``len(community_cards)`` no longer
        # distinguishes the street the hand ended on; this preserves it.  Consumed
        # by :meth:`vector_payout` so a pre-river fold does card removal against
        # the board it actually saw, not the force-dealt completion.
        self._terminal_board_len: Optional[int] = None

        # Round setup: reset pot, assign order, post blinds
        self.pot.reset()
        dynamics.assign_order(self)
        dynamics.assign_blinds(self)

        # Deal private cards
        self.deck.deal_private_cards(self.players)

        # CFR tracking state
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._betting_stage: str = "pre_flop"

        # Player-order LUT for each stage
        player_i_order: List[int] = list(range(n_players))
        self.players[0].is_small_blind = True
        self.players[1].is_big_blind = True
        # In heads-up the dealer is the small blind (players[0]); for 3+
        # players the dealer sits at the end of the list (players[-1]).
        if n_players == 2:
            self.players[0].is_dealer = True
        else:
            self.players[-1].is_dealer = True
        # Post-flop, action opens with the first seat left of the button.  For
        # 3+ players that is the small blind (seat 0), so the natural seat order
        # is correct.  Heads-up is the exception: the button *is* the small
        # blind (seat 0) and acts LAST post-flop, so the big blind (seat 1)
        # leads — the post-flop order is the seat order reversed.  (Pre-flop the
        # button/SB still acts first, which the ``[2:]+[:2]`` rotation already
        # yields as ``[0, 1]`` for two players.)
        postflop_order = (
            player_i_order[::-1] if n_players == 2 else player_i_order
        )
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop":     postflop_order,
            "turn":     postflop_order,
            "river":    postflop_order,
            "show_down": postflop_order,
            "terminal": postflop_order,
        }

        # Betting round counters
        self._skip_counter: int = 0
        self._last_raise_amount: int = self.big_blind
        # Memo of ``legal_actions`` keyed by public state (lazily filled — see
        # :attr:`legal_actions`).  Each entry is ``(overlay_version, actions)``;
        # an entry whose stamped version differs from the current shared
        # ``_overlay_version`` is stale (an inject/reset happened somewhere in the
        # lineage) and is recomputed.  The public state is the rest of the key, so
        # ``step_in_place`` / ``undo`` still need no cache bookkeeping.
        self._legal_actions_cache: Dict[
            Tuple, Tuple[int, List[Optional[str]]]
        ] = {}
        # Single-value memo for ``_current_public_state`` (rebuilds the full
        # cross-street history tuple, called many times per node as the
        # legal-actions cache key, public_key, and overlay lookups).  It is a
        # pure function of ``_betting_stage`` + ``_history``, both mutated only
        # by ``_apply_action_in_place`` / ``undo``, which bump ``_state_version``
        # on exit — so a version mismatch is the sole invalidation signal.
        self._state_version: int = 0
        self._public_state_cache: Optional[Tuple[int, Tuple]] = None
        self._reset_betting_round_state()

        # Mark the first player to act
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

    # ------------------------------------------------------------------
    # Deep-copy optimisation
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo: dict) -> PokerEnv:
        """Return a deep copy, sharing immutable config fields by reference.

        Configuration scalars (blind sizes, rank bounds, etc.) are shared
        rather than copied. All mutable game state (players, pot, deck,
        history) is deep-copied. ``card_info_lut`` is shared by reference:
        it is read-only (never mutated per-env) and is the same loaded
        object for the whole session, so copying it would be pure waste.

        Parameters
        ----------
        memo : dict
            Standard memo dict passed by ``copy.deepcopy``.

        Returns
        -------
        PokerEnv
            A new instance with an independent copy of mutable state.
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        # Immutable / read-only shared state — share references, no copy.
        # `_extra_legal_actions` is mutated in place by inject_action, so
        # every env in a deepcopy lineage sees the same augmented game tree;
        # `_overlay_version` rides alongside it (shared by reference too) so a
        # bump on any env is visible to every lineage env's cache-staleness
        # check; `card_info_lut` is read-only and large, so it is shared too.
        for attr in (
            "small_blind", "big_blind", "_low_card_rank", "_high_card_rank",
            "_initial_n_chips", "_initial_chips_by_seat",
            "_betting_stage_to_round", "_player_i_lut",
            "_extra_legal_actions", "_overlay_version",
            "card_info_lut",
        ):
            object.__setattr__(new, attr, getattr(self, attr))
        # Mutable game state — deep copy
        for attr in (
            "players", "pot", "deck", "community_cards",
            "_history", "_betting_stage",
            "_skip_counter",
            "_last_raise_amount",
            "_n_actions", "_n_raises", "_player_i_index",
            "_n_players_started_round", "_runout_info",
            "_terminal_contributions", "_terminal_board_len",
        ):
            object.__setattr__(new, attr, copy.deepcopy(getattr(self, attr), memo))
        # The legal-action memo is a pure derived cache; start the copy empty and
        # let it refill lazily.  Kept per-env (not shared): entries are stamped
        # with the shared ``_overlay_version``, so an overlay change on any
        # lineage env is caught by the version check rather than by sharing the
        # dict — and an empty copy can never carry stale entries anyway.
        object.__setattr__(new, "_legal_actions_cache", {})
        # Derived public-state memo: start the copy with a fresh, empty cache
        # (its own version counter) so it can never serve a stale entry.
        object.__setattr__(new, "_state_version", 0)
        object.__setattr__(new, "_public_state_cache", None)
        return new

    # ------------------------------------------------------------------
    # Core CFR interface
    # ------------------------------------------------------------------

    def _apply_action_in_place(
        self, action_str: Optional[str], settle_winners: bool = True
    ) -> None:
        """Apply ``action_str`` by mutating ``self`` in place.

        The forward game-logic core used by :meth:`step_in_place` (which
        calls it on ``self`` after snapshotting an undo token).  Operates
        entirely on ``self``; undo bookkeeping is the caller's concern.

        ``settle_winners`` (default ``True``) controls terminal settlement: when
        a hand ends it normally ranks the concrete dealt hands and distributes
        chips (:func:`dynamics.compute_winners`).  A caller that never reads the
        concrete result — the vector regime values terminals over *ranges* via
        :meth:`vector_payout` — passes ``False`` to skip the ranking/distribution
        and only snapshot ``_terminal_contributions`` (the matched stake the
        payout needs); ~15-17% of a vector iteration otherwise goes to a concrete
        showdown whose result is discarded.
        """
        original_action = action_str
        if action_str not in self.legal_actions:
            logger.warning(
                "Invalid action '%s'. Legal: %s. Mapping to closest.",
                action_str, self.legal_actions,
            )
            action_str = self._map_to_closest_legal_action(action_str)
            logger.info("Mapped '%s' -> '%s'", original_action, action_str)

        if action_str is None:
            assert (
                not self.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            self.current_player.call(
                players=self.players, pot=self.pot
            )
            logger.debug("calling")
        elif action_str == "fold":
            self.current_player.fold()
        elif action_str == "all_in":
            n_chips_to_add = self.current_player.n_chips
            biggest_bet = max(p.n_bet_chips for p in self.players)
            current_bet = self.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self._last_raise_amount:
                self._last_raise_amount = actual_raise_amount
                self._n_raises += 1
            logger.debug("going all-in with %d chips", n_chips_to_add)
            self.current_player.raise_to(pot=self.pot, n_chips=n_chips_to_add)
        elif action_str.startswith("raise:"):
            pot_fraction = float(action_str.split(":")[1])
            # Actions reaching this branch are guaranteed in
            # ``legal_actions``: canonical sizes were filtered to
            # >= min raise in ``_get_available_raise_sizes`` and
            # injected sizes were validated by
            # ``_raise_fraction_is_playable``.  ``enforce_minimum=True``
            # therefore agrees with ``False`` on every valid input and
            # keeps the chip math consistent with ``chips_to_add`` /
            # ``string_for_chips`` / ``_raise_fraction_is_playable``.
            n_chips_to_add = self._compute_raise_chip_amount(
                pot_fraction, enforce_minimum=True
            )
            biggest_bet = max(p.n_bet_chips for p in self.players)
            current_bet = self.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= self._last_raise_amount:
                self._last_raise_amount = actual_raise_amount
            logger.debug("adding %d chips to pot (action: %s)", n_chips_to_add, action_str)
            self.current_player.raise_to(pot=self.pot, n_chips=n_chips_to_add)
            self._n_raises += 1
        else:
            raise ValueError(
                f"Unrecognised action '{action_str}'. "
                "Expected 'fold', 'call', 'raise:<fraction>' or 'all_in'."
            )

        skip_actions = ["skip"] * self._skip_counter
        self._history[self.betting_stage] += skip_actions
        self._history[self.betting_stage].append(action_str)
        self._n_actions += 1
        self._skip_counter = 0

        # Board length as of the street the just-applied action was made on.  A
        # hand-ending action (fold/last call) can also *close* the round, which
        # advances the stage and deals the next street before the terminal is
        # detected below — so this is captured up front, not read off the
        # post-advance ``community_cards`` (see ``_terminal_board_len``).
        board_len_at_action = len(self.community_cards)

        while True:
            self._move_to_next_player()
            # Hand-ending check FIRST, before any round-closing stage advance, so
            # the runout snapshot (§6.4) sees the board betting actually ended on.
            # A player facing an unmatched all-in they can still act on is NOT
            # terminal here — :meth:`_hand_over` returns ``False`` until they call
            # or fold (the fix: the old ``n_players_with_moves() == 1`` shortcut
            # ended the hand before the opponent could respond to an all-in).
            if self._hand_over():
                if self._betting_stage not in {"show_down", "terminal"}:
                    self._betting_stage = "terminal"
                self._settle_terminal(board_len_at_action, settle_winners)
                break
            finished_betting = not dynamics.more_betting_needed(self)
            if finished_betting and self.all_players_have_actioned:
                self._increment_stage()
                self._reset_betting_round_state()
                if self._betting_stage == "show_down":
                    # Final (river) betting round completed with >=2 players still
                    # in — a normal showdown over the already-complete board.
                    self._settle_terminal(board_len_at_action, settle_winners)
                    break
            if not self.current_player.is_active:
                self._skip_counter += 1
                continue
            if self.current_player.is_all_in:
                # An all-in player has no betting decision; advance past them
                # (they stay in the hand for the showdown).  Reached only with
                # >=2 live players still to act — otherwise :meth:`_hand_over`
                # above already ended the hand.
                self._skip_counter += 1
                continue
            break

        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

        # State changed (history/stage/players); invalidate the public-state memo.
        self._state_version += 1

    def step_in_place(
        self, action_str: Optional[str], *, settle_winners: bool = True
    ) -> "UndoToken":
        """Apply ``action_str`` by **mutating this env in place**, returning
        an :class:`UndoToken` that :meth:`undo` uses to restore the
        pre-action state.

        The sole way to advance the env, for the depth-first CFR / search
        inner loop: ``token = env.step_in_place(a); ...recurse on env...;
        env.undo(token)``.  The token snapshots exactly the mutable per-hand
        state ``__deepcopy__`` copies, so a ``step_in_place`` followed by an
        ``undo`` leaves ``self`` field-identical to its pre-action state.
        A caller that needs the pre- and post-action env at once
        ``copy.deepcopy`` first.

        ``settle_winners`` (default ``True``) keeps the normal concrete terminal
        settlement; pass ``False`` from a range-valued traversal (vector regime)
        to skip the discarded hand ranking/chip distribution — see
        :meth:`_apply_action_in_place`.
        """
        token = self._capture_undo_token()
        self._apply_action_in_place(action_str, settle_winners)
        return token

    def undo(self, token: "UndoToken") -> None:
        """Reverse the most recent :meth:`step_in_place`, restoring every
        mutable field from ``token``.

        Tokens are strictly LIFO and single-use: call ``undo`` in the
        reverse order of the ``step_in_place`` calls that produced them.
        """
        self._betting_stage = token.betting_stage
        self._skip_counter = token.skip_counter
        self._last_raise_amount = token.last_raise_amount
        self._n_actions = token.n_actions
        self._n_raises = token.n_raises
        self._player_i_index = token.player_i_index
        self._n_players_started_round = token.n_players_started_round
        self._runout_info = token.runout_info
        self._terminal_contributions = token.terminal_contributions
        self._terminal_board_len = token.terminal_board_len
        self.community_cards = token.community_cards
        self.deck.restore(token.deck_cursor)
        self.pot.restore(token.pot_chips)
        for player, snap in zip(self.players, token.player_states):
            player.restore_mutable(snap)
        # Restore _history wholesale (preserving exact key presence) so a
        # later info_set build sees the pre-step history, not a defaultdict
        # key spuriously materialised during the step.
        self._history.clear()
        for stage, actions in token.history.items():
            self._history[stage] = list(actions)

        # State restored (history/stage/players); invalidate the public-state memo.
        self._state_version += 1

    def _capture_undo_token(self) -> "UndoToken":
        """Snapshot the mutable per-hand state before an in-place step.

        Mirrors the deep-copied field set of :meth:`__deepcopy__`; copies
        every mutable container so the returned token is independent of the
        subsequent mutation.  ``_history`` is read with ``.get`` semantics
        (via ``items()``) so the snapshot never materialises a defaultdict
        key.
        """
        return UndoToken(
            betting_stage=self._betting_stage,
            skip_counter=self._skip_counter,
            last_raise_amount=self._last_raise_amount,
            n_actions=self._n_actions,
            n_raises=self._n_raises,
            player_i_index=self._player_i_index,
            n_players_started_round=self._n_players_started_round,
            community_cards=self.community_cards,
            deck_cursor=self.deck.capture(),
            pot_chips=self.pot.capture(),
            player_states=[p.capture_mutable() for p in self.players],
            history={stage: list(actions) for stage, actions in self._history.items()},
            runout_info=self._runout_info,
            terminal_contributions=self._terminal_contributions,
            terminal_board_len=self._terminal_board_len,
        )

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    def _move_to_next_player(self) -> None:
        """Advance ``_player_i_index`` to the next player, wrapping around."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _reset_betting_round_state(self) -> None:
        """Reset per-round counters and advance to the first active player."""
        self._n_actions = 0
        self._n_raises = 0
        self._last_raise_amount = self.big_blind
        self._player_i_index = 0
        # Count only players who can still act this round (active AND not
        # all-in).  A player already all-in from an earlier street never acts
        # here yet stays ``is_active``; counting them (``n_active_players``)
        # made ``all_players_have_actioned`` unreachable after one lap, so the
        # live players were polled a second time.  Players who go all-in or
        # fold *during* the round were counted at its start (they had moves)
        # and do bump ``_n_actions``, so the ``>=`` gate stays correct for them.
        self._n_players_started_round = dynamics.n_players_with_moves(self)
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self) -> None:
        """Advance ``_betting_stage`` and deal the corresponding community cards.

        Transitions: pre_flop → flop (deal 3), flop → turn (deal 1),
        turn → river (deal 1), river → show_down (no deal).

        Also resets each player's ``n_bet_chips`` to zero so that
        bet-equality checks start fresh for the new betting round.
        """
        if self._betting_stage == "pre_flop":
            self._betting_stage = "flop"
            self.community_cards += self.deck.deal_community(3)
        elif self._betting_stage == "flop":
            self._betting_stage = "turn"
            self.community_cards += self.deck.deal_community(1)
        elif self._betting_stage == "turn":
            self._betting_stage = "river"
            self.community_cards += self.deck.deal_community(1)
        elif self._betting_stage == "river":
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")
        for player in self.players:
            player.n_bet_chips = 0

    def _hand_over(self) -> bool:
        """True when no active player has any further betting decision.

        The hand ends when every player but one has folded, or every remaining
        active player is all-in, or the single active player who still has chips
        has already matched the largest bet (nothing left to call).

        Crucially — and unlike the old ``n_players_with_moves() == 1`` shortcut
        this replaces — a lone live player who still owes an outstanding call
        (the classic case: facing an all-in they have not yet called) is **not**
        treated as terminal.  They must first get to call or fold; only then does
        the hand resolve.  This is the fix for the mis-written contract where an
        all-in ended the hand before the opponent could respond.
        """
        active = [p for p in self.players if p.is_active]
        if len(active) <= 1:
            return True
        live = [p for p in active if not p.is_all_in]
        if not live:
            return True
        if len(live) == 1:
            max_bet = max(p.n_bet_chips for p in active)
            return live[0].n_bet_chips >= max_bet
        return False

    def _settle_terminal(self, board_len_at_action: int, settle_winners: bool) -> None:
        """Deal out any remaining board and settle a terminal node.

        When the board is incomplete and >=2 players still contest it, snapshots
        the pre-runout state (board prefix, contributions, active mask) for the
        decision-free runout-equity integration (§6.4) *before* dealing and
        settling.  Then deals the board to five and either runs the concrete
        winner settlement (``settle_winners=True``) or — for range-valued (vector
        regime) callers — only snapshots ``_terminal_contributions``.
        """
        if self._terminal_board_len is None:
            self._terminal_board_len = board_len_at_action
        cards_needed = 5 - len(self.community_cards)
        if cards_needed > 0:
            if dynamics.n_active_players(self) >= 2:
                self._runout_info = (
                    tuple(self.community_cards),
                    tuple(self.pot.capture()),
                    tuple(p.is_active for p in self.players),
                )
            self.community_cards += self.deck.deal_community(cards_needed)
        if settle_winners:
            dynamics.compute_winners(self)
        else:
            self._terminal_contributions = tuple(self.pot.capture())

    def _map_to_closest_legal_action(self, invalid_action: str) -> str:
        """Map an out-of-range action to the closest legal one.

        For raise actions, selects the legal raise with the nearest
        pot fraction. Falls back to ``all_in``, then ``call``, then
        ``fold`` if no closer match exists.

        Parameters
        ----------
        invalid_action : str
            Action string that is not in ``self.legal_actions``.

        Returns
        -------
        str
            A legal action string.
        """
        legal = self.legal_actions
        if invalid_action == "all_in" and "all_in" in legal:
            return "all_in"
        if invalid_action and invalid_action.startswith("raise:"):
            try:
                target = float(invalid_action.split(":")[1])
                legal_raises = [a for a in legal if a and a.startswith("raise:")]
                if legal_raises:
                    return min(legal_raises, key=lambda a: abs(float(a.split(":")[1]) - target))
                elif "all_in" in legal:
                    return "all_in"
            except (ValueError, IndexError):
                pass
        if "all_in" in legal:
            return "all_in"
        if "call" in legal:
            return "call"
        if "fold" in legal:
            return "fold"
        if legal and legal[0] is not None:
            return legal[0]
        return "fold"

    # ------------------------------------------------------------------
    # Raise size helpers
    # ------------------------------------------------------------------

    def _compute_raise_chip_amount(
        self, pot_fraction: float, enforce_minimum: bool = True
    ) -> int:
        """Compute chips to add for a raise of size ``pot_fraction × pot``.

        Parameters
        ----------
        pot_fraction : float
            Raise size as a multiple of the current pot.
        enforce_minimum : bool
            If ``True``, clamps the result to at least the call amount
            plus the last raise increment.

        Returns
        -------
        int
            Total chips the current player must add to the pot.
        """
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        n_chips_to_add = math.ceil(self.pot_size * pot_fraction)
        if enforce_minimum:
            n_chips_to_add = max(n_chips_to_add, n_chips_to_call + self._last_raise_amount)
        return n_chips_to_add

    def _get_available_raise_sizes(self) -> List[str]:
        """Return legal raise action strings for the current player and stage.

        Returns
        -------
        list[str]
            Raise action strings of the form ``"raise:<fraction>"`` that
            are legal given the player's stack, the current round's raise
            count, and the minimum raise increment. ``"all_in"`` is
            appended when the player can go all-in but not make a
            standard raise.
        """
        if self._betting_stage in {"terminal", "show_down"}:
            return []
        stage_config = RAISE_SIZES_BY_STAGE.get(self._betting_stage, {})
        fractions = (
            stage_config.get("first_raise", [1.0])
            if self._n_raises == 0
            else stage_config.get("subsequent_raise", [1.0])
        )
        player = self.current_player
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - player.n_bet_chips
        chips_available = player.n_chips
        added: set = set()
        raise_actions: List[str] = []
        for fraction in fractions:
            chips_raw = self._compute_raise_chip_amount(fraction, enforce_minimum=False)
            actual_raise = chips_raw - n_chips_to_call
            if actual_raise < self._last_raise_amount:
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
    # Pseudo-harmonic action translation (Ganzfried & Sandholm 2013)
    # ------------------------------------------------------------------
    # Off-tree raise sizes are mapped onto a static abstraction grid.  The
    # mapping is purely on pot-fractions, so the grid depends only on
    # (stage, first-vs-subsequent raise) — never on pot/stack — which lets
    # history canonicalisation run as a pure history walk (§6.3).

    @staticmethod
    def _pseudo_harmonic_prob(a: float, x: float, b: float) -> float:
        """P(map ``x`` to ``A``) for ``A < x < B`` (Ganzfried-Sandholm).

        ``((b - x) * (1 + a)) / ((b - a) * (1 + x))``.  Pure arithmetic;
        the caller guarantees ``a < x < b`` (via
        :meth:`_pseudo_harmonic_neighbours`).
        """
        return ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x))

    @staticmethod
    def _abstraction_fractions(
        stage: str,
        raise_index: int,
        sizes_by_stage: Dict[str, Dict[str, List[float]]] = RAISE_SIZES_BY_STAGE,
    ) -> List[float]:
        """Sorted abstraction fractions for a ``(stage, raise_index)`` cell.

        ``raise_index == 0`` selects the ``"first_raise"`` list, otherwise
        ``"subsequent_raise"``.  Depends only on the size table — no chip
        state — so it is valid for canonicalising a historical action.
        """
        cell = sizes_by_stage.get(stage, {})
        key = "first_raise" if raise_index == 0 else "subsequent_raise"
        return sorted(cell.get(key, []))

    @staticmethod
    def _pseudo_harmonic_neighbours(
        fractions: List[float], x: float
    ) -> Tuple[Optional[float], Optional[float], float]:
        """Locate ``x`` in the sorted grid ``fractions``; return ``(A, B, P_A)``.

        - empty grid           -> ``(None, None, 1.0)``  (caller leaves x as-is)
        - ``x <= fractions[0]`` -> ``(None, fractions[0], 0.0)``  (below: always B)
        - ``x >= fractions[-1]`` -> ``(fractions[-1], None, 1.0)`` (above: always A)
        - ``x == f`` exactly    -> ``(f, f, 1.0)``  (on-tree identity)
        - otherwise A < x < B bracketing pair, ``P_A`` from the formula.
        """
        if not fractions:
            return (None, None, 1.0)
        if x <= fractions[0]:
            return (None, fractions[0], 0.0)
        if x >= fractions[-1]:
            return (fractions[-1], None, 1.0)
        for i in range(len(fractions) - 1):
            a, b = fractions[i], fractions[i + 1]
            if x == a:
                return (a, a, 1.0)
            if a < x < b:
                return (a, b, PokerEnv._pseudo_harmonic_prob(a, x, b))
        # x == fractions[-1] is caught by the >= branch above; any
        # remaining exact hit on an interior point returns identity.
        return (x, x, 1.0)

    def _translate_fraction(
        self,
        x: float,
        stage: str,
        raise_index: int,
        *,
        randomized: bool,
        rng: Optional["np.random.Generator"] = None,
        sizes_by_stage: Dict[str, Dict[str, List[float]]] = RAISE_SIZES_BY_STAGE,
    ) -> float:
        """Map off-tree pot-fraction ``x`` onto the abstraction grid.

        ``randomized=True`` (round-1 mapping, §5): sample ``A`` with
        probability ``P_A`` else ``B``; an explicit ``rng`` is required so
        a fixed seed is reproducible end-to-end.  ``randomized=False``
        (deterministic — history canonicalisation / continuation-rollout
        blueprint lookups): return ``A`` iff ``P_A >= 0.5`` else ``B``.

        An exact grid hit returns that fraction with **no** ``rng`` draw;
        an empty grid returns ``x`` unchanged.
        """
        fractions = self._abstraction_fractions(stage, raise_index, sizes_by_stage)
        a, b, p_a = self._pseudo_harmonic_neighbours(fractions, x)
        if a is None and b is None:
            return x
        if a is None:
            return b
        if b is None or a == b:
            return a
        if randomized:
            if rng is None:
                raise ValueError(
                    "_translate_fraction: randomized=True requires an rng"
                )
            return a if rng.random() < p_a else b
        return a if p_a >= 0.5 else b

    # ------------------------------------------------------------------
    # Public chip <-> action conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_raise_fraction(action: str, caller: str) -> float:
        """Parse and validate the fraction of a ``"raise:<f>"`` action.

        Centralises the shape check used by every public method that
        consumes a raise action string (``inject_action``,
        ``chips_to_add``).  A fraction must be parseable, finite, and
        positive; anything else is a caller bug and raises
        :class:`ValueError`.

        ``caller`` is the method name; it is interpolated into error
        messages so the failure site is obvious to the user.
        """
        try:
            fraction = float(action.split(":", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"{caller}: unparseable raise fraction in {action!r}"
            )
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise ValueError(
                f"{caller}: raise fraction must be a positive finite "
                f"float, got {fraction}"
            )
        return fraction

    def canonical_raise_fractions(self) -> List[float]:
        """Currently-playable blueprint raise fractions for the actor.

        Mirrors the gating that :meth:`legal_actions` applies before
        delegating to :meth:`_get_available_raise_sizes`: an inactive
        player, a call amount that meets or exceeds the stack, having
        already hit :data:`MAX_RAISES_PER_ROUND`, or **facing a lone
        all-in** (no other live player could call a raise) all yield an
        empty list.  Stack-clamping and min-raise enforcement come from
        ``_get_available_raise_sizes`` itself.
        """
        if not self.current_player.is_active:
            return []
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        if n_chips_to_call >= self.current_player.n_chips:
            return []
        if self._n_raises >= MAX_RAISES_PER_ROUND:
            return []
        if dynamics.n_players_with_moves(self) < 2:
            # Facing a lone all-in — a raise/over-shove would only be returned
            # uncalled, so ``legal_actions`` offers only call/fold here and this
            # helper must agree (``_compute_legal_actions`` gates raises on the
            # same ``n_players_with_moves() >= 2`` condition).
            return []
        return [
            float(s.split(":", 1)[1])
            for s in self._get_available_raise_sizes()
            if s.startswith("raise:")
        ]

    def chips_to_add(self, action: str) -> int:
        """Chips the actor must add to play ``action``.

        Public inverse of :meth:`step_in_place`'s chip math.  Accepts
        the same action vocabulary as :meth:`step_in_place` and
        :meth:`inject_action` and applies the same shape validation
        on raise fractions (must be finite and positive).

        Mapping:

        - ``"fold"``      -> ``0``
        - ``"call"``      -> ``biggest_bet - actor.n_bet_chips`` (``0`` when
          the actor is already the highest bettor — i.e. a check)
        - ``"all_in"``    -> ``actor.n_chips`` (full remaining stack)
        - ``"raise:<f>"`` -> :meth:`_compute_raise_chip_amount` with
          min-raise enforcement

        Raises
        ------
        ValueError
            On an unknown action prefix, an unparseable ``<f>``, or a
            non-positive / non-finite ``<f>``.  Same shape and
            message style as :meth:`inject_action`.
        """
        if action == "fold":
            return 0
        if action == "call":
            biggest_bet = max(p.n_bet_chips for p in self.players)
            return biggest_bet - self.current_player.n_bet_chips
        if action == "all_in":
            return self.current_player.n_chips
        if action.startswith("raise:"):
            fraction = self._parse_raise_fraction(action, "chips_to_add")
            return self._compute_raise_chip_amount(fraction, enforce_minimum=True)
        raise ValueError(f"chips_to_add: unknown action {action!r}")

    def string_for_chips(self, chip_amount: int) -> str:
        """Map an observed chip raise to the abstraction's action string.

        ``chip_amount`` is the total chips the actor added to the pot
        at this decision (matching :meth:`_compute_raise_chip_amount`'s
        ``n_chips_to_add`` convention).  The caller is responsible for
        routing fold / call / check separately; this method handles
        raises (including ``all_in``) and so requires
        ``chip_amount > 0``.

        Conversion rule (§6.3 — the chip→string boundary; it does **not**
        approximate off-tree sizes):

        1. ``chip_amount == actor.n_chips`` -> ``"all_in"``.
        2. Exact match against any canonical clamp -> ``"raise:<f>"``.
        3. Else an off-tree ``"raise:<f_obs>"`` string with ``f_obs``
           rounded to 4 decimals (a stable overlay key).

        Off-tree sizes are returned verbatim — they are **not** snapped
        to the nearest abstraction fraction here.  Approximation is the
        job of the explicit translation layer: rounds 2-4 inject the
        off-tree size (:meth:`inject_action`) and re-search, while
        round-1 / blueprint-lookup paths apply pseudo-harmonic
        translation (:meth:`_translate_fraction`).  Snapping here would
        pre-empt and corrupt both.

        The runtime then either feeds the result directly to
        :meth:`step_in_place` (if it is already canonical) or first
        injects it via :meth:`inject_action` (if it is off-tree).
        Whether to inject is a membership check on
        :attr:`legal_actions`, not a property of this return value.

        Off-tree returns are not guaranteed to be *playable*: at
        ``_n_raises >= MAX_RAISES_PER_ROUND`` or otherwise unplayable
        states, :meth:`inject_action` will reject the string and
        return ``False``.  That rejection is the env's signal that
        the abstraction tree cannot represent the observation; the
        runtime should handle it as an integration-layer condition.

        Raises
        ------
        ValueError
            If ``chip_amount <= 0``.  Fold / call / check have their
            own dedicated handling upstream and never reach this
            method.
        """
        if chip_amount <= 0:
            raise ValueError(
                f"string_for_chips: chip_amount must be positive "
                f"(fold/call/check are routed separately), got {chip_amount}"
            )
        if chip_amount == self.current_player.n_chips:
            return "all_in"
        for f in self.canonical_raise_fractions():
            if self._compute_raise_chip_amount(f, enforce_minimum=True) == chip_amount:
                return f"raise:{f}"
        f_obs = chip_amount / self.pot_size
        return f"raise:{round(f_obs, 4)}"

    # ------------------------------------------------------------------
    # Off-tree action overlay
    # ------------------------------------------------------------------

    def _current_public_state(self) -> Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        """Identifier for the current public game-tree node.

        Pure function of state visible to every seat: the betting stage
        and the **full cross-street** action history (every stage's
        actions, in play order).  This is the card-free projection of
        :meth:`_compute_info_set` — the same history, minus the actor's
        card cluster — so two seats at the same public node share the key,
        while public nodes that differ only in *earlier* streets' betting
        (hence pot size, stack depth, and the raise sizes that are legal
        there) stay distinct.

        The history **must** span all streets, not just the current one:
        the betting line alone determines pot and stacks, so truncating to
        the current stage would collapse e.g. every river node reached via
        a different flop/turn line onto ``("river", ())`` — distinct
        decisions (often with distinct legal raise sets) sharing one key.

        Reads ``_history`` via ``items()`` — a pure read that never inserts
        an empty stage (a bare ``self._history[stage]`` on the defaultdict
        would, and that empty entry would then leak into
        :meth:`_compute_info_set`, which iterates ``_history.items()``).
        The current stage is carried explicitly as the first element, so a
        node at the *start* of a street (that street absent from
        ``_history`` until its first action) is still distinguished from
        the end of the previous one.
        """
        cache = self._public_state_cache
        if cache is not None and cache[0] == self._state_version:
            return cache[1]
        result = (
            self._betting_stage,
            tuple(
                (stage, tuple(actions))
                for stage, actions in self._history.items()
            ),
        )
        self._public_state_cache = (self._state_version, result)
        return result

    def inject_action(self, action: str) -> bool:
        """Inject `action` into the legal set at the current public state.

        Idempotent.  Recorded in a dict shared by every env in this
        deepcopy lineage, so subgame searches rooted at any descendant
        env see the augmented game tree at the matching public state.

        Sanity-checked: a raise that would fall below the minimum raise
        increment, exceed the current player's stack (where canonical
        code would substitute ``all_in``), or arrive while
        ``_n_raises >= MAX_RAISES_PER_ROUND`` / at a non-betting stage
        is rejected.  Rejection is communicated via the return value
        plus a warning log; the overlay is not mutated.

        Parameters
        ----------
        action : str
            Action string accepted by :meth:`step_in_place`.  Only
            ``"raise:<fraction>"`` is a meaningful injection — canonical
            ``"fold"`` / ``"call"`` / ``"all_in"`` are always already
            legal and `inject_action` is a no-op returning ``True`` for
            them.

        Returns
        -------
        bool
            ``True`` iff ``action`` is in
            ``[a for a in legal_actions if a is not None]`` after the
            call.  ``False`` iff the injection was rejected because the
            action is not legally playable at the current state.

        Raises
        ------
        ValueError
            If ``action`` is malformed (unknown prefix, unparseable
            fraction, or non-positive fraction).  These indicate caller
            bugs, not game-state conditions.
        """
        if action in ("fold", "call", "all_in"):
            # Already canonical; nothing to record.  Return value
            # honours the documented contract — True iff the action
            # is actually legal at this state (inactive players and
            # zero-stack actors can render canonical actions illegal).
            return action in self.legal_actions
        if not action.startswith("raise:"):
            raise ValueError(
                f"inject_action: only 'fold' / 'call' / 'all_in' / "
                f"'raise:<fraction>' are supported, got {action!r}"
            )
        fraction = self._parse_raise_fraction(action, "inject_action")
        if not self._raise_fraction_is_playable(fraction):
            logger.warning(
                "inject_action rejected %r at public state %r — "
                "not playable (stage %s, n_raises=%d).",
                action, self._current_public_state(),
                self._betting_stage, self._n_raises,
            )
            return False
        key = self._current_public_state()
        existing = self._extra_legal_actions.get(key, frozenset())
        if action not in existing:
            self._extra_legal_actions[key] = existing | {action}
            # The shared overlay just widened the legal set at this public state.
            # Bump the lineage-shared version so every env's cache (this one and
            # any sibling/parent that already memoised this node) re-derives
            # against the new overlay on its next read.
            self._overlay_version[0] += 1
        return True

    def _raise_fraction_is_playable(self, fraction: float) -> bool:
        """Mirror of the canonical raise-validity checks in
        :meth:`_get_available_raise_sizes`.  Used by :meth:`inject_action`
        so injected raise sizes obey the same minimum-raise / max-stack
        / stage / raise-count rules as canonical raises.
        """
        if self._betting_stage in {"terminal", "show_down"}:
            return False
        if not self.current_player.is_active:
            return False
        if self._n_raises >= MAX_RAISES_PER_ROUND:
            return False
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        chips_available = self.current_player.n_chips
        chips_raw = self._compute_raise_chip_amount(
            fraction, enforce_minimum=False
        )
        actual_raise = chips_raw - n_chips_to_call
        if actual_raise < self._last_raise_amount:
            return False
        chips = self._compute_raise_chip_amount(
            fraction, enforce_minimum=True
        )
        if chips > chips_available or chips >= chips_available - 1:
            return False
        return True

    def reset_overlay(self) -> None:
        """Clear all injected actions across every public state.

        Called at the start of every new hand so off-tree actions
        recorded during one hand do not leak into the next hand's
        tree at matching public states.
        """
        self._extra_legal_actions.clear()
        # Clearing the shared overlay can narrow the legal set at any injected
        # state.  Bump the lineage-shared version so every env's memo (this one
        # and every sibling/parent) re-derives against the now-empty overlay —
        # a bare ``self._legal_actions_cache.clear()`` would leave sibling caches
        # still offering the purged actions.
        self._overlay_version[0] += 1

    @property
    def has_overlay_at_current_node(self) -> bool:
        """True iff ``legal_actions`` includes any injected (non-canonical) action.

        Returns ``False`` when ``current_player.is_active`` is False —
        in that case ``legal_actions`` short-circuits to ``[None]``
        and exposes no overlay, so the two properties stay in sync.
        """
        if not self.current_player.is_active:
            return False
        return bool(self._extra_legal_actions.get(self._current_public_state()))

    def with_hole_cards(
        self, holes: "Sequence[Tuple[int, int]]"
    ) -> "PokerEnv":
        """Return a deepcopy with every seat's hole cards replaced.

        Batched atomic replacement: the caller supplies one
        ``(c0, c1)`` tuple per seat.  Seats whose new hole equals
        their current hole are no-ops on the deck; the rest swap as
        a multiset via a single :meth:`Deck.replace_drawn` call,
        followed by :meth:`Deck.shuffle_undealt` so the next
        community deal samples uniformly over the undealt set.

        Used by the leaf evaluator's hole-resampling step: live and
        folded opponents are sampled from their respective ranges
        and the bot's seat is filled with ``ctx.my_hole``, then this
        single call applies them all.  The returned env is
        indistinguishable from a regular state that was dealt these
        cards from the start: future community deals via
        :meth:`step_in_place` will not collide with the new holes.

        Validation (all raise :class:`ValueError`):

        - ``len(holes) == self.n_players``.
        - Each ``holes[i]`` has ``c0 != c1`` — no duplicate within a hole.
        - No card appears in more than one hole — pairwise disjoint
          across seats.
        - No card overlaps ``self.community_cards``.

        There is no per-seat "collides with another seat's current
        hole" check: because the caller specifies every seat's new
        hole at once, no seat retains any "old" card the batch
        needs to honour.  Two seats may swap holes
        (``new[A] == old[B]`` and ``new[B] == old[A]``) and the
        deck still ends up consistent.

        Parameters
        ----------
        holes : Sequence[tuple[int, int]]
            One ``(c0, c1)`` tuple per seat, indexed by seat position
            (``holes[i]`` is seat ``i``'s new hole).  Length must
            equal ``self.n_players``.

        Returns
        -------
        PokerEnv
            Independent copy of the env with every seat's cards
            replaced and the deck synced and reshuffled.

        Raises
        ------
        ValueError
            If ``holes`` is the wrong length or any card constraint
            is violated.
        """
        n = self.n_players
        if len(holes) != n:
            raise ValueError(
                f"with_hole_cards: expected {n} hole tuples (one per "
                f"seat), got {len(holes)}."
            )
        # Validate per-hole, accumulate the new card set as we go.
        new_cards: List[int] = []
        for seat, h in enumerate(holes):
            c0, c1 = int(h[0]), int(h[1])
            if c0 == c1:
                raise ValueError(
                    f"with_hole_cards: seat {seat}'s hole must contain "
                    f"distinct cards, got ({c0}, {c1})."
                )
            new_cards.append(c0)
            new_cards.append(c1)
        # Pairwise disjoint across seats: every card appears at most once.
        if len(set(new_cards)) != len(new_cards):
            raise ValueError(
                f"with_hole_cards: cards must be pairwise disjoint "
                f"across seats; got duplicates in {new_cards}."
            )
        # No overlap with the community.
        community = set(int(c) for c in self.community_cards)
        overlap = community & set(new_cards)
        if overlap:
            raise ValueError(
                f"with_hole_cards: cards {sorted(overlap)} overlap the "
                f"community."
            )
        new = copy.deepcopy(self)
        old_union: List[int] = []
        for seat in range(n):
            old_union.extend(int(c) for c in new.players[seat]._cards)
        new_union = tuple(new_cards)
        for seat in range(n):
            h = holes[seat]
            new.players[seat]._cards = (int(h[0]), int(h[1]))
        new.deck.replace_drawn(tuple(old_union), new_union)
        new.deck.shuffle_undealt()
        return new

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Legal actions for the current player.

        Includes any actions injected at the current public state via
        :meth:`inject_action` (off-tree size handling for subgame search).
        Injected actions are appended after the canonical set in a
        deterministic order (lexicographic) and deduped.

        **Ordering contract.**  Canonical actions appear first in the
        canonical order defined by
        :meth:`get_canonical_actions` — this is the order the blueprint
        regret tables are indexed by via
        :data:`environment.action_space.ACTION_TO_IDX`.  Overlay
        actions follow in :func:`sorted` order so the returned list is
        reproducible across processes and Python hash-randomisation
        seeds — important for subgame solver tables that build a
        per-node ``a_to_i`` mapping from ``env.legal_actions`` and
        rely on that mapping being stable across iterations and runs.

        **Caching.**  The legal set is a pure function of the current public
        state (plus the overlay, which is itself public-state-keyed), but
        deriving it — notably the raise-size enumeration in
        :meth:`_get_available_raise_sizes` — is the single hottest per-advance
        cost in the search inner loop, where the same node is re-validated and
        re-visited many times.  The result is therefore memoised in
        ``_legal_actions_cache`` keyed by :meth:`_current_public_state`.  The key
        *is* the validation: a lookup returns the value for exactly the current
        state, so ``step_in_place`` / ``undo`` need no cache bookkeeping (the
        public state they restore re-selects the right entry).  The overlay
        mutators (:meth:`inject_action` / :meth:`reset_overlay`) bump the
        lineage-shared :attr:`_overlay_version`; each entry is stamped with the
        version it was built at, and a mismatch is treated as a miss — so an
        inject/reset on **any** env in the deepcopy lineage invalidates this
        (and every sibling's) memo, matching the overlay's shared-by-reference
        semantics.  A defensive copy is returned so callers keep the historical
        "fresh list each call" contract and can never corrupt the memo.
        """
        key = self._current_public_state()
        version = self._overlay_version[0]
        cached = self._legal_actions_cache.get(key)
        if cached is None or cached[0] != version:
            actions = self._compute_legal_actions(key)
            self._legal_actions_cache[key] = (version, actions)
            return list(actions)
        return list(cached[1])

    def _compute_legal_actions(self, public_state: Tuple) -> List[Optional[str]]:
        """Derive the legal-action list for ``public_state`` (the current state)."""
        if not self.current_player.is_active:
            return [None]
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        chips_available = self.current_player.n_chips
        actions: List[Optional[str]] = ["fold"]
        if n_chips_to_call >= chips_available:
            if chips_available > 0:
                actions.append("all_in")
        else:
            actions.append("call")
            # Raises (and shove-as-raise) are only meaningful when at least one
            # *other* live player could call them.  Facing a lone all-in — the
            # current player is the only one with chips — the sole legal
            # responses are call or fold (a raise/over-shove would just be
            # returned uncalled).  ``n_players_with_moves`` counts the current
            # live player, so ``>= 2`` means another live player remains.
            if (self._n_raises < MAX_RAISES_PER_ROUND
                    and dynamics.n_players_with_moves(self) >= 2):
                actions += self._get_available_raise_sizes()
        overlay = self._extra_legal_actions.get(public_state)
        if overlay:
            seen = {a for a in actions if a is not None}
            # `sorted` is critical: `overlay` is a frozenset whose
            # iteration order depends on hash randomisation and is not
            # stable across processes.  Sorting makes legal_actions
            # reproducible run-to-run.
            actions += sorted(a for a in overlay if a not in seen)
        return actions

    def _compute_info_set(self, cards: Sequence[int]) -> bytes:
        """Build the compact info-set key for the current actor under ``cards``.

        Factored out of :attr:`info_set` so :meth:`policy_state_for`
        can compute the info-set under a *hypothetical* hole without
        reading any seat's actual cards — the leak-free path used by
        the ``sigma_for_combo`` driver.

        Parameters
        ----------
        cards : Sequence[int]
            The hole cards to assume for the current actor.  Length
            must match the env's per-seat card count (typically 2).

        Returns
        -------
        bytes
            Compact info-set key (see :func:`encode_info_set`), identical
            in format to :attr:`info_set`.

        Raises
        ------
        ValueError
            If the combined cards are missing from ``card_info_lut``
            outside of terminal / show-down stages.
        """
        lookup_cards = tuple(sorted(cards) + sorted(self.community_cards))
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self._betting_stage not in {"terminal", "show_down"}:
                raise ValueError("Cards missing from LUT — load it correctly.")
            return _INFO_SET_DEFAULT
        return encode_info_set(cards_cluster, self._history.items())

    def _canonicalize_history(
        self, history: "Mapping[str, Sequence[str]]"
    ) -> List[Tuple[str, List[str]]]:
        """History with off-tree raise sizes snapped to the blueprint grid.

        Walks each stage's action list in order, tracking a per-stage
        raise index (0 for the first raise/all-in, 1+ thereafter) so the
        ``first_raise`` vs ``subsequent_raise`` grid matches what the env
        used when the action was played.  ``fold`` / ``call`` / ``skip``
        are copied verbatim and leave the index unchanged; ``all_in`` is
        copied verbatim and advances the index; an on-tree ``raise:<f>``
        is copied verbatim while an off-tree one is replaced by its
        deterministic pseudo-harmonic neighbour, both advancing the index.

        Strict no-op when every raise is already on-tree (the universal
        offline case): the returned ``(stage, actions)`` pairs preserve
        ``history`` order and content, so :meth:`_blueprint_info_set`
        produces a key byte-identical to :meth:`_compute_info_set`.
        """
        out: List[Tuple[str, List[str]]] = []
        for stage, actions in history.items():
            grid = self._abstraction_fractions(stage, 0)
            sub_grid = self._abstraction_fractions(stage, 1)
            raise_index = 0
            rewritten: List[str] = []
            for token in actions:
                if isinstance(token, str) and token.startswith("raise:"):
                    cell = grid if raise_index == 0 else sub_grid
                    canonical_strs = {f"raise:{g}" for g in cell}
                    if token in canonical_strs:
                        rewritten.append(token)
                    else:
                        f = float(token.split(":", 1)[1])
                        f_canon = self._translate_fraction(
                            f, stage, raise_index, randomized=False
                        )
                        rewritten.append(f"raise:{f_canon}")
                    raise_index += 1
                elif token == "all_in":
                    rewritten.append(token)
                    raise_index += 1
                else:  # fold / call / skip
                    rewritten.append(token)
            out.append((stage, rewritten))
        return out

    def _blueprint_info_set(self, cards: Sequence[int]) -> bytes:
        """Like :meth:`_compute_info_set`, but with the action history
        canonicalised (off-tree raises snapped to the nearest on-tree
        node via deterministic pseudo-harmonic translation) so a
        blueprint table lookup resolves instead of missing the table.

        A strict no-op versus :meth:`_compute_info_set` whenever the
        history contains no off-tree raises (see
        :meth:`_canonicalize_history`) — the canonicalised tokens are
        byte-identical to the on-tree ones and hit the same alphabet
        bytes, so the two keys are equal.
        """
        lookup_cards = tuple(sorted(cards) + sorted(self.community_cards))
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self._betting_stage not in {"terminal", "show_down"}:
                raise ValueError("Cards missing from LUT — load it correctly.")
            return _INFO_SET_DEFAULT
        return encode_info_set(
            cards_cluster, self._canonicalize_history(self._history)
        )

    def info_set_fields(
        self, cards: Optional[Sequence[int]] = None
    ) -> Optional[Tuple[int, List[Tuple[str, List[str]]]]]:
        """Structured view of the info-set: ``(cluster, [(stage, actions), ...])``.

        The pre-encoding fields behind :attr:`info_set` / :meth:`_compute_info_set`,
        exposed for human-readable dumps and tests that previously parsed the
        JSON key (the key itself is now opaque compact :func:`encode_info_set`
        ``bytes``).  ``cards`` defaults to the current actor's hole.  Returns
        ``None`` at a terminal / show-down node whose hole+board is missing
        from the LUT (the case :meth:`_compute_info_set` maps to
        :data:`_INFO_SET_DEFAULT`).
        """
        if cards is None:
            cards = self.current_player._cards
        lookup_cards = tuple(sorted(cards) + sorted(self.community_cards))
        try:
            cluster = int(self.card_info_lut[self._betting_stage][lookup_cards])
        except KeyError:
            if self._betting_stage not in {"terminal", "show_down"}:
                raise ValueError("Cards missing from LUT — load it correctly.")
            return None
        history = [(stage, list(actions)) for stage, actions in self._history.items()]
        return cluster, history

    @property
    def info_set(self) -> bytes:
        """Compact binary information-set key for the current player.

        The information set captures everything the current player
        knows: the card cluster for their hole cards and the community
        cards, plus the full action history for all betting stages.
        Used as the (opaque, hashed) key into the CFR strategy tables.
        See :func:`encode_info_set`; use :meth:`info_set_fields` for a
        human-readable structured view.

        Returns
        -------
        bytes
            Compact key encoding ``(cards_cluster, history)``.

        Raises
        ------
        ValueError
            If the current cards are not found in ``card_info_lut`` and
            the game is not in a terminal or show-down stage.
        """
        return self._compute_info_set(self.current_player._cards)

    @property
    def policy_state(self) -> PolicyState:
        """Decoupled view of fields a :class:`Policy.strategy` needs.

        Bundles ``betting_round``, ``info_set``, ``get_valid_mask()``
        and the filtered ``legal_actions`` into a frozen
        :class:`PolicyState` so policies don't have to hold a live
        env reference.  Built fresh on every read — cheap.
        """
        legal = tuple(a for a in self.legal_actions if a is not None)
        mask = self.get_valid_mask()
        mask.setflags(write=False)
        return PolicyState(
            player_i=self.player_i,
            betting_round=self.betting_round,
            info_set=self.info_set,
            valid_mask=mask,
            legal_actions=legal,
        )

    def policy_public_fields(self) -> PublicPolicyFields:
        """The combo-independent fields of a :class:`PolicyState` at the current state.

        ``legal_actions`` and ``valid_mask`` (plus ``player_i`` / ``betting_round``)
        read only public state, so a caller sweeping many combos at the *same* env
        state computes them once here and threads them into
        :meth:`policy_state_for` via ``public=`` — only the per-combo ``info_set`` is
        then rebuilt per combo.  The returned ``valid_mask`` is immutable.
        """
        legal = tuple(a for a in self.legal_actions if a is not None)
        mask = self.get_valid_mask()
        mask.setflags(write=False)
        return PublicPolicyFields(
            player_i=self.player_i,
            betting_round=self.betting_round,
            legal_actions=legal,
            valid_mask=mask,
        )

    def policy_state_for(
        self,
        combo: Sequence[int],
        *,
        for_blueprint: bool = False,
        public: "Optional[PublicPolicyFields]" = None,
    ) -> PolicyState:
        """:class:`PolicyState` for the current actor under hypothetical hole ``combo``.

        Reads only public state (community cards, history,
        ``card_info_lut``) plus the supplied ``combo``.  Does **not**
        read any seat's ``_cards`` — including the current actor's —
        so the result is identical regardless of opponents' actual
        holes.  This is the leak-free path a ``sigma_for_combo``
        driver uses: ``policy.strategy(env.policy_state_for(combo), bias)``.

        Parameters
        ----------
        combo : Sequence[int]
            Hypothetical hole for the current actor; typically a
            length-2 tuple drawn from ``env.combo_cards``.
        for_blueprint : bool
            When ``True`` the ``info_set`` is built with the action
            history canonicalised (off-tree raises snapped to the
            nearest on-tree node, §6.3) so a :class:`BlueprintPolicy`
            lookup resolves to a populated regret row instead of the
            uniform fallback.  A no-op on fully on-tree histories.
        public : PublicPolicyFields, optional
            Precomputed combo-independent fields from
            :meth:`policy_public_fields` (the same env state).  When
            given, the public part is reused instead of recomputed —
            so a per-combo sweep skips re-deriving ``legal_actions`` /
            ``valid_mask`` each combo.  When ``None`` it is computed
            here, so the result is identical either way.
        """
        if public is None:
            public = self.policy_public_fields()
        info_set = (
            self._blueprint_info_set(combo)
            if for_blueprint
            else self._compute_info_set(combo)
        )
        return PolicyState(
            player_i=public.player_i,
            betting_round=public.betting_round,
            info_set=info_set,
            valid_mask=public.valid_mask,
            legal_actions=public.legal_actions,
        )

    @property
    def is_terminal(self) -> bool:
        """True when the hand has ended."""
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def current_player(self) -> Player:
        """The player whose turn it is to act."""
        return self.players[self.player_i]

    @property
    def player_i(self) -> int:
        """Index of the current player in ``self.players``."""
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @player_i.setter
    def player_i(self, _: Any) -> None:
        """Raise an error; ``player_i`` is read-only."""
        raise ValueError("player_i is read-only.")

    @property
    def n_players(self) -> int:
        """Total number of players."""
        return len(self.players)

    @property
    def betting_stage(self) -> str:
        """Current betting stage string."""
        return self._betting_stage

    @property
    def betting_round(self) -> int:
        """Current betting stage as an integer index.

        Returns
        -------
        int
            0=pre_flop, 1=flop, 2=turn, 3=river, 4=show_down.

        Raises
        ------
        ValueError
            If the current betting stage does not have a numeric mapping
            (e.g. ``"terminal"``).
        """
        try:
            return self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Unsupported betting stage '{self._betting_stage}' for betting_round."
            )

    @property
    def pot_size(self) -> int:
        """Current total chips in the pot."""
        return self.pot.total

    @property
    def min_raise_amount(self) -> int:
        """Minimum legal raise increment."""
        return self._last_raise_amount

    @property
    def payout(self) -> Dict[int, int]:
        """Chip delta per player index relative to *that seat's* starting stack.

        Nets each seat's current chips against its **own** starting stack
        (``_initial_chips_by_seat``), so the mapping is always zero-sum — equal
        to ``won[i] - contributed[i]`` at a terminal.  For the equal-stack hands
        every current caller deals (``new_game``) this is identical to netting
        against seat 0's stack; it differs (and only this is correct) once seats
        start a hand with unequal stacks.

        Returns
        -------
        dict[int, int]
            Mapping of player index → chips gained (positive) or lost
            (negative) compared to that seat's starting stack.
        """
        return {
            i: player.n_chips - self._initial_chips_by_seat[i]
            for i, player in enumerate(self.players)
        }

    @property
    def terminal_contributions(self) -> Optional[Tuple[int, ...]]:
        """Per-seat pot contributions captured at the terminal, before reset.

        ``None`` until the hand ends; otherwise the tuple of chips each seat put
        in, snapshotted by ``compute_winners`` **before** it resets the pot (so
        the matched/contested stake is recoverable even though ``payout`` has
        already netted the winnings).  The authoritative stake source for the
        :meth:`vector_payout` evaluator — in heads-up the **smaller** of the two
        contributions is the winner-takes amount, the larger stack's excess being
        the uncalled difference.
        """
        return self._terminal_contributions

    @property
    def terminal_board_len(self) -> Optional[int]:
        """Community cards on the board when betting ended (``None`` until then).

        The engine force-deals the board out to five at every terminal, so
        ``len(community_cards)`` cannot tell a turn-side fold from a river-side
        one.  This is the count captured *before* that force-deal — the board the
        hand actually reached.  Used by :meth:`vector_payout` so a pre-river fold
        does card removal against the board it saw, not the dealt-out completion.
        """
        return self._terminal_board_len

    @property
    def runout_key(self) -> Optional[Tuple]:
        """Hashable identity of a decision-free runout (``None`` if not one).

        The pre-runout snapshot ``(board_prefix, pot_contributions, active_mask)``
        recorded at an incomplete-board all-in.  Because the holes are fixed at a
        terminal, this triple fully determines :meth:`runout_equity`'s value, so
        callers may memoise that integration on it without reaching into the env's
        internals.
        """
        return self._runout_info

    @property
    def is_decision_free(self) -> bool:
        """True iff the hand resolved as an all-in showdown over an incomplete
        board — a *decision-free* runout whose value depends only on the
        remaining community cards (§6.4).

        Equivalent to "a :meth:`runout_equity` is applicable and differs from
        the single sampled :attr:`payout`".  Set when the env force-resolves to
        showdown with the board not yet complete and at least two players still
        active; ``False`` for fold-terminals, complete-board (river) showdowns,
        and every non-terminal state.  The bot/solver use it to decide whether
        to replace the env's single sampled runout with the exact board-average.
        """
        return self._runout_info is not None

    def runout_equity(
        self, *, rng: Optional["np.random.Generator"] = None, cap: int = 5000
    ) -> Dict[int, float]:
        """Exact expected per-seat chip delta over every board completion of a
        decision-free all-in runout (§6.4).

        Replaces the env's single sampled runout (one random board scored by
        ``compute_winners``) with the mean over **all** completions of the
        committed board prefix — the value the depth-limited solver and the
        leaf evaluator want for an all-in showdown.  Reuses the recorded
        pre-runout snapshot (``_runout_info``): the board prefix, the frozen
        per-player pot contributions (side-pot structure is board-independent
        once all bets are in), and the active mask.  Side-pot distribution is
        delegated to :meth:`Pot.compute_utility` — not re-implemented.

        Card removal excludes every dealt hole (active *and* folded seats) and
        the prefix board from the completion deck.  The result has the same
        shape/sign convention as :attr:`payout` but is float-valued.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Source for the sampling fallback (below).  Unused on the exact
            path; defaults to a fresh default generator only if sampling is
            actually needed.
        cap : int
            Maximum number of completions enumerated exactly.  If the number of
            distinct completions exceeds ``cap`` (e.g. a pre-flop all-in with
            five board cards to come), ``cap`` completions are Monte-Carlo
            sampled instead and a warning is logged.  In search the runout is
            <= 2 cards, so the exact path always runs.

        Returns
        -------
        dict[int, float]
            Mapping of player index → expected chip delta vs. their starting
            stack, averaged over the runout.

        Raises
        ------
        ValueError
            If called on a state that is not a decision-free runout
            (:attr:`is_decision_free` is ``False``).
        """
        if self._runout_info is None:
            raise ValueError(
                "runout_equity requires a decision-free all-in runout state "
                "(is_decision_free is False); nothing to integrate."
            )
        prefix, pot_chips, active = self._runout_info
        n = len(self.players)
        active_players = [
            self.players[i] for i in range(n) if active[i]
        ]
        k = 5 - len(prefix)

        # Completion deck: every card not already dealt to a hole or on the
        # prefix board (card removal across all seats, incl. folded).
        used = set(int(c) for c in prefix)
        for p in self.players:
            used.update(int(c) for c in p._cards)
        available = [int(c) for c in self.deck._cards if int(c) not in used]

        if k <= 0:
            # Board already complete — single deterministic showdown.
            completions: "Sequence" = [()]
        else:
            n_combos = _n_choose_k(len(available), k)
            if n_combos <= cap:
                completions = itertools.combinations(available, k)
            else:
                logger.warning(
                    "runout_equity: %d completions exceed cap %d; sampling %d "
                    "boards instead (street prefix=%d).",
                    n_combos, cap, cap, len(prefix),
                )
                gen = rng if rng is not None else np.random.default_rng()
                completions = (
                    tuple(gen.choice(available, size=k, replace=False))
                    for _ in range(cap)
                )

        prefix_list = list(prefix)
        accum = [0.0] * n

        # Materialise the (cap-bounded) completions and rank every
        # (completion x active player) seven-card hand in one vectorised batch,
        # then settle every completion's side pots (delegated to the swappable
        # module-level ``_settle_runout`` so the compiled ``runout`` kernel can
        # replace only the settlement while the ranking stays identical).
        completions = [
            prefix_list + [int(c) for c in comp] for comp in completions
        ]
        count = len(completions)
        if count and active_players:
            boards = np.asarray(completions, dtype=np.int64)  # (count, 5)
            holes = np.asarray(
                [[int(c) for c in p._cards] for p in active_players], dtype=np.int64
            )  # (n_active, 2)
            n_active = len(active_players)
            hands = np.empty((count, n_active, 5 + 2), dtype=np.int64)
            hands[:, :, :5] = boards[:, None, :]
            hands[:, :, 5:] = holes[None, :, :]
            rank_mat = default_evaluator.evaluate_batch(
                hands.reshape(count * n_active, 7)
            ).reshape(count, n_active)
            accum = _settle_runout(
                active_players, self.players, pot_chips, rank_mat, count, n
            )

        if count == 0:
            # No feasible completion (degenerate card exhaustion) — fall back to
            # the contributions, i.e. everyone loses what they put in.
            return {i: float(-pot_chips[i]) for i in range(n)}
        return {i: accum[i] / count - pot_chips[i] for i in range(n)}

    def vector_payout(
        self,
        seat: int,
        opp_seat: int,
        opp_reach: "np.ndarray",
        river: Optional[int] = None,
    ) -> "np.ndarray":
        """Per-combo counterfactual value to ``seat`` vs the opponent's range.

        The **vectorised** terminal payout (heads-up): values an entire range
        against an entire range at a terminal, returning a value per hole combo
        (aligned to :attr:`combo_cards`) for ``seat`` against ``opp_reach`` (the
        ``opp_seat`` reach-weighted range).  The counterpart of :attr:`payout`
        (one dealt hand) and :meth:`runout_equity` (board-average); all three
        rank hands with the one shared evaluator, so they agree to the chip —
        including **dead money**: folded seats' contributions go to the winner
        (split on a chop), exactly as the concrete settlement pays them.

        The env owns the settlement entirely — the caller supplies only the
        search quantities it owns:

        Parameters
        ----------
        seat, opp_seat : int
            The two live (contesting) seats; ``seat`` is the acting/traverser
            seat whose combos index the result.
        opp_reach : numpy.ndarray
            ``(n_combos,)`` reach-weighted range of ``opp_seat`` (board masking
            and card removal are applied here).
        river : int, optional
            The board runout card the search sampled for this iteration (a turn
            subgame's chance outcome).  ``None`` for a river subgame / complete
            board.  The engine's own dealt river is ignored in favour of this.

        Returns
        -------
        numpy.ndarray
            ``(n_combos,)`` float64 value to ``seat``; ``0`` on combos that cannot
            be held given the board.
        """
        if self._terminal_contributions is None:
            raise ValueError("vector_payout is only defined at a terminal node.")
        # Matched stake: the smaller of the two contesting seats' final
        # contributions (winner-takes; the larger stack's excess is uncalled).
        tc = self._terminal_contributions
        stake = float(min(tc[seat], tc[opp_seat]))
        # Dead money: every other seat's contribution.  This settlement is only
        # called with exactly two contesting (non-folded) seats — the vector
        # regime's precondition — so the remaining seats have all folded, their
        # chips carry no side-pot claim, and the winner collects them in full
        # (half each on a chopped pot).
        dead = float(sum(tc) - tc[seat] - tc[opp_seat])
        low, high = self._low_card_rank, self._high_card_rank
        combo_cards = self.combo_cards
        removal = range_showdown.removal_for(low, high)
        community = list(self.community_cards)

        if self.players[seat].is_active and self.players[opp_seat].is_active:
            # Showdown: complete the board to five with the search's sampled river
            # (substituting it for whatever the engine dealt), then settle ranges.
            board = community[:4] + [int(river)] if river is not None else community
            ranks, valid = range_showdown.ranked_board(low, high, board)
            return range_showdown.showdown_cfv(
                ranks, valid, combo_cards, opp_reach, stake, dead=dead, removal=removal
            )

        # Fold: the still-active contesting seat wins; value is rank-independent.
        winner = seat if self.players[seat].is_active else opp_seat
        sign = 1.0 if winner == seat else -1.0
        # Mask the opponent reach by the board the hand actually reached.  The
        # engine force-deals the community out to five on a fold, so
        # ``len(community)`` is always 5 here and cannot distinguish a river-side
        # fold from a turn-side one — ``terminal_board_len`` (captured before the
        # force-deal) can.  Only a genuine river-side fold (real board complete)
        # sees the search's sampled river; a pre-river fold uses the shorter board
        # it saw, so card removal does not include cards that were never dealt.
        real_len = self._terminal_board_len
        if real_len is None:
            real_len = len(community)
        if river is not None and real_len == 5:
            board = community[:4] + [int(river)]
        else:
            board = community[:real_len]
        # A fold needs only board-compatibility (no showdown ranking), so use the
        # rank-free mask — cheaper, and it never ranks a partial pre-river board.
        valid = range_showdown.board_valid_mask(low, high, board)
        opp = np.where(valid, opp_reach, 0.0)
        avail = range_showdown.reach_after_removal(combo_cards, opp, removal)
        # The fold's winner collects the loser's matched stake plus the dead
        # money; the folder forfeits only its own (matched) contribution.
        gain = stake + dead if winner == seat else stake
        return sign * gain * np.where(valid, avail, 0.0)

    @property
    def deck_size(self) -> int:
        """Total cards in the deck (including dealt cards)."""
        return (self._high_card_rank - self._low_card_rank + 1) * 4

    @property
    def n_combos(self) -> int:
        """Number of distinct unordered 2-card hole combos in this deck."""
        return self.combo_cards.shape[0]

    @property
    def combo_cards(self) -> np.ndarray:
        """All hole combos as an ``(n_combos, 2)`` int32 array.

        Rows are ordered by card-int value (``c0 < c1``). Shared across
        env instances with the same deck (cached in
        :func:`environment.utils.enumerate_combos`).
        """
        cards, _ = enumerate_combos(self._low_card_rank, self._high_card_rank)
        return cards

    @property
    def combo_index(self) -> Dict[tuple, int]:
        """Inverse of :attr:`combo_cards`: ``(c0, c1) -> row index``."""
        _, index = enumerate_combos(self._low_card_rank, self._high_card_rank)
        return index

    @property
    def public_key(self) -> Tuple[str, Tuple[str, ...]]:
        """Hashable identifier of the current public state.

        ``(betting_stage, full cross-street action history)`` — the key the
        subgame solver's in-memory tables and the off-tree overlay share.
        Public alias of :meth:`_current_public_state`; two seats at the
        same public node see the same key (it embeds no actor cards), while
        nodes differing in earlier streets' betting (pot / stack depth)
        stay distinct.
        """
        return self._current_public_state()

    @property
    def canonical_public_key(self) -> Tuple[str, Tuple[str, ...]]:
        """:meth:`public_key` with off-tree raise sizes in the history snapped to
        the blueprint grid via deterministic pseudo-harmonic translation (§7).

        Byte-identical to :meth:`public_key` when every raise in the history is
        already on-tree (the common case — :meth:`_canonicalize_history` is a strict
        no-op there).  Lets a caller that chose *not* to inject a near-canonical
        off-tree raise resolve the node in the canonical-abstraction subgame the
        solver actually built, instead of missing the tree entirely.
        """
        return (
            self._betting_stage,
            tuple(
                (stage, tuple(actions))
                for stage, actions in self._canonicalize_history(self._history)
            ),
        )

    @property
    def n_raises_this_round(self) -> int:
        """Number of raises made so far in the current betting round.

        Reset to zero at every round boundary; used by the search's
        depth-limit descriptor for the after-2nd-raise cutoff (§3).
        """
        return self._n_raises

    def cluster_for(self, combo: Tuple[int, int]) -> int:
        """LUT cluster id for ``combo`` on the current street and board.

        Returns exactly the cluster :meth:`_compute_info_set` embeds for
        ``combo`` — read straight from ``card_info_lut`` without building
        the JSON info-set string.  ``combo`` is assumed board-compatible
        (the solver only queries combos that share no card with the
        community); a conflicting combo has no LUT entry and raises
        ``KeyError``.
        """
        lookup_cards = tuple(sorted(combo) + sorted(self.community_cards))
        return int(self.card_info_lut[self._betting_stage][lookup_cards])

    @property
    def low_card_rank(self) -> int:
        """Lowest rank in the deck (2=Two, ..., 14=Ace)."""
        return self._low_card_rank

    @property
    def high_card_rank(self) -> int:
        """Highest rank in the deck (2=Two, ..., 14=Ace)."""
        return self._high_card_rank

    @property
    def all_players_have_actioned(self) -> bool:
        """True once all players who started the round have acted."""
        return self._n_actions >= self._n_players_started_round

    @property
    def n_players_started_round(self) -> int:
        """Number of active players at the start of this betting round."""
        return self._n_players_started_round

    @property
    def private_hands(self) -> Dict[int, tuple]:
        """Hole cards for every player.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Mapping of ``player_i`` → tuple of card integers.
        """
        return {p.player_i: p.cards for p in self.players}

    @property
    def initial_regret(self) -> Dict[str, float]:
        """Default regret dictionary for this state (all zeros).

        Returns
        -------
        dict[str, float]
            Mapping of each legal action to 0.0.
        """
        return {action: 0 for action in self.legal_actions}

    @property
    def initial_strategy(self) -> Dict[str, float]:
        """Default strategy dictionary for this state (all zeros).

        Returns
        -------
        dict[str, float]
            Mapping of each legal action to 0.0.
        """
        return {action: 0 for action in self.legal_actions}

    def __repr__(self) -> str:
        return (
            f"<PokerEnv player_i={self.player_i} "
            f"betting_stage={self._betting_stage} "
            f"deck={self.deck_size}>"
        )

    # ------------------------------------------------------------------
    # Canonical action helpers (used by training)
    # ------------------------------------------------------------------

    @staticmethod
    def get_canonical_actions(betting_round: int) -> List[str]:
        """Return the full abstract action set for ``betting_round`` in stable order.

        Parameters
        ----------
        betting_round : int
            Betting round index: 0=pre_flop, 1=flop, 2=turn, 3=river.

        Returns
        -------
        list[str]
            All possible action strings for this stage, ordered as
            ``["fold", "call", "all_in", "raise:<f1>", ...]``.

        Raises
        ------
        ValueError
            If ``betting_round`` is not in [0, 3].
        """
        stage_names = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}
        stage = stage_names.get(betting_round)
        if stage is None:
            raise ValueError(f"betting_round must be 0-3, got {betting_round}")
        stage_config = RAISE_SIZES_BY_STAGE[stage]
        all_fracs = sorted(
            set(stage_config.get("first_raise", []))
            | set(stage_config.get("subsequent_raise", []))
        )
        return ["fold", "call", "all_in"] + [f"raise:{f}" for f in all_fracs]

    def get_valid_mask(self) -> np.ndarray:
        """Return a boolean mask over the canonical action set for this state.

        Returns
        -------
        numpy.ndarray
            1-D boolean array of length ``len(get_canonical_actions(...))``.
            Entry ``i`` is ``True`` when canonical action ``i`` is legal.
        """
        canonical = PokerEnv.get_canonical_actions(self.betting_round)
        legal_set = {a for a in self.legal_actions if a is not None}
        return np.array([a in legal_set for a in canonical], dtype=bool)



# --------------------------------------------------------------------------- #
# Runout settlement (Phase 2 seam).  Extracted from ``runout_equity`` so the
# compiled ``runout`` kernel can replace the per-completion side-pot scoring while
# the deck/completion/ranking setup stays shared.  ``runout_equity`` calls the
# module global ``_settle_runout`` at run time, so the rebind below is transparent.
#
# Every quantity summed here is INTEGER (chips won per seat, or the fast path's
# ``count * pot_total``), so ``accum`` is an integer-valued float independent of
# summation order — the Cython kernel need only reproduce the same per-completion
# settlement (``Pot.compute_utility``) and the caller does the ``/count`` division.
# --------------------------------------------------------------------------- #
def _settle_runout(active_players, all_players, pot_chips, rank_mat, count, n):
    """Sum ``Pot.compute_utility`` over every board completion → ``accum`` (len n).

    Pure-Python reference (the byte-exact oracle): the vectorised fast path for
    completions whose every side pot has a unique best eligible hand, plus the
    scalar tie-split for the rest.  ``accum[i]`` is chips won by seat ``i`` summed
    over all completions.
    """
    scratch = Pot(n)
    scratch._chips = list(pot_chips)
    accum = [0.0] * n

    def _score_scalar(indices):
        for ci in indices:
            groups = collections.defaultdict(list)
            for a, p in enumerate(active_players):
                groups[int(rank_mat[ci, a])].append(p)
            ranked = [groups[r] for r in sorted(groups)]
            winnings = scratch.compute_utility(all_players, ranked)
            for i in range(n):
                accum[i] += winnings[i]

    specs = []
    degenerate = False
    for sp in scratch.side_pots:
        cols = np.fromiter(
            (a for a, p in enumerate(active_players) if p.player_i in sp),
            dtype=np.intp,
        )
        if cols.size == 0:
            degenerate = True
            break
        glob = np.fromiter(
            (active_players[a].player_i for a in cols),
            dtype=np.intp,
            count=cols.size,
        )
        specs.append((cols, float(sum(sp.values())), glob))

    if degenerate:
        _score_scalar(range(count))
    else:
        clean = np.ones(count, dtype=bool)
        per_pot = []
        for cols, total, glob in specs:
            sub = rank_mat[:, cols]                       # (count, |cols|)
            best = sub.min(axis=1)
            clean &= (sub == best[:, None]).sum(axis=1) == 1
            per_pot.append((glob[sub.argmin(axis=1)], total))
        if clean.any():
            for glob_win, total in per_pot:
                wins = np.bincount(glob_win[clean], minlength=n)
                for i in range(n):
                    accum[i] += float(wins[i]) * total
        _score_scalar(np.flatnonzero(~clean).tolist())
    return accum


_settle_runout_py = _settle_runout

try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    if _CORE_AVAILABLE and _kernel_enabled("runout"):
        from poker_ai._core._runout import settle_runout as _core_settle_runout

        def _settle_runout(active_players, all_players, pot_chips, rank_mat, count, n):
            """Cython-backed runout settlement (flag ``runout``).

            Marshals the plain integer inputs the kernel wants — per-``player_i``
            ``order`` and the active columns' global seats — and returns the summed
            chips-won as floats (integer-valued, matching the Python accumulator).
            """
            order = [0] * n
            for p in all_players:
                order[p.player_i] = p.order
            active_glob = [p.player_i for p in active_players]
            acc = _core_settle_runout(
                rank_mat, list(pot_chips), order, active_glob, n
            )
            return [float(acc[i]) for i in range(n)]
except ImportError:
    pass
