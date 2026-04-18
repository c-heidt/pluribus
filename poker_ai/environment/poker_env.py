"""Poker game environment for CFR/MCCFR training.

The ``PokerEnv`` class is the central state object. It holds all live
game data (players, pot, deck, community cards) and provides the CFR
interface: ``apply_action``, ``legal_actions``, ``info_set``, and
``is_terminal``.

Deterministic game-logic functions are in ``dynamics.py``; stochastic
(dealing) operations are in ``chance.py``.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from poker_ai import utils
from poker_ai.environment import dynamics
from poker_ai.environment.chance import Deck
from poker_ai.environment.player import Player
from poker_ai.environment.pot import Pot
from poker_ai.information_abstraction import InfoSetLut, load_info_set_lut

logger = logging.getLogger("poker_ai.environment.poker_env")

# ---------------------------------------------------------------------------
# Action abstraction configuration
# ---------------------------------------------------------------------------
# Raise sizes as fractions of the current pot, by betting stage.
# Based on the Pluribus blueprint strategy design.
RAISE_SIZES_BY_STAGE: Dict[str, Dict[str, List[float]]] = {
    "pre_flop": {
        "first_raise":      [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        "subsequent_raise": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "flop": {
        "first_raise":      [0.33, 0.5, 0.75, 1.0, 1.5, 2.0],
        "subsequent_raise": [0.5, 0.75, 1.0, 1.5],
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
# Factory
# ---------------------------------------------------------------------------


def new_game(
    n_players: int,
    card_info_lut: InfoSetLut = None,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
    **kwargs,
) -> PokerEnv:
    """Create a new poker game.

    The deck is determined automatically from ``card_info_lut``: the rank
    bounds are read from the pre-flop keys so the game always uses the same
    card set the LUT was built for.  When no LUT is provided (e.g. in tests),
    a full 52-card deck is used.

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut : InfoSetLut, optional
        Pre-loaded card cluster lookup table.  When provided, both the disk
        load and the deck configuration are derived from it automatically.
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
        # Pre-load from disk so deck bounds can be derived before construction.
        lut_path = kwargs.get("lut_path", ".")
        pickle_dir_flag = kwargs.get("pickle_dir", False)
        card_info_lut = load_info_set_lut(lut_path, pickle_dir_flag)

    low_card_rank, high_card_rank = 2, 14  # default: full deck
    if card_info_lut:
        from poker_ai.environment.utils import card_rank_int as _rank
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
        load_card_lut=False,  # already loaded above
        **kwargs,
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

    ``apply_action()`` returns a *new* ``PokerEnv`` rather than
    mutating the current instance, so the caller always has an
    immutable snapshot of the state before the action.

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
        lut_path: str = ".",
        pickle_dir: bool = False,
        load_card_lut: bool = True,
        low_card_rank: int = 2,
        high_card_rank: int = 14,
    ):
        """Initialise the environment and deal the first hand.

        Parameters
        ----------
        players : list[Player]
            Pre-constructed player objects.
        small_blind : int
            Small blind amount.
        big_blind : int
            Big blind amount.
        lut_path : str
            Path to the card information LUT file/directory.
        pickle_dir : bool
            Legacy: whether lut_path is a directory of pickle files.
        load_card_lut : bool
            Whether to load the card clustering LUT from disk.
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
        self._pickle_dir: bool = pickle_dir
        self._initial_n_chips: int = players[0].n_chips
        self.small_blind: int = small_blind
        self.big_blind: int = big_blind
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0, "flop": 1, "turn": 2,
            "river": 3, "show_down": 4,
        }

        # LUT (also excluded from deep-copy)
        if load_card_lut:
            self.card_info_lut = load_info_set_lut(lut_path, pickle_dir)
        else:
            self.card_info_lut = {}

        # Live game state (deep-copied in apply_action)
        self.players: List[Player] = players
        self.pot: Pot = Pot(n_players)
        self.deck: Deck = Deck(low_card_rank, high_card_rank)
        self.community_cards: tuple = ()

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
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop":     player_i_order,
            "turn":     player_i_order,
            "river":    player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }

        # Betting round counters
        self._skip_counter: int = 0
        self._first_move_of_current_round: bool = True
        self._last_raise_amount: int = self.big_blind
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
        history) is deep-copied. ``card_info_lut`` is always set to an
        empty dict on the copy; the caller is responsible for restoring it.

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
        # Immutable config — share references, no copy needed
        for attr in (
            "small_blind", "big_blind", "_low_card_rank", "_high_card_rank",
            "_initial_n_chips", "_pickle_dir",
            "_betting_stage_to_round", "_player_i_lut",
        ):
            object.__setattr__(new, attr, getattr(self, attr))
        # Mutable game state — deep copy
        for attr in (
            "players", "pot", "deck", "community_cards",
            "_history", "_betting_stage",
            "_skip_counter", "_first_move_of_current_round",
            "_last_raise_amount", "_all_players_have_made_action",
            "_n_actions", "_n_raises", "_player_i_index",
            "_n_players_started_round",
        ):
            object.__setattr__(new, attr, copy.deepcopy(getattr(self, attr), memo))
        # LUT always excluded (set by caller)
        new.card_info_lut = {}
        return new

    # ------------------------------------------------------------------
    # Core CFR interface
    # ------------------------------------------------------------------

    def apply_action(self, action_str: Optional[str]) -> PokerEnv:
        """Return a new PokerEnv after applying ``action_str``.

        Parameters
        ----------
        action_str : str or None
            One of ``{"fold", "call", "raise:<fraction>", "all_in"}``
            or ``None`` for inactive players.

        Returns
        -------
        PokerEnv
            New game state after the action.
        """
        original_action = action_str
        if action_str not in self.legal_actions:
            logger.warning(
                "Invalid action '%s'. Legal: %s. Mapping to closest.",
                action_str, self.legal_actions,
            )
            action_str = self._map_to_closest_legal_action(action_str)
            logger.info("Mapped '%s' -> '%s'", original_action, action_str)

        lut = self.card_info_lut
        self.card_info_lut = {}
        new_env = copy.deepcopy(self)
        new_env.card_info_lut = self.card_info_lut = lut
        new_env._first_move_of_current_round = False

        if action_str is None:
            assert (
                not new_env.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            new_env.current_player.call(
                players=new_env.players, pot=new_env.pot
            )
            logger.debug("calling")
        elif action_str == "fold":
            new_env.current_player.fold()
        elif action_str == "all_in":
            n_chips_to_add = new_env.current_player.n_chips
            biggest_bet = max(p.n_bet_chips for p in new_env.players)
            current_bet = new_env.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= new_env._last_raise_amount:
                new_env._last_raise_amount = actual_raise_amount
                new_env._n_raises += 1
            logger.debug("going all-in with %d chips", n_chips_to_add)
            new_env.current_player.raise_to(pot=new_env.pot, n_chips=n_chips_to_add)
        elif action_str.startswith("raise:"):
            pot_fraction = float(action_str.split(":")[1])
            n_chips_to_add = new_env._compute_raise_chip_amount(
                pot_fraction, enforce_minimum=False
            )
            biggest_bet = max(p.n_bet_chips for p in new_env.players)
            current_bet = new_env.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            if actual_raise_amount >= new_env._last_raise_amount:
                new_env._last_raise_amount = actual_raise_amount
            logger.debug("adding %d chips to pot (action: %s)", n_chips_to_add, action_str)
            new_env.current_player.raise_to(pot=new_env.pot, n_chips=n_chips_to_add)
            new_env._n_raises += 1
        else:
            raise ValueError(
                f"Unrecognised action '{action_str}'. "
                "Expected 'fold', 'call', 'raise:<fraction>' or 'all_in'."
            )

        skip_actions = ["skip"] * new_env._skip_counter
        new_env._history[new_env.betting_stage] += skip_actions
        new_env._history[new_env.betting_stage].append(action_str)
        new_env._n_actions += 1
        new_env._skip_counter = 0

        while True:
            new_env._move_to_next_player()
            finished_betting = not dynamics.more_betting_needed(new_env)
            if finished_betting and new_env.all_players_have_actioned:
                new_env._increment_stage()
                new_env._reset_betting_round_state()
                new_env._first_move_of_current_round = True
            if not new_env.current_player.is_active:
                new_env._skip_counter += 1
            elif new_env.current_player.is_active:
                if dynamics.n_players_with_moves(new_env) == 1:
                    new_env._betting_stage = "terminal"
                    cards_needed = 5 - len(new_env.community_cards)
                    if cards_needed > 0:
                        new_env.community_cards += new_env.deck.deal_community(cards_needed)
                if new_env._betting_stage in {"terminal", "show_down"}:
                    dynamics.compute_winners(new_env)
                break

        for player in new_env.players:
            player.is_turn = False
        new_env.current_player.is_turn = True
        return new_env

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
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._last_raise_amount = self.big_blind
        self._player_i_index = 0
        self._n_players_started_round = dynamics.n_active_players(self)
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
    # Properties
    # ------------------------------------------------------------------

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Legal actions for the current player."""
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
            if self._n_raises < MAX_RAISES_PER_ROUND:
                actions += self._get_available_raise_sizes()
        return actions

    @property
    def info_set(self) -> str:
        """JSON-encoded information set string for the current player.

        The information set captures everything the current player
        knows: the card cluster for their hole cards and the community
        cards, plus the full action history for all betting stages.
        Used as the key into the CFR strategy tables.

        Returns
        -------
        str
            JSON string with ``"cards_cluster"`` and ``"history"`` keys.

        Raises
        ------
        ValueError
            If the current cards are not found in ``card_info_lut`` and
            the game is not in a terminal or show-down stage.
        """
        # Cards are ints — sort directly, no .eval_card needed
        lookup_cards = tuple(
            sorted(self.current_player._cards) + sorted(self.community_cards)
        )
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self._betting_stage not in {"terminal", "show_down"}:
                raise ValueError("Cards missing from LUT — load it correctly.")
            return "default info set, please ensure you load it correctly"
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {stage: list(actions)}
                for stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
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
        """Chip delta per player index relative to their starting stack.

        Returns
        -------
        dict[int, int]
            Mapping of player index → chips gained (positive) or lost
            (negative) compared to ``initial_chips``.
        """
        return {
            i: player.n_chips - self._initial_n_chips
            for i, player in enumerate(self.players)
        }

    @property
    def deck_size(self) -> int:
        """Total cards in the deck (including dealt cards)."""
        return (self._high_card_rank - self._low_card_rank + 1) * 4

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

