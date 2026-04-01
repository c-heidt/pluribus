"""Unified poker game state supporting any deck size.

This module provides a single PokerState class and new_game() factory that
work with arbitrary deck sizes specified via low_card_rank/high_card_rank
parameters. All decks use the standard poker hand evaluator.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
import operator
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from poker_ai import utils
from poker_ai.environment.card import Card
from poker_ai.environment.engine import PokerEngine
from poker_ai.environment.evaluation.evaluator import Evaluator
from poker_ai.environment.player import Player
from poker_ai.environment.pot import Pot
from poker_ai.environment.table import PokerTable

logger = logging.getLogger("poker_ai.environment.game_state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]

# Action abstraction configuration: raise sizes as fractions of the pot.
# Based on Pluribus blueprint strategy design.
# Pre-flop: fine-grained abstraction (no real-time search typically used)
# Flop: coarser abstraction
# Turn/River: most coarse (3 sizes for first raise, 2 for subsequent raises)
#
# IMPORTANT NOTES:
# 1. Raise amounts are calculated as: pot_fraction * effective_pot
#    where effective_pot = current_pot (previous round pot + all bets committed so far
#    in this round). The total chips added by the player equals this raise amount,
#    which covers both the implicit call and the raise above it.
#
# 2. Positional asymmetry is INTENDED: In multi-way pots, players acting later
#    see a larger pot (includes earlier actions) and thus have larger raise sizes
#    for the same pot fraction. This is realistic poker behavior where position
#    matters. The CFR algorithm learns position-dependent strategies naturally.
#
# 3. All raise amounts are rounded UP (math.ceil) to ensure pot fractions are
#    not under-bet due to integer rounding.
RAISE_SIZES_BY_STAGE: Dict[str, Dict[str, List[float]]] = {
    "pre_flop": {
        # Fine-grained: many bet sizes for blueprint strategy
        # Includes small probing bets, standard bets, and large overbets
        "first_raise": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        "subsequent_raise": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "flop": {
        # More coarse than pre-flop
        "first_raise": [0.33, 0.5, 0.75, 1.0, 1.5, 2.0],
        "subsequent_raise": [0.5, 0.75, 1.0, 1.5],
    },
    "turn": {
        # At most 3 raise sizes for first raise, 2 for remaining (all_in is always added)
        "first_raise": [0.5, 1.0],
        "subsequent_raise": [1.0],
    },
    "river": {
        # Same as turn: at most 3 raise sizes for first raise, 2 for remaining (all_in is always added)
        "first_raise": [0.5, 1.0],
        "subsequent_raise": [1.0],
    },
}

# Maximum number of raises allowed per betting round
MAX_RAISES_PER_ROUND: int = 3


def new_game(
    n_players: int,
    card_info_lut: InfoSetLookupTable = None,
    low_card_rank: int = 2,
    high_card_rank: int = 14,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
    **kwargs,
) -> PokerState:
    """Create a new poker game with the specified deck configuration.

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut : InfoSetLookupTable, optional
        Card information cluster lookup table. If provided, it will be
        attached without reloading from disk.
    low_card_rank : int
        Lowest rank to include in the deck (2=Two, ..., 14=Ace).
    high_card_rank : int
        Highest rank to include in the deck (2=Two, ..., 14=Ace).
    small_blind : int
        Small blind amount.
    big_blind : int
        Big blind amount.
    initial_chips : int
        Starting chip count per player.

    Returns
    -------
    state : PokerState
        Initial game state.
    """
    pot = Pot()
    players = [
        Player(player_i=player_i, initial_chips=initial_chips, pot=pot)
        for player_i in range(n_players)
    ]
    if card_info_lut is not None:
        state = PokerState(
            players=players,
            load_card_lut=False,
            low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
            small_blind=small_blind,
            big_blind=big_blind,
            **kwargs,
        )
        state.card_info_lut = card_info_lut
    else:
        state = PokerState(
            players=players,
            low_card_rank=low_card_rank,
            high_card_rank=high_card_rank,
            small_blind=small_blind,
            big_blind=big_blind,
            **kwargs,
        )
    return state


class PokerState:
    """Poker game state at some given point in time.

    Supports any deck size via low_card_rank/high_card_rank parameters.
    All deck configurations use the standard poker hand evaluator.

    The class is immutable and new state can be instantiated from once an
    action is applied via the `apply_action` method.
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
        """Initialise state.

        Parameters
        ----------
        players : List[Player]
            List of players in the game.
        small_blind : int
            Small blind amount.
        big_blind : int
            Big blind amount.
        lut_path : str
            Path to card information lookup table.
        pickle_dir : bool
            Whether lut_path is a directory of pickle files (deprecated).
        load_card_lut : bool
            Whether to load card clustering lookup tables.
        low_card_rank : int
            Lowest rank to include in the deck (2=Two, ..., 14=Ace).
        high_card_rank : int
            Highest rank to include in the deck (2=Two, ..., 14=Ace).
        """
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        if low_card_rank < 2 or high_card_rank > 14:
            raise ValueError(
                f"Card ranks must be in range [2, 14], got "
                f"low_card_rank={low_card_rank}, high_card_rank={high_card_rank}."
            )
        if low_card_rank > high_card_rank:
            raise ValueError(
                f"low_card_rank ({low_card_rank}) must be <= "
                f"high_card_rank ({high_card_rank})."
            )
        n_ranks = high_card_rank - low_card_rank + 1
        n_cards = n_ranks * 4
        # Need at least 2 hole cards per player + 5 community cards
        min_cards = n_players * 2 + 5
        if n_cards < min_cards:
            raise ValueError(
                f"Deck has {n_cards} cards but need at least {min_cards} "
                f"({n_players} players * 2 + 5 community cards). "
                f"Use fewer players or a larger deck."
            )

        self._low_card_rank = low_card_rank
        self._high_card_rank = high_card_rank
        self._pickle_dir = pickle_dir
        if load_card_lut:
            self.card_info_lut = self.load_card_lut(lut_path, self._pickle_dir)
        else:
            self.card_info_lut = {}
        # Get deck ranks from parameters.
        deck_ranks = self._get_deck_ranks()
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=deck_ranks
        )
        # Get a reference of the initial number of chips for the payout.
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        # Always use standard evaluator.
        evaluator = Evaluator()
        self._poker_engine = PokerEngine(
            table=self._table,
            small_blind=small_blind,
            big_blind=big_blind,
            evaluator=evaluator,
        )
        # Reset the pot, assign betting order to players, assign blinds.
        self._poker_engine.round_setup()
        # Deal private cards to players.
        self._table.dealer.deal_private_cards(self._table.players)
        # Store the actions as they come in here.
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._betting_stage = "pre_flop"
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "show_down": 4,
        }
        # Rotate the big and small blind to the final positions for the pre
        # flop round only.
        player_i_order: List[int] = [p_i for p_i in range(n_players)]
        self.players[0].is_small_blind = True
        self.players[1].is_big_blind = True
        self.players[-1].is_dealer = True
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": player_i_order,
            "turn": player_i_order,
            "river": player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }
        self._skip_counter = 0
        self._first_move_of_current_round = True
        self._last_raise_amount = self.big_blind  # Track last raise size for min raise
        self._reset_betting_round_state()
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

    def _get_deck_ranks(self) -> List[int]:
        """Return the list of card ranks for this deck configuration."""
        return list(range(self._low_card_rank, self._high_card_rank + 1))

    @property
    def deck_size(self) -> int:
        """Return the total number of cards in the deck."""
        return (self._high_card_rank - self._low_card_rank + 1) * 4

    @property
    def low_card_rank(self) -> int:
        """Return the lowest card rank in the deck."""
        return self._low_card_rank

    @property
    def high_card_rank(self) -> int:
        """Return the highest card rank in the deck."""
        return self._high_card_rank

    def __repr__(self):
        """Return a helpful description of object in strings and debugger."""
        return (
            f"<PokerState player_i={self.player_i} "
            f"betting_stage={self._betting_stage} "
            f"deck={self.deck_size}>"
        )

    def _map_to_closest_legal_action(self, invalid_action: str) -> str:
        """Map an invalid action to the closest legal action."""
        legal = self.legal_actions

        if invalid_action == "all_in" and "all_in" in legal:
            return "all_in"

        if invalid_action and invalid_action.startswith("raise:"):
            try:
                target_fraction = float(invalid_action.split(":")[1])
                legal_raises = [a for a in legal if a and a.startswith("raise:")]

                if legal_raises:
                    def get_fraction(action_str):
                        parts = action_str.split(":")
                        return float(parts[1]) if len(parts) > 1 else 0.0

                    closest = min(legal_raises, key=lambda a: abs(get_fraction(a) - target_fraction))
                    return closest
                elif "all_in" in legal:
                    return "all_in"
            except (ValueError, IndexError):
                pass

        if "all_in" in legal:
            return "all_in"
        elif "call" in legal:
            return "call"
        elif "fold" in legal:
            return "fold"
        elif legal and legal[0] is not None:
            return legal[0]

        return "fold"

    def apply_action(self, action_str: Optional[str]) -> PokerState:
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            The description of the action the current player is making. Can be
            any of {"fold", "call", "raise:<fraction>", "all_in"}.

        Returns
        -------
        new_state : PokerState
            A poker state instance that represents the game in the next
            timestep, after the action has been applied.
        """
        original_action = action_str
        if action_str not in self.legal_actions:
            logger.warning(
                f"Invalid action '{action_str}' attempted. "
                f"Legal actions: {self.legal_actions}. "
                f"Mapping to closest legal action."
            )
            action_str = self._map_to_closest_legal_action(action_str)
            logger.info(f"Mapped '{original_action}' -> '{action_str}'")

        lut = self.card_info_lut
        self.card_info_lut = {}
        new_state = copy.deepcopy(self)
        new_state.card_info_lut = self.card_info_lut = lut
        new_state._first_move_of_current_round = False
        if action_str is None:
            assert (
                not new_state.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            action = new_state.current_player.call(players=new_state.players)
            logger.debug("calling")
        elif action_str == "fold":
            action = new_state.current_player.fold()
        elif action_str == "all_in":
            n_chips_to_add = new_state.current_player.n_chips

            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            current_bet = new_state.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call

            if actual_raise_amount >= new_state._last_raise_amount:
                new_state._last_raise_amount = actual_raise_amount
                new_state._n_raises += 1

            logger.debug(f"going all-in with {n_chips_to_add} chips")
            action = new_state.current_player.raise_to(n_chips=n_chips_to_add)
        elif action_str.startswith("raise:"):
            raise_type = action_str.split(":")[1]
            pot_fraction = float(raise_type)
            n_chips_to_add = new_state._compute_raise_chip_amount(
                pot_fraction, enforce_minimum=False
            )

            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            current_bet = new_state.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call

            if actual_raise_amount >= new_state._last_raise_amount:
                new_state._last_raise_amount = actual_raise_amount

            logger.debug(f"adding {n_chips_to_add} chips to pot (action: {action_str})")
            action = new_state.current_player.raise_to(n_chips=n_chips_to_add)
            new_state._n_raises += 1
        else:
            raise ValueError(
                f"Unrecognized action '{action_str}'. Expected 'fold', 'call', "
                f"'raise:<fraction>' or 'all_in'."
            )
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        new_state._history[new_state.betting_stage].append(action_str)
        new_state._n_actions += 1
        new_state._skip_counter = 0
        while True:
            new_state._move_to_next_player()
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                new_state._first_move_of_current_round = True
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
                assert not new_state.current_player.is_active
            elif new_state.current_player.is_active:
                if new_state._poker_engine.n_players_with_moves == 1:
                    new_state._betting_stage = "terminal"
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                if new_state._betting_stage in {"terminal", "show_down"}:
                    new_state._poker_engine.compute_winners()
                break
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state

    def _move_to_next_player(self):
        """Ensure state points to next valid active player."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _reset_betting_round_state(self):
        """Reset the state related to counting types of actions."""
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._last_raise_amount = self.big_blind
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        if self._betting_stage == "pre_flop":
            self._betting_stage = "flop"
            self._poker_engine.table.dealer.deal_flop(self._table)
        elif self._betting_stage == "flop":
            self._betting_stage = "turn"
            self._poker_engine.table.dealer.deal_turn(self._table)
        elif self._betting_stage == "turn":
            self._betting_stage = "river"
            self._poker_engine.table.dealer.deal_river(self._table)
        elif self._betting_stage == "river":
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")

    @property
    def community_cards(self) -> List[Card]:
        """Return all shared/public cards."""
        return self._table.community_cards

    @property
    def private_hands(self) -> Dict[Player, List[Card]]:
        """Return all private hands."""
        return {p: p.cards for p in self.players}

    @property
    def initial_regret(self) -> Dict[str, float]:
        """Returns the default regret for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def initial_strategy(self) -> Dict[str, float]:
        """Returns the default strategy for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def betting_stage(self) -> str:
        """Return betting stage."""
        return self._betting_stage

    @property
    def all_players_have_actioned(self) -> bool:
        """Return whether all players have made at least one action."""
        return self._n_actions >= self._n_players_started_round

    @property
    def n_players_started_round(self) -> bool:
        """Return n_players that started the round."""
        return self._n_players_started_round

    @property
    def player_i(self) -> int:
        """Get the index of the players turn it is."""
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @player_i.setter
    def player_i(self, _: Any):
        """Raise an error if player_i is set."""
        raise ValueError(f"The player_i property should not be set.")

    @property
    def betting_round(self) -> int:
        """Betting stage in integer form."""
        try:
            betting_round = self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Attempted to get betting round for stage "
                f"{self._betting_stage} but was not supported in the lut with "
                f"keys: {list(self._betting_stage_to_round.keys())}"
            )
        return betting_round

    @property
    def payout(self) -> Dict[int, int]:
        """Return player index to payout number of chips dictionary."""
        n_chips_delta = dict()
        for player_i, player in enumerate(self.players):
            n_chips_delta[player_i] = player.n_chips - self._initial_n_chips
        return n_chips_delta

    @property
    def is_terminal(self) -> bool:
        """Returns whether this state is terminal or not."""
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def players(self) -> List[Player]:
        """Returns players in table."""
        return self._table.players

    @property
    def current_player(self) -> Player:
        """Returns a reference to player that makes a move for this state."""
        return self._table.players[self.player_i]

    @property
    def pot_size(self) -> int:
        """Return the current size of the pot."""
        return self._table.pot.total

    @property
    def min_raise_amount(self) -> int:
        """Return the minimum legal raise amount."""
        return self._last_raise_amount

    def _compute_raise_chip_amount(self, pot_fraction: float, enforce_minimum: bool = True) -> int:
        """Compute the number of chips to ADD to pot based on pot fraction."""
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        effective_pot = self.pot_size
        n_chips_to_add = math.ceil(effective_pot * pot_fraction)
        if enforce_minimum:
            min_total = n_chips_to_call + self.min_raise_amount
            n_chips_to_add = max(n_chips_to_add, min_total)
        return n_chips_to_add

    def _get_available_raise_sizes(self) -> List[str]:
        """Get the available raise sizes for the current game state."""
        if self._betting_stage in {"terminal", "show_down"}:
            return []

        stage_config = RAISE_SIZES_BY_STAGE.get(self._betting_stage, {})

        if self._n_raises == 0:
            pot_fractions = stage_config.get("first_raise", [1.0])
        else:
            pot_fractions = stage_config.get("subsequent_raise", [1.0])

        raise_actions: List[str] = []
        player = self.current_player
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - player.n_bet_chips
        chips_available = player.n_chips

        added_amounts: set = set()

        for fraction in pot_fractions:
            n_chips_for_fraction = self._compute_raise_chip_amount(fraction, enforce_minimum=False)
            actual_raise_amount = n_chips_for_fraction - n_chips_to_call

            if actual_raise_amount < self.min_raise_amount:
                continue

            n_chips_to_add = self._compute_raise_chip_amount(fraction, enforce_minimum=True)

            if n_chips_to_add > chips_available:
                continue

            if n_chips_to_add >= chips_available - 1:
                continue

            if n_chips_to_add in added_amounts:
                continue

            added_amounts.add(n_chips_to_add)
            raise_actions.append(f"raise:{fraction}")

        if chips_available > 0 and chips_available >= n_chips_to_call:
            if chips_available not in added_amounts:
                raise_actions.append("all_in")

        return raise_actions

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state."""
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            biggest_bet = max(p.n_bet_chips for p in self.players)
            n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
            chips_available = self.current_player.n_chips

            actions.append("fold")

            if n_chips_to_call >= chips_available:
                if chips_available > 0:
                    actions.append("all_in")
            else:
                actions.append("call")

                if self._n_raises < MAX_RAISES_PER_ROUND:
                    raise_actions = self._get_available_raise_sizes()
                    actions += raise_actions
        else:
            actions += [None]
        return actions

    @staticmethod
    def get_canonical_actions(betting_round: int) -> List[str]:
        """Return the full abstract action set for *betting_round* in stable order."""
        stage_names = {0: "pre_flop", 1: "flop", 2: "turn", 3: "river"}
        stage = stage_names.get(betting_round)
        if stage is None:
            raise ValueError(f"betting_round must be 0-3, got {betting_round}")
        stage_config = RAISE_SIZES_BY_STAGE[stage]
        first_fracs = stage_config.get("first_raise", [])
        subseq_fracs = stage_config.get("subsequent_raise", [])
        all_fracs = sorted(set(first_fracs) | set(subseq_fracs))
        return ["fold", "call", "all_in"] + [f"raise:{f}" for f in all_fracs]

    def get_valid_mask(self) -> np.ndarray:
        """Return a boolean mask over the canonical action set for this state."""
        r = self.betting_round
        canonical = PokerState.get_canonical_actions(r)
        legal_set = {a for a in self.legal_actions if a is not None}
        return np.array([a in legal_set for a in canonical], dtype=bool)

    @staticmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False,
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """Load card information lookup table.

        Parameters
        ----------
        lut_path : str
            Path to lookup table.
        pickle_dir : bool
            Whether the lut_path is a path to pickle files or not.

        Returns
        -------
        card_info_lut : InfoSetLookupTable
            Card information cluster lookup table.
        """
        if pickle_dir:
            logger.info("Loading card information lut in deprecated way")
            file_names = [
                "preflop_lossless.pkl",
                "flop_lossy_2.pkl",
                "turn_lossy_2.pkl",
                "river_lossy_2.pkl",
            ]
            betting_stages = ["pre_flop", "flop", "turn", "river"]
            card_info_lut: Dict[str, Dict[Tuple[int, ...], str]] = {}
            for file_name, betting_stage in zip(file_names, betting_stages):
                file_path = os.path.join(lut_path, file_name)
                if not os.path.isfile(file_path):
                    raise ValueError(
                        f"File path not found {file_path}. Ensure lut_path is "
                        f"set to directory containing pickle files"
                    )
                with open(file_path, "rb") as fp:
                    card_info_lut[betting_stage] = joblib.load(fp)
        elif lut_path:
            logger.info(f"Loading card from single file at path: {lut_path}")
            card_info_lut = joblib.load(lut_path + '/card_info_lut.joblib')
        else:
            card_info_lut = {}
        return card_info_lut

    @property
    def info_set(self) -> str:
        """Get the information set for the current player."""
        cards = sorted(
            self.current_player.cards,
            key=operator.attrgetter("eval_card"),
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
        )
        if self._pickle_dir:
            lookup_cards = tuple([card.eval_card for card in cards])
        else:
            lookup_cards = tuple(cards)
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self.betting_stage not in {"terminal", "show_down"}:
                raise ValueError("You should have these cards in your lut.")
            return "default info set, please ensure you load it correctly"
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )
