"""Base state class for poker game variants.

This module provides an abstract base class that implements all common poker
game logic, allowing specific variants (Short Deck, Texas Hold'em, etc.) to
inherit and override only variant-specific behavior.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from poker_ai import utils
from poker_ai.poker.card import Card
from poker_ai.poker.engine import PokerEngine
from poker_ai.poker.player import Player
from poker_ai.poker.pot import Pot
from poker_ai.poker.table import PokerTable

logger = logging.getLogger("poker_ai.games.base.state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]

# Action abstraction configuration: raise sizes as fractions of the pot.
# Based on Pluribus blueprint strategy design.
# Pre-flop: fine-grained abstraction (no real-time search typically used)
# Flop: coarser abstraction
# Turn/River: most coarse (3 sizes for first raise, 2 for subsequent raises)
#
# IMPORTANT NOTES:
# 1. Raise amounts are calculated as: call + (pot_fraction * effective_pot)
#    where effective_pot = current_pot + amount_to_call
#    This follows standard NLHE conventions for pot-sized betting.
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


class PokerState(ABC):
    """Abstract base class for poker game state at some given point in time.

    The class is immutable and new state can be instantiated from once an
    action is applied via the `apply_action` method.
    
    Subclasses must implement:
    - _get_deck_ranks(): Return list of ranks to include in deck
    - info_set: Property that returns information set string
    - load_card_lut(): Static method to load card clustering lookup tables
    """

    def __init__(
        self,
        players: List[Player],
        small_blind: int = 50,
        big_blind: int = 100,
        lut_path: str = ".",
        pickle_dir: bool = False,
        load_card_lut: bool = True,
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
        """
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        self._pickle_dir = pickle_dir
        if load_card_lut:
            self.card_info_lut = self.load_card_lut(lut_path, self._pickle_dir)
        else:
            self.card_info_lut = {}
        # Get a reference of the pot from the first player.
        # Get variant-specific deck ranks.
        deck_ranks = self._get_deck_ranks()
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=deck_ranks
        )
        # Get a reference of the initial number of chips for the payout.
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        # Get variant-specific evaluator.
        evaluator = self._get_evaluator()
        self._poker_engine = PokerEngine(
            table=self._table,
            small_blind=small_blind,
            big_blind=big_blind,
            evaluator=evaluator,
        )
        # Reset the pot, assign betting order to players (might need to remove
        # this), assign blinds to the players.
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

    @abstractmethod
    def _get_deck_ranks(self) -> List[int]:
        """Return the list of card ranks to include in the deck.
        
        Returns
        -------
        ranks : List[int]
            List of ranks (e.g., [2, 3, ..., 14] for full deck,
            [10, 11, 12, 13, 14] for short deck).
        """
        pass

    @abstractmethod
    def _get_evaluator(self):
        """Return the hand evaluator for this poker variant.
        
        Returns
        -------
        evaluator : Evaluator
            Hand evaluator instance (e.g., Evaluator for standard poker,
            ShortDeckEvaluator for short deck).
        """
        pass

    @staticmethod
    @abstractmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """Load card information lookup table.

        Must be implemented by subclasses to load variant-specific
        clustering data.

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
        pass

    @property
    @abstractmethod
    def info_set(self) -> str:
        """Get the information set for the current player.
        
        Must be implemented by subclasses to provide variant-specific
        card clustering and information set generation.
        
        Returns
        -------
        info_set : str
            JSON string encoding the information set.
        """
        pass

    def __repr__(self):
        """Return a helpful description of object in strings and debugger."""
        class_name = self.__class__.__name__
        return f"<{class_name} player_i={self.player_i} betting_stage={self._betting_stage}>"

    def _map_to_closest_legal_action(self, invalid_action: str) -> str:
        """Map an invalid action to the closest legal action.
        
        Parameters
        ----------
        invalid_action : str
            The invalid action that was attempted.
            
        Returns
        -------
        closest_action : str
            The closest legal action to the invalid action.
        """
        legal = self.legal_actions
        
        # If the invalid action is all_in, map to all_in if available
        if invalid_action == "all_in" and "all_in" in legal:
            return "all_in"
        
        # If the invalid action is a raise with a specific fraction
        if invalid_action and invalid_action.startswith("raise:"):
            try:
                target_fraction = float(invalid_action.split(":")[1])
                
                # Find all legal raise actions and their fractions
                legal_raises = [a for a in legal if a and a.startswith("raise:")]
                
                if legal_raises:
                    # Find the closest raise by fraction
                    def get_fraction(action_str):
                        parts = action_str.split(":")
                        return float(parts[1]) if len(parts) > 1 else 0.0
                    
                    closest = min(legal_raises, key=lambda a: abs(get_fraction(a) - target_fraction))
                    return closest
                elif "all_in" in legal:
                    # If no raise fractions available but all_in is, use it
                    return "all_in"
            except (ValueError, IndexError):
                pass
        
        # Default mapping: prefer all_in > call > fold > first available
        if "all_in" in legal:
            return "all_in"
        elif "call" in legal:
            return "call"
        elif "fold" in legal:
            return "fold"
        elif legal and legal[0] is not None:
            return legal[0]
        
        return "fold"  # Last resort

    def apply_action(self, action_str: Optional[str]) -> PokerState:
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            The description of the action the current player is making. Can be
            any of {"fold", "call", "raise:<fraction>", "all_in"}. The "all_in"
            action is used whenever the player goes all-in, including when calling
            would result in going all-in.

        Returns
        -------
        new_state : PokerState
            A poker state instance that represents the game in the next
            timestep, after the action has been applied.
        """
        # Map invalid actions to closest legal action with logging
        original_action = action_str
        if action_str not in self.legal_actions:
            logger.warning(
                f"Invalid action '{action_str}' attempted. "
                f"Legal actions: {self.legal_actions}. "
                f"Mapping to closest legal action."
            )
            action_str = self._map_to_closest_legal_action(action_str)
            logger.info(f"Mapped '{original_action}' -> '{action_str}'")
        
        # Deep copy the parts of state that are needed that must be immutable
        # from state to state.
        lut = self.card_info_lut
        self.card_info_lut = {}
        new_state = copy.deepcopy(self)
        new_state.card_info_lut = self.card_info_lut = lut
        # An action has been made, so alas we are not in the first move of the
        # current betting round.
        new_state._first_move_of_current_round = False
        if action_str is None:
            # Assert active player has folded already.
            assert (
                not new_state.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            action = new_state.current_player.call(players=new_state.players)
            logger.debug("calling")
        elif action_str == "fold":
            action = new_state.current_player.fold()
        elif action_str == "all_in":
            # All-in: add all remaining chips to the pot
            # This action is used for any all-in situation (raises or calls that go all-in)
            n_chips_to_add = new_state.current_player.n_chips
            
            # Determine if this is a raise or just a call/all-in
            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            current_bet = new_state.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            
            # Only count as a raise and update minimum if it exceeds current minimum
            if actual_raise_amount >= new_state._last_raise_amount:
                new_state._last_raise_amount = actual_raise_amount
                new_state._n_raises += 1
            
            logger.debug(f"going all-in with {n_chips_to_add} chips")
            action = new_state.current_player.raise_to(n_chips=n_chips_to_add)
        elif action_str.startswith("raise:"):
            # Parse the raise action to get the pot fraction
            raise_type = action_str.split(":")[1]
            
            # IMPORTANT: Player.raise_to(n) ADDS n chips to the pot (on top of current bet)
            # So we calculate the total chips to ADD, not the final bet total
            
            # Pot-fraction raise: use EXACT pot fraction since action is legal
            # Do NOT enforce minimum - it was already validated in legal_actions
            pot_fraction = float(raise_type)
            n_chips_to_add = new_state._compute_raise_chip_amount(
                pot_fraction, enforce_minimum=False
            )
            
            # Track the actual raise amount for minimum raise enforcement
            # actual_raise_amount is the amount raised ABOVE the call amount
            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            current_bet = new_state.current_player.n_bet_chips
            n_chips_to_call = biggest_bet - current_bet
            actual_raise_amount = n_chips_to_add - n_chips_to_call
            
            # Only update minimum raise if this raise meets the current minimum
            # This prevents all-ins below minimum from allowing subsequent small raises
            if actual_raise_amount >= new_state._last_raise_amount:
                new_state._last_raise_amount = actual_raise_amount
            
            logger.debug(f"adding {n_chips_to_add} chips to pot (action: {action_str})")
            # Player.raise_to() adds chips to pot (semantically confusing name, but correct)
            action = new_state.current_player.raise_to(n_chips=n_chips_to_add)
            new_state._n_raises += 1
        else:
            raise ValueError(
                f"Unrecognized action '{action_str}'. Expected 'fold', 'call', "
                f"'raise:<fraction>' or 'all_in'."
            )
        # Update the new state.
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        new_state._history[new_state.betting_stage].append(action_str)
        new_state._n_actions += 1
        new_state._skip_counter = 0
        # Player has made move, increment the player that is next.
        while True:
            new_state._move_to_next_player()
            # If we have finished betting, (i.e: All players have put the
            # same amount of chips in), then increment the stage of
            # betting.
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
                # We have done atleast one full round of betting, increment
                # stage of the game.
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                new_state._first_move_of_current_round = True
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
                assert not new_state.current_player.is_active
            elif new_state.current_player.is_active:
                if new_state._poker_engine.n_players_with_moves == 1:
                    # No players left.
                    new_state._betting_stage = "terminal"
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                # Now check if the game is terminal.
                if new_state._betting_stage in {"terminal", "show_down"}:
                    # Distribute winnings.
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
        self._last_raise_amount = self.big_blind  # Reset to big blind each round
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        # Progress the stage of the game.
        if self._betting_stage == "pre_flop":
            # Progress from private cards to the flop.
            self._betting_stage = "flop"
            self._poker_engine.table.dealer.deal_flop(self._table)
        elif self._betting_stage == "flop":
            # Progress from flop to turn.
            self._betting_stage = "turn"
            self._poker_engine.table.dealer.deal_turn(self._table)
        elif self._betting_stage == "turn":
            # Progress from turn to river.
            self._betting_stage = "river"
            self._poker_engine.table.dealer.deal_river(self._table)
        elif self._betting_stage == "river":
            # Progress to the showdown.
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
        """Return whether all players have made atleast one action."""
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
        """Betting stagee in integer form."""
        try:
            betting_round = self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Attemped to get betting round for stage "
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
        """Returns whether this state is terminal or not.

        The state is terminal once all rounds of betting are complete and we
        are at the show down stage of the game or if all players have folded.
        """
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
        """Return the current size of the pot.
        
        Returns
        -------
        pot_size : int
            Total chips in the pot from all players.
        """
        return self._table.pot.total

    @property
    def min_raise_amount(self) -> int:
        """Return the minimum legal raise amount.
        
        In No-Limit Hold'em, the minimum raise must be at least the size
        of the previous raise, or the big blind if no previous raise.
        
        Returns
        -------
        min_raise : int
            Minimum chips to raise (on top of calling).
        """
        return self._last_raise_amount

    def _compute_raise_chip_amount(self, pot_fraction: float, enforce_minimum: bool = True) -> int:
        """Compute the number of chips to ADD to pot based on pot fraction.
        
        This method implements standard NLHE pot-sized raise conventions:
        - Effective pot = current pot + amount needed to call
        - Raise amount = pot_fraction * effective_pot (rounded up)
        - Total chips to add = call amount + raise amount
        
        IMPORTANT: Returns the total chips to ADD via Player.raise_to(n_chips).
        Player.raise_to() ADDS n_chips to the pot on top of current bet, despite
        the confusing method name. It does NOT set total bet to n_chips.
        
        INTENDED BEHAVIOR:
        - Raises are applied ON TOP OF the call amount, not instead of it
        - In multi-way pots, later positions see larger pots (positional asymmetry)
          and thus have larger raise sizes for the same abstract action
        - This asymmetry is realistic and allows CFR to learn position-dependent strategies
        
        Example: Pot=$100, need to call $50
        - Effective pot = $100 + $50 = $150
        - Pot-sized raise (fraction=1.0): raise $150, ADD $50 + $150 = $200 to pot
        
        Parameters
        ----------
        pot_fraction : float
            Fraction of the pot to raise (e.g., 0.5 for half-pot, 1.0 for pot).
        enforce_minimum : bool
            If True, ensures raise meets minimum raise requirement.
            If False, returns the exact pot-fraction amount (rounded up).
            
        Returns
        -------
        n_chips_to_add : int
            Number of chips to ADD to pot (includes call amount + raise amount).
        """
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
        
        # Pot size for raise calculation includes current pot + amount to call
        # This is the "effective pot" that the opponent would face
        effective_pot = self.pot_size + n_chips_to_call
        
        # Raise amount is fraction of the effective pot (rounded UP to integer)
        raise_amount = math.ceil(effective_pot * pot_fraction)
        
        # Ensure minimum raise requirement if requested
        if enforce_minimum:
            raise_amount = max(raise_amount, self.min_raise_amount)
        
        # Total chips to ADD to pot = call + raise
        # This is what we pass to Player.raise_to() which adds chips
        n_chips_to_add = n_chips_to_call + raise_amount
        
        return n_chips_to_add

    def _get_available_raise_sizes(self) -> List[str]:
        """Get the available raise sizes for the current game state.
        
        Returns raise actions as strings like "raise:0.5", "raise:1.0", "all_in".
        Filters out raise sizes that would exceed player's chip stack or be
        less than min raise.
        
        Returns
        -------
        raise_actions : List[str]
            List of available raise action strings (including "all_in").
        """
        if self._betting_stage in {"terminal", "show_down"}:
            return []
        
        # Get raise size configuration for current betting stage
        stage_config = RAISE_SIZES_BY_STAGE.get(self._betting_stage, {})
        
        # Determine if this is the first raise or a subsequent raise
        if self._n_raises == 0:
            pot_fractions = stage_config.get("first_raise", [1.0])
        else:
            pot_fractions = stage_config.get("subsequent_raise", [1.0])
        
        raise_actions: List[str] = []
        player = self.current_player
        biggest_bet = max(p.n_bet_chips for p in self.players)
        n_chips_to_call = biggest_bet - player.n_bet_chips
        chips_available = player.n_chips
        
        # Track chips amounts to avoid duplicates and detect all-in equivalents
        added_amounts: set = set()
        
        for fraction in pot_fractions:
            # First check if this pot fraction would meet minimum raise (without enforcement)
            n_chips_for_fraction = self._compute_raise_chip_amount(fraction, enforce_minimum=False)
            
            # Calculate the actual raise amount (not including call)
            actual_raise_amount = n_chips_for_fraction - n_chips_to_call
            
            # Skip if this raise would be below the minimum legal raise
            if actual_raise_amount < self.min_raise_amount:
                continue
            
            # Now compute with enforcement to ensure we meet all requirements
            n_chips_to_add = self._compute_raise_chip_amount(fraction, enforce_minimum=True)
            
            # Skip if player can't afford this raise
            if n_chips_to_add > chips_available:
                continue
            
            # Skip if this would be effectively all-in (within 1 chip tolerance)
            # We'll add "all_in" separately to keep semantics clear
            if n_chips_to_add >= chips_available - 1:
                continue
            
            # Skip duplicates (can happen with rounding or min raise adjustments)
            if n_chips_to_add in added_amounts:
                continue
            
            added_amounts.add(n_chips_to_add)
            raise_actions.append(f"raise:{fraction}")
        
        # ALWAYS add all-in option if player has chips and can at least call
        # All-in is valid even if below minimum raise (NLHE rules allow it)
        if chips_available > 0 and chips_available >= n_chips_to_call:
            # Only add if not already covered by a pot-fraction action
            if chips_available not in added_amounts:
                raise_actions.append("all_in")
        
        return raise_actions

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state.
        
        Action abstraction provides multiple raise sizes as fractions of the pot:
        - Pre-flop: Fine-grained (many options for blueprint strategy)
        - Flop: Coarser abstraction
        - Turn/River: Most coarse (3 options for first raise, 2 for subsequent)
        
        Raise actions are formatted as "raise:<fraction>" or "all_in".
        Examples: "raise:0.5" (half-pot), "raise:1.0" (pot-size), "all_in"
        
        The "all_in" action is used whenever going all-in, including when calling
        would result in all-in. In such cases, "call" is not available.
        
        POSITIONAL ASYMMETRY (INTENDED):
        In multi-way pots, players acting later will have different raise sizes
        than earlier players for the same pot fraction, because they see a larger
        pot after earlier players have acted. This is realistic poker behavior and
        the CFR algorithm naturally learns position-dependent strategies.
        
        Returns
        -------
        actions : List[Optional[str]]
            List of legal action strings.
        """
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            # Check if calling would result in all-in
            biggest_bet = max(p.n_bet_chips for p in self.players)
            n_chips_to_call = biggest_bet - self.current_player.n_bet_chips
            chips_available = self.current_player.n_chips
            
            actions.append("fold")
            
            # If calling would be all-in, use "all_in" action instead of "call"
            if n_chips_to_call >= chips_available:
                if chips_available > 0:  # Only if player has chips to add
                    actions.append("all_in")
            else:
                # Normal call is available
                actions.append("call")
                
                # Check if raises are still allowed this round
                if self._n_raises < MAX_RAISES_PER_ROUND:
                    # Get pot-fraction based raise actions (includes "all_in" if raising)
                    raise_actions = self._get_available_raise_sizes()
                    actions += raise_actions
        else:
            actions += [None]
        return actions
