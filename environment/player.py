"""Poker player data and betting logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.pot import Pot

logger = logging.getLogger(__name__)


class Player:
    """A poker player: holds chips, hole cards, and positional flags.

    Hole cards are stored as a tuple of 32-bit card integers (see
    ``utils.py`` for the encoding). Betting methods take the shared
    ``Pot`` as an explicit argument.

    Attributes
    ----------
    player_i : int
        Zero-based player index; used as the key in the shared Pot.
    name : str
        Display name.
    n_chips : int
        Chips currently in the player's stack (not in the pot).
    n_bet_chips : int
        Chips committed to the pot in the current betting round.
    is_small_blind : bool
        Whether this player posted the small blind.
    is_big_blind : bool
        Whether this player posted the big blind.
    is_dealer : bool
        Whether this player is the dealer (button).
    is_turn : bool
        Whether it is currently this player's turn to act.
    order : int
        Betting order index (set by ``dynamics.assign_order``).
    """

    __slots__ = (
        "player_i",
        "name",
        "n_chips",
        "n_bet_chips",
        "_cards",
        "_is_active",
        "is_small_blind",
        "is_big_blind",
        "is_dealer",
        "is_turn",
        "order",
    )

    def __init__(
        self,
        player_i: int,
        initial_chips: int = 10000,
        name: str = None,
    ):
        """Initialise a player.

        Parameters
        ----------
        player_i : int
            Zero-based player index used as the key in the shared Pot.
        initial_chips : int
            Starting chip count.
        name : str, optional
            Display name. Defaults to ``"player_{player_i}"``.
        """
        self.player_i: int = player_i
        self.name: str = name if name is not None else f"player_{player_i}"
        self.n_chips: int = initial_chips
        self.n_bet_chips: int = 0
        self._cards: tuple = ()
        self._is_active: bool = True
        self.is_small_blind: bool = False
        self.is_big_blind: bool = False
        self.is_dealer: bool = False
        self.is_turn: bool = False
        self.order: int = 0

    def __repr__(self) -> str:
        return (
            f'<Player name="{self.name}" n_chips={self.n_chips:05d} '
            f"n_bet_chips={self.n_bet_chips:05d} folded={int(not self._is_active)}>"
        )

    # ------------------------------------------------------------------
    # Chip management
    # ------------------------------------------------------------------

    def add_chips(self, chips: int) -> None:
        """Add chips to this player's stack.

        Parameters
        ----------
        chips : int
            Number of chips to add.
        """
        self.n_chips += chips

    def add_to_pot(self, pot: Pot, n_chips: int) -> int:
        """Commit chips to the pot, capped to the player's remaining stack.

        Parameters
        ----------
        pot : Pot
            The shared pot instance.
        n_chips : int
            Desired number of chips to add.

        Returns
        -------
        int
            Actual number of chips added (may be less if going all-in).
        """
        if n_chips < 0:
            raise ValueError("Cannot subtract chips from pot.")
        actual = min(n_chips, self.n_chips)
        pot.add_chips(self.player_i, actual)
        self.n_chips -= actual
        self.n_bet_chips += actual
        return actual

    # ------------------------------------------------------------------
    # Actions (called from dynamics.py or poker_env.py)
    # ------------------------------------------------------------------

    def fold(self) -> None:
        """Fold: deactivate this player for the rest of the hand."""
        self._is_active = False

    def call(self, players: list, pot: Pot) -> None:
        """Call the highest current bet.

        Parameters
        ----------
        players : list[Player]
            All players at the table (to find the biggest bet).
        pot : Pot
            The shared pot instance.
        """
        if self.is_all_in:
            return
        biggest_bet = max(p.n_bet_chips for p in players)
        n_chips_to_call = biggest_bet - self.n_bet_chips
        self.add_to_pot(pot, n_chips_to_call)

    def raise_to(self, pot: Pot, n_chips: int) -> None:
        """Raise the total bet for this round to ``n_chips``.

        Parameters
        ----------
        pot : Pot
            The shared pot instance.
        n_chips : int
            Total chips to commit this round (includes implied call amount).
        """
        self.add_to_pot(pot, n_chips)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cards(self) -> tuple:
        """Hole cards as a tuple of eval_card integers."""
        return self._cards

    @property
    def is_active(self) -> bool:
        """Whether the player is still in the hand (has not folded)."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._is_active = value

    @property
    def is_all_in(self) -> bool:
        """True if the player is active but has no chips remaining."""
        return self._is_active and self.n_chips == 0

    def capture_mutable(self) -> tuple:
        """Snapshot the fields an action can mutate, for make/undo.

        ``_cards`` / positional flags / ``order`` are fixed for the hand
        and are not included.  Returns an immutable tuple, cheap to hold.
        """
        return (self.n_chips, self.n_bet_chips, self._is_active, self.is_turn)

    def restore_mutable(self, snap: tuple) -> None:
        """Restore the fields captured by :meth:`capture_mutable`."""
        self.n_chips, self.n_bet_chips, self._is_active, self.is_turn = snap
