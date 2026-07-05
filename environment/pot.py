"""Pot: chip tracking, side pot computation, and utility calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.player import Player


class Pot:
    """Tracks chips contributed by each player and computes payouts.

    Players are identified by their integer ``player_i`` index.
    Side pot computation and utility distribution are handled here
    because the Pot owns the per-player contribution data needed to
    determine how much each player is eligible to win from each sub-pot.

    Parameters
    ----------
    n_players : int
        Number of players at the table.

    Attributes
    ----------
    total : int
        Total chips currently in the pot across all players.
    side_pots : list[dict[int, int]]
        Computed side pot structure (see ``side_pots`` property).
    """

    __slots__ = ("_chips",)

    def __init__(self, n_players: int):
        self._chips: list = [0] * n_players

    def __repr__(self) -> str:
        return f"<Pot n_chips={self.total}>"

    def __getitem__(self, player_i: int) -> int:
        """Return chips contributed by player ``player_i``."""
        return self._chips[player_i]

    def add_chips(self, player_i: int, n_chips: int) -> None:
        """Record ``n_chips`` contributed by ``player_i``.

        Parameters
        ----------
        player_i : int
            Index of the contributing player.
        n_chips : int
            Number of chips to add.
        """
        self._chips[player_i] += n_chips

    def reset(self) -> None:
        """Clear all contributions, zeroing every player's entry."""
        self._chips = [0] * len(self._chips)

    def capture(self) -> list:
        """Snapshot per-player contributions for make/undo (a copy)."""
        return list(self._chips)

    def restore(self, snap: list) -> None:
        """Restore the contributions captured by :meth:`capture`."""
        self._chips = list(snap)

    @property
    def total(self) -> int:
        """Total chips currently in the pot."""
        return sum(self._chips)

    @property
    def side_pots(self) -> list:
        """Compute the side pot structure from per-player contributions.

        Returns
        -------
        list[dict[int, int]]
            Each element is a dict mapping player_i → chips contributed
            to that specific side pot.  The list is ordered from the
            smallest (all-in) pot to the largest.
        """
        side_pots: list = []
        if not any(self._chips):
            return []
        pot = {i: v for i, v in enumerate(self._chips) if v > 0}
        while pot:
            side_pots.append({})
            min_chips = min(pot.values())
            to_remove = []
            for player_i, chips in pot.items():
                side_pots[-1][player_i] = min_chips
                pot[player_i] -= min_chips
                if pot[player_i] == 0:
                    to_remove.append(player_i)
            for player_i in to_remove:
                del pot[player_i]
        return side_pots

    def compute_utility(self, players: list, ranked_groups: list) -> dict:
        """Compute chip delta per player at the end of a hand.

        Distributes each side pot to the highest-ranked group of players
        that contributed to it.  Remainder chips (from integer division)
        are awarded one at a time in bet-order.

        Parameters
        ----------
        players : list[Player]
            All players (used to look up ``player_i`` and ``order``).
        ranked_groups : list[list[Player]]
            Players grouped by hand rank, best first (each group shares
            the same hand strength and splits the pot equally).

        Returns
        -------
        dict[int, int]
            Mapping of ``player_i`` → chips won (0 for losers).
        """
        payouts: dict = {p.player_i: 0 for p in players}
        for side_pot in self.side_pots:
            for player_group in ranked_groups:
                winners = self._players_eligible_for_pot(player_group, side_pot)
                if winners:
                    self._split_side_pot(winners, side_pot, payouts)
                    break
        return payouts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _players_eligible_for_pot(self, player_group: list, side_pot: dict) -> list:
        """Return players from ``player_group`` who contributed to ``side_pot``."""
        eligible = [p for p in player_group if p.player_i in side_pot]
        return sorted(eligible, key=lambda p: p.order)

    def _split_side_pot(self, winners: list, side_pot: dict, payouts: dict) -> None:
        """Split ``side_pot`` equally among ``winners``, remainder to first winner."""
        n_total = sum(side_pot.values())
        n_winners = len(winners)
        per_player = n_total // n_winners
        remainder = n_total - n_winners * per_player
        for player in winners:
            payouts[player.player_i] += per_player
        for i in range(remainder):
            payouts[winners[i].player_i] += 1


# ---------------------------------------------------------------------------
# Optional compiled-core settlement (Phase 1e)
# ---------------------------------------------------------------------------
# Keep the pure-Python ``compute_utility`` as the byte-exact oracle, then swap it
# for the Cython kernel when the extension is built AND enabled
# (``PLURIBUS_CORE_KERNELS`` includes ``settlement``).  The kernel reproduces the
# side-pot peel + odd-chip remainder exactly (verified by randomized differential
# fixtures and the golden trace).  The thin wrapper hands the kernel the plain
# integer inputs it wants — ``pot._chips``, per-``player_i`` ``order``, and the
# ranked groups as ``player_i`` lists — and rebuilds the ``{player_i: won}`` dict
# ``compute_winners`` expects.  ``compute_winners`` calls ``self.compute_utility``
# at hand end, so swapping the class method is transparent.
_compute_utility_py = Pot.compute_utility

try:
    from poker_ai._core import CORE_AVAILABLE as _CORE_AVAILABLE
    from poker_ai._core.flags import kernel_enabled as _kernel_enabled

    if _CORE_AVAILABLE and _kernel_enabled("settlement"):
        from poker_ai._core._settle import (
            compute_utility_won as _core_compute_utility_won,
        )

        def _compute_utility_core(self, players: list, ranked_groups: list) -> dict:
            """Cython-backed :meth:`Pot.compute_utility` (flag ``settlement``)."""
            order = [0] * len(self._chips)
            for p in players:
                order[p.player_i] = p.order
            groups_pi = [[p.player_i for p in g] for g in ranked_groups]
            won = _core_compute_utility_won(self._chips, groups_pi, order)
            return {p.player_i: won[p.player_i] for p in players}

        Pot.compute_utility = _compute_utility_core
except ImportError:
    pass
