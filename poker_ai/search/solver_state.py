"""Shared state and configuration for the depth-limited subgame solver (§6.5).

This module is the **regime-agnostic foundation** of the solver: it owns the
in-memory CFR tables and the operations on them (regret matching, accumulation,
Linear-CFR discount, averaging, freezing), plus the config and the ``Key``
helpers.  It deliberately imports **nothing** from the rest of the ``search``
package at runtime — the two CFR regimes (``mccfr.py`` / ``vector.py``) and the
orchestrator (``solver.py``) import *it*, so keeping it a leaf of the import
graph is what lets the package split cleanly without an import cycle (the
``LeafConfig`` reference on :class:`SolverConfig` is a ``TYPE_CHECKING``-only
annotation for exactly this reason).

The tables are keyed by ``Key = (public_key, hand_row)`` where ``public_key`` is
``env.public_key`` (the shared public-state identifier) and ``hand_row`` is the
acting seat's combo index on the **root street** (lossless) or its LUT cluster id
on later streets (:func:`_hand_row`).  Rows are **per-node width** — the number of
legal actions at that public node — so the same dicts hold both the MCCFR regime's
scalar rows and (later) the vector regime's per-combo rows without any reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

from poker_ai.blueprint.tree_utils import calculate_strategy_from_row

if TYPE_CHECKING:  # avoid a runtime import cycle (leaf -> policy -> solver_state)
    from environment.poker_env import PokerEnv
    from poker_ai.search.leaf import LeafConfig


# A public node identifier.  ``env.public_key`` is ``(betting_stage, history)``;
# the meta-game extends it to ``(env.public_key, "META", seat)``.  Either way it
# is a hashable tuple — the alias documents intent, it is not enforced.
PublicKey = Tuple
# ``(public_key, hand_row)`` — hand_row is a combo index (root street) or an LUT
# cluster id (later streets); see :func:`_hand_row`.
Key = Tuple[PublicKey, int]


@dataclass(frozen=True)
class SolverConfig:
    """Static solver hyperparameters (§6.5).

    ``leaf`` is required and carries the continuation-strategy fleet + the
    ``use_decision_free_equity`` toggle consumed at depth-limit leaves and
    forced-runout terminals.
    """

    leaf: "LeafConfig"
    max_iterations: int = 10_000
    max_wall_seconds: float = 15.0
    discount_interval: int = 1_000  # Linear-CFR discount cadence (iterations)


def _hand_row(env: "PokerEnv", combo: Sequence[int], street_at_root: int) -> int:
    """Row id for ``combo`` at ``env``'s current node.

    Per-combo (lossless) on the root street, an LUT cluster id on later streets
    (§6.5).  ``combo`` is sorted to match ``env.combo_index`` / ``cluster_for``
    key order.  The combo is assumed board-compatible (the solver only ever keys
    on the acting seat's sampled, board-disjoint hole).
    """
    c = tuple(sorted(int(x) for x in combo))
    if env.betting_round == street_at_root:
        return int(env.combo_index[c])
    return int(env.cluster_for(c))


@dataclass
class SolverState:
    """In-memory CFR tables for one search (warm-start carrier, §6.5).

    All four dicts are mutated in place across iterations.  ``regret`` and
    ``strat_sum`` rows are float64 and exactly ``width(public_key)`` wide;
    ``frozen`` pins a probability vector for the bot's actual-hand rows (§5).
    """

    regret: Dict[Key, np.ndarray] = field(default_factory=dict)
    strat_sum: Dict[Key, np.ndarray] = field(default_factory=dict)
    legal_at: Dict[PublicKey, Tuple[str, ...]] = field(default_factory=dict)
    actor_at: Dict[PublicKey, int] = field(default_factory=dict)
    frozen: Dict[Key, np.ndarray] = field(default_factory=dict)
    # Vector regime (§6.5): per-public-node ``(n_combos, width)`` matrices, the
    # combo axis indexed by ``combo_index`` (lossless, every depth).  These are the
    # vector regime's native storage — the hot loop is one vectorised op per node
    # rather than ~n_combos dict rows.  A given ``SolverState`` is only ever written
    # by one regime, so ``regret``/``strat_sum`` (MCCFR) and ``vregret``/``vstrat``
    # (vector) never both populate; the readers below dispatch on which is present.
    vregret: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    vstrat: Dict[PublicKey, np.ndarray] = field(default_factory=dict)
    # Search-lifetime caches (§6.4.2, §6.7 Tier 1).  Both hold values that are
    # **invariant across CFR iterations** for a fixed key, so they are *not*
    # touched by ``discount`` and persist across a warm-started re-search:
    #   ``leaf_value_cache`` — ``continuation_value`` keyed by
    #     ``(leaf public_key, all-seat holes, profile)`` (§6.4.1).
    #   ``runout_cache`` — exact decision-free ``runout_equity`` keyed by
    #     ``(all-seat holes, runout snapshot)``; shared by the leaf rollouts
    #     (a leaf's four bias calls) and the forced-runout terminal (§6.4.2).
    leaf_value_cache: Dict = field(default_factory=dict)
    runout_cache: Dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SolverState":
        return cls()

    # ------------------------------------------------------------------
    # Node registration (+ warm-start widening)
    # ------------------------------------------------------------------

    def width(self, public_key: PublicKey) -> int:
        return len(self.legal_at[public_key])

    def ensure_node(
        self, public_key: PublicKey, legal_actions: Sequence[str], actor: int
    ) -> None:
        """Register a public node's legal-action list and acting seat.

        First visit records ``legal_at``/``actor_at``.  On a **widened** legal
        set (warm-start re-search after an off-tree action was injected at this
        node) every row under ``public_key`` is grown to the new width, with old
        columns preserved and injected actions zero-initialised — so prior
        regrets/strategy survive the re-search.  Widening only ever *adds*
        actions; the action→column mapping is by string, so a reordered legal
        set is handled correctly.
        """
        legal = tuple(legal_actions)
        prev = self.legal_at.get(public_key)
        if prev is None:
            self.legal_at[public_key] = legal
            self.actor_at[public_key] = actor
            return
        if prev != legal:
            self._grow_rows(public_key, prev, legal)
            self.legal_at[public_key] = legal

    def ensure_vnode(
        self,
        public_key: PublicKey,
        legal_actions: Sequence[str],
        actor: int,
        n_combos: int,
    ) -> None:
        """Register a public node and lazily allocate its vector-regime matrices.

        Like :meth:`ensure_node` (it delegates the ``legal_at``/``actor_at``
        bookkeeping and warm-start widening — which now also grows the matrix
        **columns**), but additionally allocates the ``(n_combos, width)``
        ``vregret``/``vstrat`` matrices for ``public_key`` on first visit.  The
        combo axis is indexed by ``combo_index`` (lossless at every depth, §6.5).
        """
        self.ensure_node(public_key, legal_actions, actor)
        if public_key not in self.vregret:
            width = len(self.legal_at[public_key])
            self.vregret[public_key] = np.zeros((n_combos, width), dtype=np.float64)
            self.vstrat[public_key] = np.zeros((n_combos, width), dtype=np.float64)

    def _grow_rows(
        self,
        public_key: PublicKey,
        prev_legal: Tuple[str, ...],
        new_legal: Tuple[str, ...],
    ) -> None:
        new_index = {a: i for i, a in enumerate(new_legal)}
        cols = [new_index[a] for a in prev_legal]  # old col -> new col
        width = len(new_legal)
        for table in (self.regret, self.strat_sum, self.frozen):
            for key in list(table):
                if key[0] == public_key:
                    old = table[key]
                    grown = np.zeros(width, dtype=old.dtype)
                    grown[cols] = old
                    table[key] = grown
        # Vector regime: grow the per-public-node matrices along the action axis.
        for table in (self.vregret, self.vstrat):
            mat = table.get(public_key)
            if mat is not None:
                grown = np.zeros((mat.shape[0], width), dtype=mat.dtype)
                grown[:, cols] = mat
                table[public_key] = grown

    # ------------------------------------------------------------------
    # Strategy / regret operations
    # ------------------------------------------------------------------

    def sigma(self, key: Key) -> np.ndarray:
        """Pure regret-matched strategy at ``key``.

        The row *is* the legal set, so no ``valid_mask`` is needed — an unseen
        or all-non-positive row falls back to uniform over the node's actions.

        Freezing is **not** applied here: on later streets a row is keyed by an
        LUT *cluster*, which many holes share, so the bot's pinned strategy must
        only be substituted when the *sampled hole equals the bot's actual hole*
        — a condition the caller checks (it is encoded in the key only on the
        root street).  ``frozen`` is therefore consulted by the consumers
        (:mod:`mccfr`, :class:`SearchPolicy`) under that guard, not here.

        Vector regime: when ``key``'s public node has a regret **matrix**
        (``vregret``), regret-match its ``combo_index`` row.
        """
        mat = self.vregret.get(key[0])
        if mat is not None:
            return calculate_strategy_from_row(mat[key[1]])
        width = len(self.legal_at[key[0]])
        row = self.regret.get(key)
        if row is None:
            row = np.zeros(width, dtype=np.float64)
        return calculate_strategy_from_row(row)

    def add_regret(self, key: Key, delta: np.ndarray) -> None:
        row = self.regret.get(key)
        if row is None:
            row = np.zeros(len(delta), dtype=np.float64)
            self.regret[key] = row
        row += np.asarray(delta, dtype=np.float64)

    def add_strat(self, key: Key, sigma: np.ndarray) -> None:
        row = self.strat_sum.get(key)
        if row is None:
            row = np.zeros(len(sigma), dtype=np.float64)
            self.strat_sum[key] = row
        row += np.asarray(sigma, dtype=np.float64)

    def discount(self, factor: float) -> None:
        """Linear-CFR discount: scale every regret and strategy-sum row.

        Mirrors ``CFRTables.apply_discount`` (in-memory, no floor clamp — the
        blueprint's regret floor guards a long training run, not a short search).
        Scales the vector regime's per-node matrices too.
        """
        for table in (self.regret, self.strat_sum, self.vregret, self.vstrat):
            for row in table.values():
                row *= factor

    def average_sigma(self, key: Key) -> "np.ndarray | None":
        """Normalised cumulative strategy at ``key``; ``None`` if unaccumulated.

        Vector regime: read the ``combo_index`` row of the ``vstrat`` matrix.
        """
        mat = self.vstrat.get(key[0])
        if mat is not None:
            row = mat[key[1]]
        else:
            row = self.strat_sum.get(key)
        if row is None:
            return None
        total = row.sum()
        if total <= 0.0:
            return None
        return (row / total).astype(np.float32)
