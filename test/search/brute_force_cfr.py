"""Independent brute-force CFR oracle for the subgame solver (§9, row 6.2).

This module is a **wholly independent** equilibrium reference for the depth-limited
subgame solver.  It solves a tiny **heads-up river** subgame to its exact Nash
equilibrium with its own regret/strategy tables and its own recursion, reusing the
:class:`~environment.poker_env.PokerEnv` only for game **rules** (the legal-action
tree) and **terminal payoffs** (concrete ``env.payout``).  It deliberately imports
**none** of the production solver internals — no ``SolverState``, ``_MCCFRSolver``,
``_VectorSolver``, or ``vector_payout`` — so that a regression in either regime
shows up as a measurable disagreement against this reference (§9
"Independent CFR cross-validation").

The river is the documented sweet spot (§9): the board is complete, so the betting
tree is **deterministic and exhaustively enumerable** — no board chance, no
depth-limit leaf, no decision-free runout, no meta-game — and the game is
**2-player zero-sum**, so a true Nash exists for both solver paths to match.

Three pieces:

- :func:`build_subgame` walks the env once (make/undo) to capture the **public
  betting tree** — hole-independent on the river — then builds a concrete
  **payoff tensor** ``M[leaf][(a, b)]`` by replaying each terminal line under every
  card-disjoint support pair via :meth:`PokerEnv.with_hole_cards` + ``env.payout``.
- :class:`BruteForceCFR` runs full-enumeration Linear CFR over that static tree.
- :func:`game_value`, :func:`br_value`, :func:`exploitability` are the metrics —
  the unique zero-sum game value, an exact best response by per-hole backward
  induction, and the best-response gap (the Nash-equilibrium test).

All of the metrics operate purely on the in-memory tree + tensor, independent of
the env and of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# Infoset: (acting seat, that seat's hole-combo index, public_key).  Imperfect
# information — the actor knows its own hole and the public history, not the
# opponent's hole.  ``Hole`` is the sorted ``(c0, c1)`` card-int tuple.
Hole = Tuple[int, int]
Infoset = Tuple[int, int, Tuple]
Pair = Tuple[int, int]  # (seat-0 combo index, seat-1 combo index)


@dataclass
class Subgame:
    """The static public tree + concrete payoff tensor of one river subgame.

    Attributes
    ----------
    root : dict
        The public betting tree.  Internal nodes are
        ``{"type": "node", "actor", "pk", "legal", "children"}``; terminals are
        ``{"type": "term", "leaf", "pk"}``.
    payoff : dict[int, dict[Pair, float]]
        ``M[leaf_id][(a, b)]`` — net chips to **seat 0** at terminal ``leaf_id``
        when seat 0 holds combo ``a`` and seat 1 holds combo ``b`` (zero-sum, so
        seat 1's payoff is the negation).  Only card-disjoint pairs are present.
    support : dict[int, list[int]]
        Per-seat list of combo indices with positive, board-compatible reach.
    holes : dict[int, Hole]
        Combo index → sorted ``(c0, c1)`` card-int tuple.
    weight : dict[Pair, float]
        Chance measure: the joint hole-deal distribution ``∝ range0[a]·range1[b]``
        over card-disjoint pairs, normalised to sum to 1.
    """

    root: dict
    payoff: Dict[int, Dict[Pair, float]]
    support: Dict[int, List[int]]
    holes: Dict[int, Hole]
    weight: Dict[Pair, float]


def build_subgame(
    env,
    range0: Sequence[float],
    range1: Sequence[float],
    support0: Sequence[int],
    support1: Sequence[int],
) -> Subgame:
    """Capture the public tree and concrete payoff tensor rooted at ``env``.

    ``env`` must be a heads-up river state (complete board, not terminal).  The
    walk uses make/undo and leaves ``env`` unchanged; ``with_hole_cards`` then
    deepcopies it for each (terminal line, support pair) to read the concrete
    ``env.payout``.  ``support0`` / ``support1`` must list board-compatible combo
    indices (their cards share nothing with the community).
    """
    holes: Dict[int, Hole] = {
        idx: tuple(int(x) for x in env.combo_cards[idx])
        for idx in set(support0) | set(support1)
    }

    # --- Public betting tree (hole-independent on the river) via make/undo. ---
    leaves: List[List[str]] = []

    def build(e, line: List[str]) -> dict:
        if e.is_terminal:
            leaf_id = len(leaves)
            leaves.append(list(line))
            return {"type": "term", "leaf": leaf_id, "pk": e.public_key}
        legal = [a for a in e.legal_actions if a is not None]
        node = {
            "type": "node",
            "actor": e.player_i,
            "pk": e.public_key,
            "legal": legal,
            "children": {},
        }
        for a in legal:
            token = e.step_in_place(a)
            node["children"][a] = build(e, line + [a])
            e.undo(token)
        return node

    root = build(env, [])

    # --- Chance measure: joint hole deal over card-disjoint support pairs. ---
    raw: Dict[Pair, float] = {}
    for a in support0:
        for b in support1:
            if set(holes[a]) & set(holes[b]):
                continue  # share a card — impossible joint deal (card removal)
            raw[(a, b)] = float(range0[a]) * float(range1[b])
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("subgame has no card-disjoint support pairs with mass.")
    weight = {pair: m / total for pair, m in raw.items()}

    # --- Concrete payoff tensor: replay each terminal line per support pair. ---
    payoff: Dict[int, Dict[Pair, float]] = {}
    for leaf_id, line in enumerate(leaves):
        row: Dict[Pair, float] = {}
        for (a, b) in raw:
            e2 = env.with_hole_cards([holes[a], holes[b]])
            for action in line:
                if e2.is_terminal:
                    break
                e2.step_in_place(action)
            row[(a, b)] = float(e2.payout[0])
        payoff[leaf_id] = row

    return Subgame(
        root=root,
        payoff=payoff,
        support={0: list(support0), 1: list(support1)},
        holes=holes,
        weight=weight,
    )


def _regret_match(regret: np.ndarray) -> np.ndarray:
    """Positive-regret-proportional strategy; uniform fallback on no positive."""
    pos = np.maximum(regret, 0.0)
    total = pos.sum()
    if total > 0.0:
        return pos / total
    return np.full(len(regret), 1.0 / len(regret))


class BruteForceCFR:
    """Full-enumeration Linear CFR over a :class:`Subgame` (the oracle).

    One :meth:`iterate` enumerates **every** card-disjoint hole pair (no
    sampling) and runs a full vanilla-CFR tree pass per pair, updating the acting
    seat's infoset at every node: regret weighted by the counterfactual reach
    (chance × opponent action-reach) and strategy-sum by the seat's own action
    reach.  Contributions are Linear-CFR ``t``-weighted.  The per-pair
    enumeration over-weights an infoset's strategy-sum by a constant factor (the
    count of compatible opponent holes), which cancels under per-infoset
    normalisation, so the average strategy is unbiased.
    """

    def __init__(self, subgame: Subgame) -> None:
        self.sub = subgame
        self.regret: Dict[Infoset, np.ndarray] = {}
        self.strat: Dict[Infoset, np.ndarray] = {}

    def _row(self, table: Dict[Infoset, np.ndarray], key: Infoset, width: int) -> np.ndarray:
        row = table.get(key)
        if row is None:
            row = np.zeros(width, dtype=np.float64)
            table[key] = row
        return row

    def iterate(self, t: float) -> None:
        for (a, b), w in self.sub.weight.items():
            self._cfr(self.sub.root, a, b, 1.0, 1.0, w, t)

    def _cfr(self, node, a: int, b: int, r0: float, r1: float, w: float, t: float) -> float:
        if node["type"] == "term":
            return self.sub.payoff[node["leaf"]][(a, b)]
        seat = node["actor"]
        legal = node["legal"]
        width = len(legal)
        hole = a if seat == 0 else b
        key = (seat, hole, node["pk"])
        regret = self._row(self.regret, key, width)
        sigma = _regret_match(regret)

        util = np.empty(width, dtype=np.float64)
        node_util = 0.0
        for k, action in enumerate(legal):
            child = node["children"][action]
            if seat == 0:
                u = self._cfr(child, a, b, r0 * sigma[k], r1, w, t)
            else:
                u = self._cfr(child, a, b, r0, r1 * sigma[k], w, t)
            util[k] = u
            node_util += sigma[k] * u

        # Counterfactual reach excludes the actor's own contribution; utilities
        # are in seat-0 units, so seat 1 negates them.
        if seat == 0:
            cf_reach, own_reach, sign = w * r1, r0, 1.0
        else:
            cf_reach, own_reach, sign = w * r0, r1, -1.0
        regret += t * cf_reach * sign * (util - node_util)
        self._row(self.strat, key, width)[:] += t * own_reach * sigma
        return node_util

    def solve(self, iterations: int) -> Dict[Infoset, np.ndarray]:
        for t in range(1, iterations + 1):
            self.iterate(float(t))
        return self.average_strategy()

    def average_strategy(self) -> Dict[Infoset, np.ndarray]:
        """Normalised cumulative strategy per infoset (uniform if unaccumulated)."""
        out: Dict[Infoset, np.ndarray] = {}
        for key, row in self.strat.items():
            total = row.sum()
            out[key] = row / total if total > 0.0 else np.full(len(row), 1.0 / len(row))
        return out


# --------------------------------------------------------------------------- #
# Metrics (operate purely on the static tree + payoff tensor)
# --------------------------------------------------------------------------- #

def _sigma_row(sigma: Mapping[Infoset, np.ndarray], seat: int, hole: int, node) -> np.ndarray:
    """Strategy row for ``(seat, hole, node.pk)``; uniform over legal if absent."""
    row = sigma.get((seat, hole, node["pk"]))
    if row is None:
        n = len(node["legal"])
        return np.full(n, 1.0 / n)
    return row


def game_value(sub: Subgame, sigma: Mapping[Infoset, np.ndarray]) -> float:
    """Expected value to **seat 0** of the profile ``sigma`` (both seats).

    The unique zero-sum game value when ``sigma`` is an equilibrium — the robust
    quantity to compare across solver paths.
    """

    def ev(node, a: int, b: int) -> float:
        if node["type"] == "term":
            return sub.payoff[node["leaf"]][(a, b)]
        seat = node["actor"]
        hole = a if seat == 0 else b
        row = _sigma_row(sigma, seat, hole, node)
        total = 0.0
        for k, action in enumerate(node["legal"]):
            if row[k] != 0.0:
                total += row[k] * ev(node["children"][action], a, b)
        return total

    return float(sum(w * ev(sub.root, a, b) for (a, b), w in sub.weight.items()))


def br_value(sub: Subgame, br_player: int, sigma_opp: Mapping[Infoset, np.ndarray]) -> float:
    """Best-response value for ``br_player`` against the fixed opponent ``sigma_opp``.

    Exact, by per-hole backward induction on the public tree (max at the BR
    player's nodes, expectation under ``sigma_opp`` at the opponent's).  Returned
    in ``br_player``'s own utility (seat 1 negates seat-0 payoffs).  At a Nash
    profile this equals the game value; ``br_value(0,·)+br_value(1,·)`` is the
    duality gap.
    """
    opp = 1 - br_player

    def rec(node, my_hole: int, opp_reach: Dict[int, float]) -> float:
        if node["type"] == "term":
            v = 0.0
            for ob, r in opp_reach.items():
                pair = (my_hole, ob) if br_player == 0 else (ob, my_hole)
                p0 = sub.payoff[node["leaf"]].get(pair)
                if p0 is not None:
                    v += r * (p0 if br_player == 0 else -p0)
            return v
        seat = node["actor"]
        legal = node["legal"]
        if seat == br_player:
            return max(rec(node["children"][a], my_hole, opp_reach) for a in legal)
        # Opponent node: each opponent hole plays its own strategy.
        v = 0.0
        for k, action in enumerate(legal):
            nxt: Dict[int, float] = {}
            for ob, r in opp_reach.items():
                s = _sigma_row(sigma_opp, seat, ob, node)[k]
                if r * s != 0.0:
                    nxt[ob] = nxt.get(ob, 0.0) + r * s
            if nxt:
                v += rec(node["children"][action], my_hole, nxt)
        return v

    total = 0.0
    for my_hole in sub.support[br_player]:
        opp_reach = {
            ob: sub.weight[(my_hole, ob) if br_player == 0 else (ob, my_hole)]
            for ob in sub.support[opp]
            if ((my_hole, ob) if br_player == 0 else (ob, my_hole)) in sub.weight
        }
        if opp_reach:
            total += rec(sub.root, my_hole, opp_reach)
    return float(total)


def exploitability(sub: Subgame, sigma: Mapping[Infoset, np.ndarray]) -> float:
    """Best-response gap of ``sigma``: ``BR_0(σ_1) + BR_1(σ_0)`` (≥ 0; 0 at Nash)."""
    return br_value(sub, 0, sigma) + br_value(sub, 1, sigma)
