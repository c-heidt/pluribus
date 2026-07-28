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


# =========================================================================== #
# Turn subgame oracle — river as an explicit, enumerated chance node (§6.5)
# =========================================================================== #
#
# A HU **turn** subgame is turn betting → **river chance** (uniform over the
# candidate non-board cards, per-combo card removal) → **per-river river betting**
# → showdown.  The river-conditioned vector regime must match the exact Nash of
# this game, so this oracle models it independently: river-stage infosets are keyed
# by the river card ``(seat, hole, public_key, river)`` (``river=None`` above the
# chance node), and river-side terminal payoffs are read from the engine with the
# river **forced** into the deck.  The chance measure matches the solver: weight
# ``1/R`` over every candidate river, a river conflicting with either hole giving
# zero (impossible deal) — so game value and exploitability are comparable.

TInfoset = Tuple[int, int, Tuple, "int | None"]  # (seat, hole, pk, river|None)


def _stage_river(env, river: int) -> None:
    """Force ``river`` to be the next community card the engine deals.

    The deck deals community cards from ``_cards[_idx]`` onward (``chance.Deck``);
    swapping ``river`` (which must be undealt) into position ``_idx`` makes the
    turn→river transition deal exactly ``river``.  Used on a fresh
    ``with_hole_cards`` copy when reading a river-side terminal's concrete payoff.
    """
    deck = env.deck
    idx = deck._idx
    pos = int(np.where(deck._cards == int(river))[0][0])
    if pos < idx:
        raise ValueError(f"river {river} already dealt (pos {pos} < idx {idx}).")
    deck._cards[idx], deck._cards[pos] = (
        int(deck._cards[pos]),
        int(deck._cards[idx]),
    )


@dataclass
class TurnSubgame:
    """Static tree + payoff tensors of one HU turn subgame (river enumerated).

    ``root`` adds a ``{"type": "chance", "child": ...}`` node at each turn→river
    crossing.  ``payoff[leaf]`` is keyed by ``(a, b)`` for a turn-side
    (river-independent) terminal and by ``(a, b, river)`` for a river-side one.
    """

    root: dict
    payoff: Dict[int, dict]
    support: Dict[int, List[int]]
    holes: Dict[int, Hole]
    weight: Dict[Pair, float]
    rivers: List[int]


def _turn_terminal_needs_river(env) -> bool:
    """A reached-directly turn terminal that the river still affects (showdown)."""
    return env.players[0].is_active and env.players[1].is_active


def build_turn_subgame(
    env,
    range0: Sequence[float],
    range1: Sequence[float],
    support0: Sequence[int],
    support1: Sequence[int],
) -> TurnSubgame:
    """Capture the turn public tree (with river chance nodes) + payoff tensors.

    ``env`` must be a heads-up **turn** state (4-card board, not terminal).  The
    betting tree is river-independent in structure, so it is walked once via
    make/undo; chance nodes are inserted where turn betting crosses into the river
    (a continue into river betting, or a turn all-in showdown).  Payoffs are read
    from concrete ``env.payout`` replays — with the river forced for river-side
    terminals.
    """
    street = int(env.betting_round)
    board = {int(c) for c in env.community_cards}
    rivers = sorted({int(x) for x in np.unique(env.combo_cards)} - board)

    holes: Dict[int, Hole] = {
        idx: tuple(int(x) for x in env.combo_cards[idx])
        for idx in set(support0) | set(support1)
    }

    leaves: List[Tuple[List[str], bool]] = []  # (line, river_side)

    def build(e, line: List[str], river_stage: bool) -> dict:
        if e.is_terminal:
            leaf_id = len(leaves)
            leaves.append((list(line), river_stage))
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
            wrap_chance = False
            child_stage = river_stage
            if not river_stage:
                if e.is_terminal:
                    if _turn_terminal_needs_river(e):
                        wrap_chance, child_stage = True, True
                elif e.betting_round > street:
                    wrap_chance, child_stage = True, True
            child = build(e, line + [a], child_stage)
            node["children"][a] = {"type": "chance", "child": child} if wrap_chance else child
            e.undo(token)
        return node

    root = build(env, [], False)

    # --- Chance measure: joint hole deal over card-disjoint support pairs. ---
    raw: Dict[Pair, float] = {}
    for a in support0:
        for b in support1:
            if set(holes[a]) & set(holes[b]):
                continue
            raw[(a, b)] = float(range0[a]) * float(range1[b])
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("turn subgame has no card-disjoint support pairs with mass.")
    weight = {pair: m / total for pair, m in raw.items()}

    # --- Payoff tensors: replay each line per pair (+ per river for river-side). ---
    payoff: Dict[int, dict] = {}
    for leaf_id, (line, river_side) in enumerate(leaves):
        row: dict = {}
        for (a, b) in raw:
            if river_side:
                for r in rivers:
                    if r in holes[a] or r in holes[b]:
                        continue  # impossible deal (card removal)
                    e2 = env.with_hole_cards([holes[a], holes[b]])
                    _stage_river(e2, r)
                    for action in line:
                        if e2.is_terminal:
                            break
                        e2.step_in_place(action)
                    row[(a, b, r)] = float(e2.payout[0])
            else:
                e2 = env.with_hole_cards([holes[a], holes[b]])
                for action in line:
                    if e2.is_terminal:
                        break
                    e2.step_in_place(action)
                row[(a, b)] = float(e2.payout[0])
        payoff[leaf_id] = row

    return TurnSubgame(
        root=root,
        payoff=payoff,
        support={0: list(support0), 1: list(support1)},
        holes=holes,
        weight=weight,
        rivers=rivers,
    )


class BruteForceTurnCFR:
    """Full-enumeration Linear CFR over a :class:`TurnSubgame` (river enumerated).

    Mirrors :class:`BruteForceCFR` but threads a ``river`` context: above the
    chance node it is ``None`` (river-independent turn betting); at a chance node
    it enumerates every candidate river (uniform ``1/R``, card removal) and recurses
    with that river fixed, so river-betting infosets and payoffs are per-river.
    """

    def __init__(self, subgame: TurnSubgame) -> None:
        self.sub = subgame
        self.regret: Dict[TInfoset, np.ndarray] = {}
        self.strat: Dict[TInfoset, np.ndarray] = {}

    def _row(self, table, key, width):
        row = table.get(key)
        if row is None:
            row = np.zeros(width, dtype=np.float64)
            table[key] = row
        return row

    def iterate(self, t: float) -> None:
        for (a, b), w in self.sub.weight.items():
            self._cfr(self.sub.root, a, b, 1.0, 1.0, w, t, None)

    def _cfr(self, node, a, b, r0, r1, w, t, river) -> float:
        ntype = node["type"]
        if ntype == "term":
            row = self.sub.payoff[node["leaf"]]
            return row[(a, b)] if river is None else row.get((a, b, river), 0.0)
        if ntype == "chance":
            rivers = self.sub.rivers
            inv = 1.0 / len(rivers)
            val = 0.0
            for r in rivers:
                if r in self.sub.holes[a] or r in self.sub.holes[b]:
                    continue  # impossible deal → contributes 0 (1/R convention)
                val += inv * self._cfr(node["child"], a, b, r0, r1, w * inv, t, int(r))
            return val
        seat = node["actor"]
        legal = node["legal"]
        width = len(legal)
        hole = a if seat == 0 else b
        key = (seat, hole, node["pk"], river)
        regret = self._row(self.regret, key, width)
        sigma = _regret_match(regret)
        util = np.empty(width, dtype=np.float64)
        node_util = 0.0
        for k, action in enumerate(legal):
            child = node["children"][action]
            if seat == 0:
                u = self._cfr(child, a, b, r0 * sigma[k], r1, w, t, river)
            else:
                u = self._cfr(child, a, b, r0, r1 * sigma[k], w, t, river)
            util[k] = u
            node_util += sigma[k] * u
        if seat == 0:
            cf_reach, own_reach, sign = w * r1, r0, 1.0
        else:
            cf_reach, own_reach, sign = w * r0, r1, -1.0
        regret += t * cf_reach * sign * (util - node_util)
        self._row(self.strat, key, width)[:] += t * own_reach * sigma
        return node_util

    def solve(self, iterations: int) -> Dict[TInfoset, np.ndarray]:
        for t in range(1, iterations + 1):
            self.iterate(float(t))
        out: Dict[TInfoset, np.ndarray] = {}
        for key, row in self.strat.items():
            total = row.sum()
            out[key] = row / total if total > 0.0 else np.full(len(row), 1.0 / len(row))
        return out


def _turn_sigma_row(sigma, seat, hole, node, river) -> np.ndarray:
    row = sigma.get((seat, hole, node["pk"], river))
    if row is None:
        n = len(node["legal"])
        return np.full(n, 1.0 / n)
    return row


def turn_game_value(sub: TurnSubgame, sigma: Mapping[TInfoset, np.ndarray]) -> float:
    """Expected value to seat 0 of ``sigma`` (river chance enumerated)."""

    def ev(node, a, b, river) -> float:
        ntype = node["type"]
        if ntype == "term":
            row = sub.payoff[node["leaf"]]
            return row[(a, b)] if river is None else row.get((a, b, river), 0.0)
        if ntype == "chance":
            inv = 1.0 / len(sub.rivers)
            v = 0.0
            for r in sub.rivers:
                if r in sub.holes[a] or r in sub.holes[b]:
                    continue
                v += inv * ev(node["child"], a, b, int(r))
            return v
        seat = node["actor"]
        hole = a if seat == 0 else b
        row = _turn_sigma_row(sigma, seat, hole, node, river)
        total = 0.0
        for k, action in enumerate(node["legal"]):
            if row[k] != 0.0:
                total += row[k] * ev(node["children"][action], a, b, river)
        return total

    return float(sum(w * ev(sub.root, a, b, None) for (a, b), w in sub.weight.items()))


def turn_br_value(sub: TurnSubgame, br_player: int, sigma_opp: Mapping[TInfoset, np.ndarray]) -> float:
    """Exact best-response value for ``br_player`` (river chance enumerated)."""
    opp = 1 - br_player

    def rec(node, my_hole, opp_reach, river) -> float:
        ntype = node["type"]
        if ntype == "term":
            row = sub.payoff[node["leaf"]]
            v = 0.0
            for ob, r in opp_reach.items():
                if br_player == 0:
                    key = (my_hole, ob) if river is None else (my_hole, ob, river)
                else:
                    key = (ob, my_hole) if river is None else (ob, my_hole, river)
                p0 = row.get(key)
                if p0 is not None:
                    v += r * (p0 if br_player == 0 else -p0)
            return v
        if ntype == "chance":
            inv = 1.0 / len(sub.rivers)
            v = 0.0
            for rr in sub.rivers:
                if rr in sub.holes[my_hole]:
                    continue  # my hand cannot see its own card as the river
                sub_reach = {ob: rv for ob, rv in opp_reach.items() if rr not in sub.holes[ob]}
                if sub_reach:
                    v += inv * rec(node["child"], my_hole, sub_reach, int(rr))
            return v
        seat = node["actor"]
        legal = node["legal"]
        if seat == br_player:
            return max(rec(node["children"][a], my_hole, opp_reach, river) for a in legal)
        v = 0.0
        for k, action in enumerate(legal):
            nxt: Dict[int, float] = {}
            for ob, rv in opp_reach.items():
                s = _turn_sigma_row(sigma_opp, seat, ob, node, river)[k]
                if rv * s != 0.0:
                    nxt[ob] = nxt.get(ob, 0.0) + rv * s
            if nxt:
                v += rec(node["children"][action], my_hole, nxt, river)
        return v

    total = 0.0
    for my_hole in sub.support[br_player]:
        opp_reach = {
            ob: sub.weight[(my_hole, ob) if br_player == 0 else (ob, my_hole)]
            for ob in sub.support[opp]
            if ((my_hole, ob) if br_player == 0 else (ob, my_hole)) in sub.weight
        }
        if opp_reach:
            total += rec(sub.root, my_hole, opp_reach, None)
    return float(total)


def turn_exploitability(sub: TurnSubgame, sigma: Mapping[TInfoset, np.ndarray]) -> float:
    """Best-response gap of ``sigma`` for the turn subgame (≥ 0; 0 at Nash)."""
    return turn_br_value(sub, 0, sigma) + turn_br_value(sub, 1, sigma)


# =========================================================================== #
# Flop subgame oracle — turn AND river as nested, enumerated chance nodes (§6.5)
# =========================================================================== #
#
# A HU **flop** subgame is flop betting → **turn chance** → per-turn turn betting
# → **river chance** → per-(turn, river) river betting → showdown.  Both solver paths
# for a flop root — production MCCFR (sampled) and the vector regime driven directly
# (full-width, cluster-keyed future streets) — must match the Nash of this
# two-chance-level game, so this oracle generalises
# the turn oracle: infosets and payoffs are keyed by a **runout tuple** that grows
# as chance nodes are crossed — ``()`` in flop betting, ``(turn,)`` in turn
# betting, ``(turn, river)`` in river betting.  Each chance node deals the next
# board card uniformly over the still-undealt candidates (``1/N`` at that level,
# a card conflicting with either hole contributing 0 — impossible deal), exactly
# the measure the solver samples (turn without replacement, then river).  An
# all-in showdown that ends betting before the river force-deals the remaining
# board: it is modelled as the corresponding **stack** of chance nodes above a
# full-depth leaf, so its value integrates over the same runout distribution.

FInfoset = Tuple[int, int, Tuple, Tuple[int, ...]]  # (seat, hole, pk, runout)


def _stage_runout(env, runout: Sequence[int]) -> None:
    """Force ``runout`` (turn[, river]) to be the next community cards dealt.

    Generalises :func:`_stage_river` to a multi-card completion: reorders the
    deck's still-undealt tail so ``runout`` comes first (in board order), leaving
    the remaining undealt cards after it (their order is immaterial — only the
    first ``len(runout)`` are turned over on the way to a five-card board).  Used
    on a fresh ``with_hole_cards`` copy when reading a turn-/river-side terminal.
    """
    deck = env.deck
    idx = int(deck._idx)
    want = [int(c) for c in runout]
    undealt = [int(c) for c in deck._cards[idx:]]
    for c in want:
        if c not in undealt:
            raise ValueError(f"runout card {c} already dealt.")
    rest = [c for c in undealt if c not in want]
    deck._cards[idx:] = np.array(want + rest, dtype=deck._cards.dtype)


@dataclass
class FlopSubgame:
    """Static tree + payoff tensors of one HU flop subgame (turn+river enumerated).

    ``root`` carries ``{"type": "chance", "child": ...}`` nodes at each board
    crossing (a betting continuation into the next street, or an all-in showdown
    that force-deals it — the latter a stack of chance nodes).  ``payoff[leaf]``
    is keyed by ``(a, b) + runout`` where ``runout`` has as many cards as chance
    nodes crossed to reach the leaf: ``(a, b)`` in flop betting, ``(a, b, turn)``
    in turn betting, ``(a, b, turn, river)`` in river betting.
    """

    root: dict
    payoff: Dict[int, dict]
    support: Dict[int, List[int]]
    holes: Dict[int, Hole]
    weight: Dict[Pair, float]
    avail: List[int]   # candidate board cards (deck minus the flop board)


def build_flop_subgame(
    env,
    range0: Sequence[float],
    range1: Sequence[float],
    support0: Sequence[int],
    support1: Sequence[int],
) -> FlopSubgame:
    """Capture the flop public tree (with turn+river chance nodes) + payoff tensors.

    ``env`` must be a heads-up **flop** state (3-card board, not terminal).  The
    betting tree is runout-independent in structure, so it is walked once via
    make/undo; a chance node is inserted wherever the board grows — either betting
    continuing into the next street, or an all-in showdown that force-deals the
    rest of the board (a stack of ``2 - depth`` chance nodes above a full-depth
    leaf).  Payoffs are read from concrete ``env.payout`` replays with the runout
    forced for turn-/river-side terminals.
    """
    street = int(env.betting_round)
    board = {int(c) for c in env.community_cards}
    avail = sorted({int(x) for x in np.unique(env.combo_cards)} - board)

    holes: Dict[int, Hole] = {
        idx: tuple(int(x) for x in env.combo_cards[idx])
        for idx in set(support0) | set(support1)
    }

    leaves: List[Tuple[List[str], int]] = []  # (line, depth ∈ {0, 1, 2})

    def build(e, line: List[str], depth: int) -> dict:
        if e.is_terminal:
            leaf_id = len(leaves)
            leaves.append((list(line), depth))
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
            n_wrap, child_depth = 0, depth
            if e.is_terminal:
                # An all-in showdown ends betting before the river: the engine
                # force-deals the rest of the board, so its value still depends on
                # the whole runout — model that as 2−depth stacked chance nodes and
                # key the leaf at full depth.  A fold is board-independent (keyed at
                # the current depth), so no wrap.
                if e.players[0].is_active and e.players[1].is_active and depth < 2:
                    n_wrap, child_depth = 2 - depth, 2
            elif e.betting_round > street + depth:
                # Betting crossed into the next street: one card was dealt.
                n_wrap, child_depth = 1, depth + 1
            child = build(e, line + [a], child_depth)
            for _ in range(n_wrap):
                child = {"type": "chance", "child": child}
            node["children"][a] = child
            e.undo(token)
        return node

    root = build(env, [], 0)

    # --- Chance measure: joint hole deal over card-disjoint support pairs. ---
    raw: Dict[Pair, float] = {}
    for a in support0:
        for b in support1:
            if set(holes[a]) & set(holes[b]):
                continue
            raw[(a, b)] = float(range0[a]) * float(range1[b])
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("flop subgame has no card-disjoint support pairs with mass.")
    weight = {pair: m / total for pair, m in raw.items()}

    # --- Payoff tensors: replay each line per pair (+ per runout for depth ≥ 1). --
    def _runouts_for(depth: int, a: int, b: int):
        """Ordered runouts of length ``depth`` from ``avail``, card-removal applied."""
        if depth == 0:
            yield ()
            return
        for turn in avail:
            if turn in holes[a] or turn in holes[b]:
                continue
            if depth == 1:
                yield (turn,)
                continue
            for river in avail:
                if river == turn or river in holes[a] or river in holes[b]:
                    continue
                yield (turn, river)

    payoff: Dict[int, dict] = {}
    for leaf_id, (line, depth) in enumerate(leaves):
        row: dict = {}
        for (a, b) in raw:
            for runout in _runouts_for(depth, a, b):
                e2 = env.with_hole_cards([holes[a], holes[b]])
                if runout:
                    _stage_runout(e2, runout)
                for action in line:
                    if e2.is_terminal:
                        break
                    e2.step_in_place(action)
                row[(a, b) + runout] = float(e2.payout[0])
        payoff[leaf_id] = row

    return FlopSubgame(
        root=root,
        payoff=payoff,
        support={0: list(support0), 1: list(support1)},
        holes=holes,
        weight=weight,
        avail=list(avail),
    )


class BruteForceFlopCFR:
    """Full-enumeration Linear CFR over a :class:`FlopSubgame` (turn+river enumerated).

    Mirrors :class:`BruteForceTurnCFR` but threads a **runout tuple** through the
    two chance levels: ``()`` above the turn chance, ``(turn,)`` above the river
    chance, ``(turn, river)`` below.  Each chance node deals the next candidate
    card uniformly (``1/N`` at that level; a card in either hole contributes 0 —
    card removal), so betting infosets and payoffs are per-runout.
    """

    def __init__(self, subgame: FlopSubgame) -> None:
        self.sub = subgame
        self.regret: Dict[FInfoset, np.ndarray] = {}
        self.strat: Dict[FInfoset, np.ndarray] = {}

    def _row(self, table, key, width):
        row = table.get(key)
        if row is None:
            row = np.zeros(width, dtype=np.float64)
            table[key] = row
        return row

    def iterate(self, t: float) -> None:
        for (a, b), w in self.sub.weight.items():
            self._cfr(self.sub.root, a, b, 1.0, 1.0, w, t, ())

    def _cfr(self, node, a, b, r0, r1, w, t, runout: Tuple[int, ...]) -> float:
        ntype = node["type"]
        if ntype == "term":
            return self.sub.payoff[node["leaf"]].get((a, b) + runout, 0.0)
        if ntype == "chance":
            cands = [c for c in self.sub.avail if c not in runout]
            inv = 1.0 / len(cands)
            val = 0.0
            for c in cands:
                if c in self.sub.holes[a] or c in self.sub.holes[b]:
                    continue  # impossible deal → contributes 0 (1/N convention)
                val += inv * self._cfr(
                    node["child"], a, b, r0, r1, w * inv, t, runout + (int(c),)
                )
            return val
        seat = node["actor"]
        legal = node["legal"]
        width = len(legal)
        hole = a if seat == 0 else b
        key = (seat, hole, node["pk"], runout)
        regret = self._row(self.regret, key, width)
        sigma = _regret_match(regret)
        util = np.empty(width, dtype=np.float64)
        node_util = 0.0
        for k, action in enumerate(legal):
            child = node["children"][action]
            if seat == 0:
                u = self._cfr(child, a, b, r0 * sigma[k], r1, w, t, runout)
            else:
                u = self._cfr(child, a, b, r0, r1 * sigma[k], w, t, runout)
            util[k] = u
            node_util += sigma[k] * u
        if seat == 0:
            cf_reach, own_reach, sign = w * r1, r0, 1.0
        else:
            cf_reach, own_reach, sign = w * r0, r1, -1.0
        regret += t * cf_reach * sign * (util - node_util)
        self._row(self.strat, key, width)[:] += t * own_reach * sigma
        return node_util

    def solve(self, iterations: int) -> Dict[FInfoset, np.ndarray]:
        for t in range(1, iterations + 1):
            self.iterate(float(t))
        out: Dict[FInfoset, np.ndarray] = {}
        for key, row in self.strat.items():
            total = row.sum()
            out[key] = row / total if total > 0.0 else np.full(len(row), 1.0 / len(row))
        return out


def _flop_sigma_row(sigma, seat, hole, node, runout) -> np.ndarray:
    row = sigma.get((seat, hole, node["pk"], runout))
    if row is None:
        n = len(node["legal"])
        return np.full(n, 1.0 / n)
    return row


def flop_game_value(sub: FlopSubgame, sigma: Mapping[FInfoset, np.ndarray]) -> float:
    """Expected value to seat 0 of ``sigma`` (turn+river chance enumerated)."""

    def ev(node, a, b, runout) -> float:
        ntype = node["type"]
        if ntype == "term":
            return sub.payoff[node["leaf"]].get((a, b) + runout, 0.0)
        if ntype == "chance":
            cands = [c for c in sub.avail if c not in runout]
            inv = 1.0 / len(cands)
            v = 0.0
            for c in cands:
                if c in sub.holes[a] or c in sub.holes[b]:
                    continue
                v += inv * ev(node["child"], a, b, runout + (int(c),))
            return v
        seat = node["actor"]
        hole = a if seat == 0 else b
        row = _flop_sigma_row(sigma, seat, hole, node, runout)
        total = 0.0
        for k, action in enumerate(node["legal"]):
            if row[k] != 0.0:
                total += row[k] * ev(node["children"][action], a, b, runout)
        return total

    return float(sum(w * ev(sub.root, a, b, ()) for (a, b), w in sub.weight.items()))


def flop_br_value(sub: FlopSubgame, br_player: int, sigma_opp: Mapping[FInfoset, np.ndarray]) -> float:
    """Exact best-response value for ``br_player`` (turn+river chance enumerated)."""
    opp = 1 - br_player

    def rec(node, my_hole, opp_reach, runout) -> float:
        ntype = node["type"]
        if ntype == "term":
            v = 0.0
            for ob, r in opp_reach.items():
                pair = (my_hole, ob) if br_player == 0 else (ob, my_hole)
                p0 = sub.payoff[node["leaf"]].get(pair + runout)
                if p0 is not None:
                    v += r * (p0 if br_player == 0 else -p0)
            return v
        if ntype == "chance":
            cands = [c for c in sub.avail if c not in runout]
            inv = 1.0 / len(cands)
            v = 0.0
            for c in cands:
                if c in sub.holes[my_hole]:
                    continue  # my hand cannot see its own card on the board
                sub_reach = {ob: rv for ob, rv in opp_reach.items() if c not in sub.holes[ob]}
                if sub_reach:
                    v += inv * rec(node["child"], my_hole, sub_reach, runout + (int(c),))
            return v
        seat = node["actor"]
        legal = node["legal"]
        if seat == br_player:
            return max(rec(node["children"][a], my_hole, opp_reach, runout) for a in legal)
        v = 0.0
        for k, action in enumerate(legal):
            nxt: Dict[int, float] = {}
            for ob, rv in opp_reach.items():
                s = _flop_sigma_row(sigma_opp, seat, ob, node, runout)[k]
                if rv * s != 0.0:
                    nxt[ob] = nxt.get(ob, 0.0) + rv * s
            if nxt:
                v += rec(node["children"][action], my_hole, nxt, runout)
        return v

    total = 0.0
    for my_hole in sub.support[br_player]:
        opp_reach = {
            ob: sub.weight[(my_hole, ob) if br_player == 0 else (ob, my_hole)]
            for ob in sub.support[opp]
            if ((my_hole, ob) if br_player == 0 else (ob, my_hole)) in sub.weight
        }
        if opp_reach:
            total += rec(sub.root, my_hole, opp_reach, ())
    return float(total)


def flop_exploitability(sub: FlopSubgame, sigma: Mapping[FInfoset, np.ndarray]) -> float:
    """Best-response gap of ``sigma`` for the flop subgame (≥ 0; 0 at Nash)."""
    return flop_br_value(sub, 0, sigma) + flop_br_value(sub, 1, sigma)
