"""Exact reference CBV pass for OX-Search (Approach B, PO-CES-HU) — §11.3 step 10.

OX-Search's gadget (Ge et al., ICML 2024, §4.3) shifts every utility by the
opponent's **counterfactual best-response value against the bot playing the
blueprint**, ``CBV₁^σ(I₁)``, and compares the solved strategy's margin to it (the
adaptation-safety constraint, Eq 2).  This module computes that reference — the
"dominant new component" of B — for a heads-up turn/river subgame, **exactly**.

It is a best-response expectimax, not CFR: player 2 (the bot) is *fixed* to the
blueprint, so the game collapses to a single-maximiser problem for player 1 (the
opponent), whose value is obtained by one backward pass:

- **Opponent decision node** → per-combo **max** over actions.  Its infoset at the
  lossless root is its exact hole and the river is enumerated, so a per-combo max IS
  the exact lossless best response — the exploitability reference the safety guarantee
  is stated against.
- **Bot decision node** → blueprint-weighted sum, with σ (queried per LUT cluster,
  expanded to combos) folded into the counterfactual reach ``π_bot``.
- **Chance (the river on a turn root)** → **enumerated** and averaged in place, so the
  opponent's turn decision maxes over the river *expectation* rather than seeing it.
- **Terminal** → :meth:`PokerEnv.vector_payout`, the same settlement the vector regime
  uses, so ``CBV_ref`` is in the same counterfactual-value units as the solver's own
  node values — which is what lets the gadget's shift cancel in every interior regret
  delta (step 11).

Scope: heads-up **turn/river** only (the vector regime; §6.5).  These subgames run to
game end with terminal leaves only (:meth:`DepthLimit.classify` never returns
``"leaf"`` here), so the pass is exact.  Anchor = the blueprint on every solve (the
locked OX decision): ``CBV_ref`` is a property of the blueprint and the subgame
structure, independent of any earlier-street search refinement.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from poker_ai.search.cluster_maps import ClusterMapper
from poker_ai.search.context import SubgameContext
from poker_ai.search.solver_state import SolverConfig


def compute_cbv_ref(root_env, ctx: SubgameContext, cfg: SolverConfig) -> np.ndarray:
    """Per-combo ``CBV₁^σ`` for the opponent at the root of a HU turn/river subgame.

    Returns a ``(n_combos,)`` float64 vector aligned to ``root_env.combo_cards``: the
    opponent's exact best-response counterfactual value against the bot playing the
    blueprint, for each opponent root combo (board-infeasible combos are ``0``).  The
    bot is ``ctx.my_seat``; the opponent is the other live seat.  ``root_env`` is
    walked in place via make/undo and left pristine.
    """
    return _ReferenceWalk(root_env, ctx, cfg).run()


class _ReferenceWalk:
    """Backward-induction best-response walk (see the module docstring)."""

    def __init__(self, root_env, ctx: SubgameContext, cfg: SolverConfig) -> None:
        self.env = root_env
        self.ctx = ctx
        self.cfg = cfg

        live = sorted(ctx.ranges)
        if len(live) != 2:
            raise ValueError(
                f"reference pass requires exactly two live seats, got {live}."
            )
        self.bot = int(ctx.my_seat)
        if self.bot not in live:
            raise ValueError(
                f"reference pass: bot seat {self.bot} is not live ({live})."
            )
        self.opp = live[0] if live[1] == self.bot else live[1]

        # The bot plays the *unbiased* blueprint (BiasClass 'none'); the reference is
        # CBV against that fixed profile.  This is the same policy object the leaf
        # fleet and the belief update query, so the anchor is consistent everywhere.
        self.blueprint = ctx.leaf.policies["none"]

        self.n_combos = int(root_env.combo_cards.shape[0])
        self._combo_cards = root_env.combo_cards
        # Cards the two hands hold between them — the rivers an (opp, bot) combo pair
        # cancels out of the deck.  This is the chance node's divisor correction (see
        # :meth:`_integrate_river`); derived from the hand width rather than written
        # as a literal so it tracks the game, not this file.
        self._cards_in_play = 2 * int(root_env.combo_cards.shape[1])
        bc = np.asarray(ctx.board_compatible, dtype=np.float64)
        # Counterfactual weight carried down the tree = the bot's reach (its range,
        # board-masked; its blueprint σ is folded in at each bot node).  The opponent's
        # own reach is irrelevant to a counterfactual value, so it is not carried.
        self._bot_reach0 = np.asarray(ctx.ranges[self.bot], dtype=np.float64) * bc
        self._board_ok = bc

        self._street_root = int(ctx.street_at_root)
        self._depth = ctx.depth_limit
        self._cmaps = ClusterMapper(
            root_env.card_info_lut, root_env.combo_cards,
            root_env.community_cards, self._street_root,
        )
        self._avail = self._cmaps.avail
        self._n_completion = self._cmaps.n_completion   # 0 river / 1 turn
        if self._n_completion > 1:
            raise ValueError(
                "reference pass is turn/river only; a flop root (two chance nodes) "
                "is out of OX-Search-HU scope."
            )
        # The river overlay currently in force: () above the river chance node,
        # (river,) below it (river betting / river showdown).  Drives feasibility
        # masking and the terminal runout, exactly as the vector regime's sampled
        # completion does — the engine's own board deal is ignored.
        self._completion: Tuple[int, ...] = ()

    # ------------------------------------------------------------------ #
    def run(self) -> np.ndarray:
        cbv = self._walk(self.env, self._bot_reach0)
        # Board-infeasible root combos never reached a valid terminal; keep them 0.
        return cbv * self._board_ok

    # ------------------------------------------------------------------ #
    # Recursion (always entered on a non-terminal decision node).
    # ------------------------------------------------------------------ #
    def _walk(self, env, pi_bot: np.ndarray) -> np.ndarray:
        actor = env.player_i
        legal = tuple(a for a in env.legal_actions if a is not None)
        street = env.betting_round

        if actor == self.bot:
            # Bot node: fold the blueprint σ into the bot reach and SUM the children
            # (the counterfactual value averages over the bot's fixed play).
            sigma = self._blueprint_sigma(env, street, legal)
            v = np.zeros(self.n_combos, dtype=np.float64)
            for a_idx, action in enumerate(legal):
                token = env.step_in_place(action, settle_winners=False)
                v += self._child(env, pi_bot * sigma[:, a_idx], street)
                env.undo(token)
            return v

        # Opponent node: per-combo best response = max over actions.  Valid as the
        # exact lossless BR because the root infoset is per-combo and the river is
        # integrated below each action (so the max is over river-expected values).
        child_vs = np.empty((len(legal), self.n_combos), dtype=np.float64)
        for a_idx, action in enumerate(legal):
            token = env.step_in_place(action, settle_winners=False)
            child_vs[a_idx] = self._child(env, pi_bot, street)
            env.undo(token)
        return child_vs.max(axis=0)

    def _child(self, env, pi_bot: np.ndarray, parent_street: int) -> np.ndarray:
        if self._depth.classify(env) == "terminal":
            return self._terminal(env, pi_bot)
        if env.betting_round > parent_street:
            # Crossed the river-deal chance node: integrate the river EXACTLY here, so
            # the opponent's turn decision (above) maxes over the river expectation
            # rather than a clairvoyantly-known river.  Uniform over the available
            # deck (matching the vector regime's chance-sampling measure); per-combo
            # feasibility zeroes a combo's reach on the river it holds (card removal).
            return self._integrate_river(
                lambda: self._walk(
                    env, pi_bot * self._cmaps.feas(env.betting_round)
                )
            )
        return self._walk(env, pi_bot)

    def _terminal(self, env, pi_bot: np.ndarray) -> np.ndarray:
        s_bot, s_opp = self.bot, self.opp
        both_active = env.players[s_bot].is_active and env.players[s_opp].is_active
        if not both_active:
            # Fold: pot to the non-folder regardless of any future card, so the value
            # is river-independent.  Use whatever completion is in force (may be ()).
            runout = self._completion or None
            return env.vector_payout(s_opp, s_bot, pi_bot, runout=runout)

        # Showdown.
        if self._completion:
            # River already fixed (we are below the chance node): single settlement.
            feas_full = self._cmaps.feas_full
            reach = pi_bot if feas_full is None else pi_bot * feas_full
            return env.vector_payout(s_opp, s_bot, reach, runout=self._completion)
        if self._n_completion == 0:
            # River root: the board is already complete.
            return env.vector_payout(s_opp, s_bot, pi_bot, runout=None)

        # Turn all-in showdown reached before the chance node was walked: integrate the
        # river here (the same exact chance the mid-tree crossing does).
        def _one_river() -> np.ndarray:
            feas_full = self._cmaps.feas_full
            reach = pi_bot if feas_full is None else pi_bot * feas_full
            return env.vector_payout(s_opp, s_bot, reach, runout=self._completion)

        return self._integrate_river(_one_river)

    # ------------------------------------------------------------------ #
    def _integrate_river(self, fn) -> np.ndarray:
        """Average ``fn()`` over every available river, refreshing the overlay each.

        ``fn`` reads ``self._completion`` / the cluster maps for the current river.
        Per-combo feasibility (applied inside ``fn``) handles card removal, so a combo
        holding the dealt river contributes 0 for that river.  Restores the prior
        overlay on the way out so sibling branches see a clean state.

        The divisor is the number of rivers actually **dealable**, not ``len(avail)``.
        Both hands are known to the dealer, so for any (opp combo, bot combo) pair the
        four cards they hold are not in the deck: exactly ``len(avail) - 4`` of the
        summed terms are nonzero, and the other four were cancelled, not sampled.
        Dividing by ``len(avail)`` would shrink every value below the chance node by
        ``(len(avail)-4)/len(avail)`` while a turn-side FOLD — which sits above this
        node and takes no divisor at all — kept full weight.  That is the same
        fold-vs-showdown measure split already fixed in the vector regime and in the
        brute-force oracle: it does not cancel in a best response, because the max is
        taken across terminals on both sides of the chance node.  The count is exact:
        removal guarantees the two hands are disjoint, board-incompatible combos carry
        no reach, and ``avail`` already excludes the root community.
        """
        prev = self._completion
        acc = np.zeros(self.n_combos, dtype=np.float64)
        avail = self._avail
        for r in avail:
            self._completion = (int(r),)
            self._cmaps.refresh(self._completion)
            acc += fn()
        self._completion = prev
        if prev:
            self._cmaps.refresh(prev)   # restore the parent's overlay for later reads
        return acc / float(len(avail) - self._cards_in_play)

    # ------------------------------------------------------------------ #
    # Bot blueprint strategy, vectorised per combo (cluster query → combo expand).
    # ------------------------------------------------------------------ #
    def _blueprint_sigma(self, env, street: int, legal) -> np.ndarray:
        """``(n_combos, width)`` bot blueprint σ at this node, per bot combo.

        The blueprint is cluster-abstracted, so combos sharing a LUT cluster share the
        row: query :meth:`BlueprintPolicy.strategy` once per distinct cluster (via
        :meth:`PokerEnv.policy_state_for_cluster`, leak-free — no seat's real cards are
        read) and broadcast.  Root street rows are lossless (row == combo) but still
        info-set (cluster)-keyed; a future street gathers each combo's cluster row.
        Infeasible combos get a harmless uniform row (their bot reach is already 0).
        """
        width = len(legal)
        is_root = street == self._street_root
        public = env.policy_public_fields()
        sigma = np.full((self.n_combos, width), 1.0 / width, dtype=np.float64)

        if is_root:
            row_cluster = self._cmaps.root_cluster_of()      # per-combo raw LUT cluster
            feasible = row_cluster >= 0
            combos = np.flatnonzero(feasible)
            if combos.size == 0:
                return sigma
            vals = row_cluster[combos]
            uniq, first = np.unique(vals, return_index=True)
            for u, fi in zip(uniq, first):
                st = env.policy_state_for_cluster(int(u), public=public)
                row = np.asarray(self.blueprint.strategy(st), dtype=np.float64)
                members = combos[vals == u]
                sigma[members, : min(row.shape[0], width)] = row[:width]
            return sigma

        # Future (river) street: rows are clusters; query per row, gather to combos.
        cof = self._cmaps.cluster_of(street)                 # combo → dense row (-1 infeasible)
        gof = self._cmaps.gather_of(street)                  # dense row, infeasible → 0
        universe = self._cmaps.universe(street)              # dense row → raw LUT cluster
        n_rows = self._cmaps.n_rows(street)
        rows = np.full((n_rows, width), 1.0 / width, dtype=np.float64)
        present = np.unique(cof[cof >= 0])
        for r in present:
            st = env.policy_state_for_cluster(int(universe[int(r)]), public=public)
            row = np.asarray(self.blueprint.strategy(st), dtype=np.float64)
            rows[int(r), : min(row.shape[0], width)] = row[:width]
        return rows[gof]
