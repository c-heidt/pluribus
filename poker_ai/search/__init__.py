"""Real-time depth-limited subgame search.

See :doc:`docs/subgame_solving` for the design.  The policy interface
(:mod:`policy`), range tracking (:mod:`ranges`), subgame context / depth limits
(:mod:`context`), continuation-value evaluation (:mod:`leaf`), and the solver —
shared state (:mod:`solver_state`), the MCCFR and vector regimes, the ``solve()``
orchestrator and ``SearchPolicy``.

Terminal payouts are owned by the ``environment`` package and merely called from
here (``PokerEnv.payout`` / ``runout_equity`` / ``vector_payout``).
"""
