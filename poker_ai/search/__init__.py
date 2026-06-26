"""Real-time depth-limited subgame search.

See :doc:`docs/subgame_solving` for the design.  Implemented so far: the
policy interface (:mod:`policy`), range tracking (:mod:`ranges`), the
subgame context / depth limits (:mod:`context`), continuation-value
evaluation (:mod:`leaf`), and the solver — shared state
(:mod:`solver_state`), the external-sampling MCCFR regime (:mod:`mccfr`),
the vector-form Linear CFR regime (:mod:`vector`, heads-up turn/river), the
``solve()`` orchestrator (:mod:`solver`), and ``SearchPolicy``.  Terminal
payouts (concrete, decision-free, and the vectorised range-vs-range
showdown) are all owned by the ``environment`` package and merely *called*
from here — see ``PokerEnv.payout`` / ``runout_equity`` / ``vector_payout``
and :mod:`environment.range_showdown`.
"""
