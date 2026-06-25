"""Real-time depth-limited subgame search.

See :doc:`docs/subgame_solving` for the design.  Implemented so far: the
policy interface (:mod:`policy`), range tracking (:mod:`ranges`), the
subgame context / depth limits (:mod:`context`), continuation-value
evaluation (:mod:`leaf`), the vectorised showdown (:mod:`showdown`), and
the solver — shared state (:mod:`solver_state`), the MCCFR regime
(:mod:`mccfr`), the ``solve()`` orchestrator (:mod:`solver`), and
``SearchPolicy``.  The vector-form CFR regime (:mod:`vector`) is a
documented seam, still to be implemented.
"""
