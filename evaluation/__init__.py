"""Offline evaluation harness for the real-time-search agent (docs/evaluation.md).

This top-level package holds the *measurement* side of the search stack — the
machine-readable logging sink (:mod:`evaluation.sqlite_logging`), and later the
time-budgeted runner (§10.1) and the end-of-run summary (§8).  It is deliberately
kept out of :mod:`poker_ai.search`: the sink has no search-package dependency
(§9.2), so it can be imported and tested on its own and run standalone against any
past snapshot.
"""
