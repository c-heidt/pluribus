"""Unit tests for the shared training primitives in ``poker_ai/ai/training.py``.

Covers:
- Sync-cycle schedule predicates (pure logic, table-driven).
- :class:`~poker_ai.blueprint.training.DiscountState` discount window and factor formula.
- :func:`~poker_ai.blueprint.training.cfr_step` pruning-dispatch decisions.
- :func:`~poker_ai.blueprint.training.strategy_step` delegation to ``update_strategy``.
- Bug regression checks for the historical bugs fixed during the refactor.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from poker_ai.blueprint import training
from poker_ai.blueprint.training import (
    DiscountState,
    at_sync_barrier,
    cfr_step,
    should_checkpoint,
    should_discount,
    should_update_strategy,
    strategy_step,
)
from poker_ai.blueprint.cfr import cfr, cfrp, merge_local_delta
from poker_ai.blueprint.strategy import update_strategy
from poker_ai.blueprint.tree_utils import calculate_strategy_from_row


# ---------------------------------------------------------------------------
# Schedule predicates
# ---------------------------------------------------------------------------


class TestSchedulePredicates:
    @pytest.mark.parametrize(
        "t, sync_interval, expected",
        [
            (1, 10, False),
            (9, 10, False),
            (10, 10, True),
            (20, 10, True),
            (25, 10, False),
            (100, 50, True),
            (1, 1, True),
        ],
    )
    def test_at_sync_barrier(self, t, sync_interval, expected):
        assert at_sync_barrier(t, sync_interval) is expected

    @pytest.mark.parametrize(
        "sync_step, strategy_interval, update_threshold, expected",
        [
            (3, 20, 4, False),
            (4, 20, 4, False),
            (5, 20, 4, False),
            (19, 20, 4, False),
            (20, 20, 4, True),
            (40, 20, 4, True),
            (10, 10, 0, True),
            (10, 3, 0, False),
        ],
    )
    def test_should_update_strategy(
        self, sync_step, strategy_interval, update_threshold, expected
    ):
        assert (
            should_update_strategy(sync_step, strategy_interval, update_threshold)
            is expected
        )

    @pytest.mark.parametrize(
        "sync_step, discount_interval, expected",
        [
            (1, 1, True),
            (2, 1, True),
            (1, 10, False),
            (10, 10, True),
            (20, 10, True),
            (15, 10, False),
        ],
    )
    def test_should_discount(self, sync_step, discount_interval, expected):
        assert should_discount(sync_step, discount_interval) is expected

    @pytest.mark.parametrize(
        "sync_step, checkpoint_interval, expected",
        [
            (1, 40, False),
            (40, 40, True),
            (80, 40, True),
            (41, 40, False),
        ],
    )
    def test_should_checkpoint(self, sync_step, checkpoint_interval, expected):
        assert should_checkpoint(sync_step, checkpoint_interval) is expected


# ---------------------------------------------------------------------------
# DiscountState
# ---------------------------------------------------------------------------


class TestDiscountState:
    def test_factor_formula(self):
        """Factor at sync_step *s* (interval 1) must be ``s / (s + 1)``."""
        ds = DiscountState(duration_cycles=100, discount_interval=1)
        tables = MagicMock()
        for s in [1, 2, 5, 10, 50]:
            tables.apply_discount.reset_mock()
            ds.apply(tables, sync_step=s)
            expected = s / (s + 1)
            tables.apply_discount.assert_called_once()
            called_factor = tables.apply_discount.call_args[0][0]
            assert called_factor == pytest.approx(expected)

    def test_factor_with_larger_interval(self):
        """With interval N, factor at sync_step s is (s // N) / (s // N + 1)."""
        ds = DiscountState(duration_cycles=1000, discount_interval=10)
        tables = MagicMock()
        ds.apply(tables, sync_step=10)
        assert tables.apply_discount.call_args[0][0] == pytest.approx(1 / 2)
        tables.apply_discount.reset_mock()
        ds.apply(tables, sync_step=20)
        assert tables.apply_discount.call_args[0][0] == pytest.approx(2 / 3)

    def test_window_closes_after_duration(self):
        """Once sync_step >= duration_cycles, window closes and is a no-op."""
        ds = DiscountState(duration_cycles=5, discount_interval=1)
        tables = MagicMock()
        ds.apply(tables, sync_step=4)
        assert ds.active
        assert tables.apply_discount.called

        tables.apply_discount.reset_mock()
        ds.apply(tables, sync_step=5)
        assert not ds.active
        tables.apply_discount.assert_not_called()

        tables.apply_discount.reset_mock()
        ds.apply(tables, sync_step=6)
        tables.apply_discount.assert_not_called()

    def test_inactive_state_is_noop(self):
        ds = DiscountState(duration_cycles=10, discount_interval=1)
        ds.active = False
        tables = MagicMock()
        ds.apply(tables, sync_step=3)
        tables.apply_discount.assert_not_called()

    def test_duration_cycles_property(self):
        ds = DiscountState(duration_cycles=42, discount_interval=1)
        assert ds.duration_cycles == 42


# ---------------------------------------------------------------------------
# cfr_step — pruning dispatch
# ---------------------------------------------------------------------------


class TestCfrStep:
    def test_below_prune_threshold_uses_cfr(self, monkeypatch):
        """Before the prune threshold, always use standard CFR."""
        cfr_mock = MagicMock()
        cfrp_mock = MagicMock()
        monkeypatch.setattr(training, "cfr", cfr_mock)
        monkeypatch.setattr(training, "cfrp", cfrp_mock)

        tables, state, local_delta = MagicMock(), MagicMock(), {}
        cfr_step(tables, state, i=0, t=10, prune_threshold=100, c=-300_000_000, local_delta=local_delta)
        cfr_mock.assert_called_once()
        cfrp_mock.assert_not_called()

    def test_past_prune_threshold_uses_cfrp_when_pruning(self, monkeypatch):
        """Past the threshold and with pruning draw < PRUNE_PROBABILITY, use CFR-P."""
        cfr_mock = MagicMock()
        cfrp_mock = MagicMock()
        monkeypatch.setattr(training, "cfr", cfr_mock)
        monkeypatch.setattr(training, "cfrp", cfrp_mock)
        monkeypatch.setattr(training.np.random, "uniform", lambda: 0.5)

        tables, state, local_delta = MagicMock(), MagicMock(), {}
        cfr_step(tables, state, i=0, t=101, prune_threshold=100, c=-300_000_000, local_delta=local_delta)
        cfrp_mock.assert_called_once()
        cfr_mock.assert_not_called()

    def test_past_prune_threshold_uses_cfr_on_5_percent_draw(self, monkeypatch):
        """5 % of the time past the threshold we still use standard CFR."""
        cfr_mock = MagicMock()
        cfrp_mock = MagicMock()
        monkeypatch.setattr(training, "cfr", cfr_mock)
        monkeypatch.setattr(training, "cfrp", cfrp_mock)
        monkeypatch.setattr(training.np.random, "uniform", lambda: 0.99)

        tables, state, local_delta = MagicMock(), MagicMock(), {}
        cfr_step(tables, state, i=0, t=101, prune_threshold=100, c=-300_000_000, local_delta=local_delta)
        cfr_mock.assert_called_once()
        cfrp_mock.assert_not_called()


# ---------------------------------------------------------------------------
# strategy_step
# ---------------------------------------------------------------------------


class TestStrategyStep:
    def test_delegates_to_update_strategy(self, monkeypatch):
        update_strategy_mock = MagicMock()
        monkeypatch.setattr(training, "update_strategy", update_strategy_mock)

        tables, state = MagicMock(), MagicMock()
        strategy_step(tables, state, i=2)
        update_strategy_mock.assert_called_once_with(tables, state, 2)


# ---------------------------------------------------------------------------
# Bug regressions
# ---------------------------------------------------------------------------


class TestBugRegressions:
    def test_cfr_signature_no_locks_param(self):
        assert "locks" not in inspect.signature(cfr).parameters

    def test_cfrp_signature_no_locks_param(self):
        assert "locks" not in inspect.signature(cfrp).parameters

    def test_cfr_has_local_delta_param(self):
        assert "local_delta" in inspect.signature(cfr).parameters

    def test_cfrp_has_local_delta_param(self):
        assert "local_delta" in inspect.signature(cfrp).parameters

    def test_update_strategy_has_no_locks_param(self):
        assert "locks" not in inspect.signature(update_strategy).parameters

    def test_calculate_strategy_from_row_importable(self):
        from poker_ai.blueprint.tree_utils import calculate_strategy_from_row as f
        assert callable(f)

    def test_merge_local_delta_importable(self):
        from poker_ai.blueprint.cfr import merge_local_delta as f
        assert callable(f)
