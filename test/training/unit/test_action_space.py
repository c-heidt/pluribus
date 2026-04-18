"""Unit tests for ``poker_ai/ai/action_space.py``.

Verifies that the canonical action tables are internally consistent:
indices are dense, unique, and align across the three lookup structures.
"""

import pytest

from poker_ai.environment.action_space import (
    ACTION_TO_IDX,
    CANONICAL_ACTIONS,
    MAX_ACTIONS_PER_STREET,
)


class TestActionSpace:
    def test_all_four_streets_present(self):
        for r in range(4):
            assert r in CANONICAL_ACTIONS

    def test_action_to_idx_consistent(self):
        for r in range(4):
            for idx, action in enumerate(CANONICAL_ACTIONS[r]):
                assert ACTION_TO_IDX[r][action] == idx

    def test_max_actions_per_street_matches_length(self):
        for r in range(4):
            assert MAX_ACTIONS_PER_STREET[r] == len(CANONICAL_ACTIONS[r])

    def test_indices_are_dense(self):
        for r in range(4):
            indices = sorted(ACTION_TO_IDX[r].values())
            assert indices == list(range(len(indices)))

    def test_no_duplicate_actions(self):
        for r in range(4):
            actions = CANONICAL_ACTIONS[r]
            assert len(actions) == len(set(actions))
