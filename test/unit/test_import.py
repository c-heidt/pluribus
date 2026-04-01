def test_import():
    """Test the imports work"""
    import poker_ai
    from poker_ai import ai, environment
    from poker_ai.ai import runner
    from poker_ai.terminal import runner
    from poker_ai.environment import (
        PokerState, new_game, Card, Player, Pot, PokerEngine,
        PokerTable, Dealer, Deck, Evaluator,
    )
    from poker_ai.environment import actions, card, dealer, deck, engine, player
    from poker_ai.environment import table, evaluation
    from poker_ai.environment.evaluation import eval_card, evaluator, lookup
