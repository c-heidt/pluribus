#!/usr/bin/env python3
"""
Quick Strategy Query Script

This script loads a strategy CSV created by strategy_analyzer.py and provides
quick filtering and analysis capabilities for manual strategy examination.
"""

import pandas as pd
import argparse
from typing import Optional, List


def load_strategy_csv(csv_path: str) -> pd.DataFrame:
    """Load the strategy CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded strategy with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run strategy_analyzer.py first to create the CSV file.")
        exit(1)


def filter_strategy(df: pd.DataFrame, 
                   betting_stage: Optional[str] = None,
                   min_action_probability: Optional[float] = None,
                   cards_cluster: Optional[int] = None,
                   action_type: Optional[str] = None) -> pd.DataFrame:
    """Filter the strategy DataFrame based on specified criteria."""
    filtered_df = df.copy()
    
    if betting_stage:
        filtered_df = filtered_df[filtered_df['betting_stage'] == betting_stage]
        print(f"Filtered by betting stage: {betting_stage}")
    
    if cards_cluster is not None:
        filtered_df = filtered_df[filtered_df['cards_cluster'] == cards_cluster]
        print(f"Filtered by cards cluster: {cards_cluster}")
    
    if action_type and min_action_probability is not None:
        prob_column = f"{action_type}_probability"
        if prob_column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[prob_column] >= min_action_probability]
            print(f"Filtered by {action_type} probability >= {min_action_probability}")
        else:
            print(f"Warning: {prob_column} column not found")
    
    return filtered_df


def analyze_action_frequencies(df: pd.DataFrame):
    """Analyze overall action frequencies."""
    print("\n" + "="*50)
    print("OVERALL ACTION FREQUENCIES")
    print("="*50)
    
    # Find action probability columns
    action_prob_columns = [col for col in df.columns if col.endswith('_probability')]
    
    for col in action_prob_columns:
        action_name = col.replace('_probability', '')
        avg_prob = df[col].mean()
        print(f"{action_name:15}: {avg_prob:.3f} ({avg_prob*100:.1f}%)")


def analyze_by_betting_stage(df: pd.DataFrame):
    """Analyze action frequencies by betting stage."""
    print("\n" + "="*50)
    print("ACTION FREQUENCIES BY BETTING STAGE")
    print("="*50)
    
    action_prob_columns = [col for col in df.columns if col.endswith('_probability')]
    
    for stage in ['pre_flop', 'flop', 'turn', 'river']:
        stage_df = df[df['betting_stage'] == stage]
        if len(stage_df) == 0:
            continue
            
        print(f"\n{stage.upper()}:")
        for col in action_prob_columns:
            action_name = col.replace('_probability', '')
            avg_prob = stage_df[col].mean()
            print(f"  {action_name:12}: {avg_prob:.3f} ({avg_prob*100:.1f}%)")


def find_high_probability_actions(df: pd.DataFrame, threshold: float = 0.8):
    """Find situations with very high probability actions."""
    print(f"\n" + "="*50)
    print(f"HIGH PROBABILITY ACTIONS (>{threshold*100:.0f}%)")
    print("="*50)
    
    action_prob_columns = [col for col in df.columns if col.endswith('_probability')]
    high_prob_situations = []
    
    for _, row in df.iterrows():
        for col in action_prob_columns:
            if row[col] > threshold:
                action_name = col.replace('_probability', '')
                high_prob_situations.append({
                    'betting_stage': row['betting_stage'],
                    'action': action_name,
                    'probability': row[col],
                    'cards_cluster': row['cards_cluster'],
                    'num_actions': row['num_previous_actions']
                })
    
    # Sort by probability
    high_prob_situations.sort(key=lambda x: x['probability'], reverse=True)
    
    print(f"Found {len(high_prob_situations)} high-probability situations:")
    for i, situation in enumerate(high_prob_situations[:10]):
        print(f"  {situation['betting_stage']:8} | {situation['action']:8} | "
              f"{situation['probability']:.3f} | Cluster {situation['cards_cluster']} | "
              f"{situation['num_actions']} actions")


def find_mixed_strategies(df: pd.DataFrame, min_actions: int = 2, min_prob: float = 0.1):
    """Find information sets with mixed strategies (multiple viable actions)."""
    print(f"\n" + "="*50)
    print(f"MIXED STRATEGIES (>= {min_actions} actions with prob > {min_prob})")
    print("="*50)
    
    action_prob_columns = [col for col in df.columns if col.endswith('_probability')]
    mixed_strategies = []
    
    for _, row in df.iterrows():
        significant_actions = []
        for col in action_prob_columns:
            if row[col] > min_prob:
                action_name = col.replace('_probability', '')
                significant_actions.append((action_name, row[col]))
        
        if len(significant_actions) >= min_actions:
            mixed_strategies.append((row, significant_actions))
    
    print(f"Found {len(mixed_strategies)} information sets with mixed strategies:")
    for i, (row, actions) in enumerate(mixed_strategies[:5]):  # Show first 5
        print(f"\n  {i+1}. {row['betting_stage']} | Cluster {row['cards_cluster']}:")
        for action_name, prob in actions:
            print(f"     {action_name:8}: {prob:.3f} ({prob*100:.1f}%)")


def interactive_query(df: pd.DataFrame):
    """Interactive query mode."""
    print("\n" + "="*50)
    print("INTERACTIVE QUERY MODE")
    print("="*50)
    print("Available commands:")
    print("  filter <betting_stage> [action_type] [min_prob] - Filter data")
    print("  show - Show current filtered data")
    print("  reset - Reset all filters")
    print("  summary - Show summary of current data")
    print("  quit - Exit interactive mode")
    print("\nExample: filter pre_flop call 0.5")
    
    current_df = df.copy()
    
    while True:
        cmd = input("\nQuery> ").strip().split()
        if not cmd:
            continue
            
        if cmd[0] == 'quit':
            break
        elif cmd[0] == 'reset':
            current_df = df.copy()
            print("Filters reset.")
        elif cmd[0] == 'show':
            # Show key columns in a readable format
            display_cols = ['betting_stage', 'cards_cluster', 'num_previous_actions']
            action_cols = [col for col in current_df.columns if col.endswith('_probability')]
            display_df = current_df[display_cols + action_cols].head(10)
            print(display_df.to_string())
            print(f"\nShowing 10 of {len(current_df)} information sets")
        elif cmd[0] == 'summary':
            analyze_action_frequencies(current_df)
        elif cmd[0] == 'filter':
            stage = cmd[1] if len(cmd) > 1 else None
            action_type = cmd[2] if len(cmd) > 2 else None
            min_prob = float(cmd[3]) if len(cmd) > 3 else None
            current_df = filter_strategy(current_df, stage, min_prob, None, action_type)
            print(f"Filtered data now has {len(current_df)} information sets")
        else:
            print("Unknown command. Type 'quit' to exit.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Query and analyze poker strategy from CSV"
    )
    parser.add_argument(
        "csv_path",
        help="Path to the strategy CSV file created by strategy_analyzer.py"
    )
    parser.add_argument(
        "--betting-stage", 
        choices=['pre_flop', 'flop', 'turn', 'river'],
        help="Filter by betting stage"
    )
    parser.add_argument(
        "--action-type",
        choices=['call', 'fold', 'raise'],
        help="Filter by action type (use with --min-prob)"
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        help="Filter by minimum probability for the specified action type"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive query mode"
    )
    
    args = parser.parse_args()
    
    # Load the strategy CSV
    df = load_strategy_csv(args.csv_path)
    
    # Apply initial filters if specified
    filtered_df = filter_strategy(df, args.betting_stage, args.min_prob, None, args.action_type)
    
    if args.interactive:
        interactive_query(filtered_df)
    else:
        # Run standard analysis
        analyze_action_frequencies(filtered_df)
        analyze_by_betting_stage(filtered_df)
        find_high_probability_actions(filtered_df)
        find_mixed_strategies(filtered_df)


if __name__ == "__main__":
    main()