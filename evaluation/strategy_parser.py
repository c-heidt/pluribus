#!/usr/bin/env python3
"""
Strategy Analysis Script for Pluribus Poker AI

This script loads a trained strategy from an agent.joblib file and creates
a CSV file for manual analysis. It parses information set strings into
human-readable components and organizes the data for easy filtering.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import joblib
import pandas as pd


class StrategyParser:
    """Exports poker strategies to CSV format."""
    
    def __init__(self, agent_path: str, num_players: int = 2):
        """
        Initialize the analyzer with a trained agent.
        
        Parameters
        ----------
        agent_path : str
            Path to the agent.joblib file containing the trained strategy.
        num_players : int
            Number of players in the game (default: 2).
        """
        self.agent_path = agent_path
        self.num_players = num_players
        self.agent_data = self.load_agent()
        self.betting_stages = ["pre_flop", "flop", "turn", "river"]
        
    def load_agent(self) -> Dict[str, Any]:
        """Load the trained agent from joblib file."""
        if not os.path.exists(self.agent_path):
            raise FileNotFoundError(f"Agent file not found: {self.agent_path}")
        
        print(f"Loading agent from {self.agent_path}...")
        agent_data = joblib.load(self.agent_path)
        
        print(f"Agent loaded successfully!")
        print(f"Number of information sets: {len(agent_data.get('strategy', {}))}")
        return agent_data
    
    def parse_info_set(self, info_set_json: str) -> Dict[str, Any]:
        """
        Parse information set JSON string into readable components.
        
        Parameters
        ---------- 
        info_set_json : str
            JSON string containing the information set data.
            
        Returns
        -------
        parsed : Dict[str, Any]
            Dictionary with parsed components:
            - cards_cluster: Hand strength cluster
            - betting_stage: Current betting round
            - actions_taken: List of actions taken per stage
            - num_actions: Total number of actions
            - last_action: Most recent action taken
            - players_history: List of player IDs corresponding to each action
        """
        try:
            info_set_data = json.loads(info_set_json)
        except json.JSONDecodeError:
            return {
                "cards_cluster": "unknown",
                "betting_stage": "unknown", 
                "actions_taken": "parse_error",
                "num_actions": 0,
                "last_action": "unknown",
                "players_history": []
            }
        
        cards_cluster = info_set_data.get("cards_cluster", "unknown")
        history = info_set_data.get("history", [])
        
        # Extract betting stage and actions
        current_betting_stage = "pre_flop"  # Default
        all_actions = []
        actions_by_stage = {}
        
        for stage_dict in history:
            for stage, actions in stage_dict.items():
                current_betting_stage = stage
                actions_by_stage[stage] = actions
                all_actions.extend(actions)
        
        last_action = all_actions[-1] if all_actions else "none"
        
        return {
            "cards_cluster": cards_cluster,
            "betting_stage": current_betting_stage,
            "actions_taken": str(actions_by_stage),
            "num_actions": len(all_actions),
            "last_action": last_action
        }
    
    def extract_strategy_data(self) -> List[Dict[str, Any]]:
        """
        Extract and organize strategy data for CSV export.
        
        Returns
        -------
        data : List[Dict[str, Any]]
            List of dictionaries containing organized strategy data.
            Each row represents one information set with columns for each action's probability.
        """
        strategy = self.agent_data.get("strategy", {})
        data = []
        
        print("Processing information sets...")
        
        for info_set_json, action_probabilities in strategy.items():
            # Parse the information set
            parsed_info = self.parse_info_set(info_set_json)
            
            # Normalize probabilities if they are counts
            total_prob = sum(action_probabilities.values())
            if total_prob > 0:
                normalized_probs = {
                    action: prob / total_prob 
                    for action, prob in action_probabilities.items()
                }
            else:
                normalized_probs = action_probabilities
            
            # Create one row for this information set with all action probabilities
            row = {
                "info_set_raw": info_set_json,
                "cards_cluster": parsed_info["cards_cluster"],
                "betting_stage": parsed_info["betting_stage"],
                "actions_history": parsed_info["actions_taken"],
                "num_previous_actions": parsed_info["num_actions"],
                "last_action": parsed_info["last_action"]
            }
            
            # Add probability columns for each action (set to 0 if action not available)
            possible_actions = ["call", "fold", "raise"]  # Common poker actions
            for action in possible_actions:
                prob = normalized_probs.get(action, 0.0)
                row[f"{action}_probability"] = prob
            
            # Add any other actions that might exist in the strategy
            for action, prob in normalized_probs.items():
                if action not in possible_actions:
                    row[f"{action}_probability"] = prob
            
            data.append(row)
        
        return data

    def export_to_csv(self, output_path: str, include_pivot: bool = True):
        """
        Export strategy analysis to CSV file(s).
        
        Parameters
        ----------
        output_path : str
            Path for the output CSV file.
        include_pivot : bool
            Whether to also create a pivot table CSV.
        """
        # Extract strategy data
        data = self.extract_strategy_data()
        
        if not data:
            print("No strategy data found!")
            return
        
        # Create main detailed CSV
        df = pd.DataFrame(data)
        
        # Sort by betting stage, cards cluster
        stage_order = {stage: i for i, stage in enumerate(self.betting_stages)}
        df['stage_order'] = df['betting_stage'].map(stage_order)
        df = df.sort_values(['stage_order', 'cards_cluster', 'num_previous_actions'])
        df = df.drop('stage_order', axis=1)
        
        df.to_csv(output_path, index=False)
        print(f"Detailed strategy exported to {output_path}")
        print(f"Total information sets: {len(df)}")
        
        # Create pivot table if requested (now simpler since each row is an info set)
        if include_pivot:
            # Create a simplified pivot with just the action probabilities
            pivot_columns = ['betting_stage', 'cards_cluster', 'num_previous_actions']
            action_columns = [col for col in df.columns if col.endswith('_probability')]
            
            pivot_df = df[pivot_columns + action_columns].copy()
            pivot_path = output_path.replace('.csv', '_pivot.csv')
            pivot_df.to_csv(pivot_path, index=False)
            print(f"Simplified pivot table exported to {pivot_path}")
            print(f"Information sets: {len(pivot_df)}")
            print(f"Action probability columns: {action_columns}")
    
    def print_summary(self):
        """Print a summary of the strategy."""
        strategy = self.agent_data.get("strategy", {})
        
        print("\n" + "="*50)
        print("STRATEGY SUMMARY")
        print("="*50)
        
        # Count information sets by betting stage
        stage_counts = {stage: 0 for stage in self.betting_stages}
        total_info_sets = 0
        
        for info_set_json in strategy.keys():
            parsed = self.parse_info_set(info_set_json)
            stage = parsed["betting_stage"]
            if stage in stage_counts:
                stage_counts[stage] += 1
            total_info_sets += 1
        
        print(f"Total information sets: {total_info_sets}")
        print("\nInformation sets by betting stage:")
        for stage, count in stage_counts.items():
            percentage = (count / total_info_sets * 100) if total_info_sets > 0 else 0
            print(f"  {stage:10}: {count:6d} ({percentage:5.1f}%)")
        
        # Show training iteration
        timestep = self.agent_data.get("timestep", "unknown")
        print(f"\nTraining iteration: {timestep}")


def main():
    """Main function to run the strategy analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze poker strategy from trained agent and export to CSV"
    )
    parser.add_argument(
        "agent_path", 
        help="Path to the agent.joblib file"
    )
    parser.add_argument(
        "-o", "--output",
        default="strategy_analysis.csv",
        help="Output CSV file path (default: strategy_analysis.csv)"
    )
    parser.add_argument(
        "--no-pivot",
        action="store_true",
        help="Don't create pivot table CSV"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players in the game (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    try:
        analyzer = StrategyParser(args.agent_path, args.num_players)
        analyzer.print_summary()
        analyzer.export_to_csv(args.output, include_pivot=not args.no_pivot)
        
        print(f"\nAnalysis complete! Files created:")
        print(f"  - {args.output}")
        if not args.no_pivot:
            pivot_file = args.output.replace('.csv', '_pivot.csv')
            print(f"  - {pivot_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())