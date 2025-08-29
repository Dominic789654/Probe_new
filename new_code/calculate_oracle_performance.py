import json
import argparse
from collections import defaultdict
import os
import pandas as pd

def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Calculate Oracle performance from full baseline evaluation results.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the JSONL file containing evaluation results from a full baseline run.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the Oracle selection results as a JSONL file.")
    return parser.parse_args()

def load_jsonl(path):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Saves a list of dictionaries to a JSONL file."""
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    """Main function to calculate and report Oracle performance."""
    args = parse_args()

    # --- Step 1: Load data and group by problem ---
    print(f"Loading evaluation results from: {args.input_file}")
    eval_results = load_jsonl(args.input_file)
    
    problems = defaultdict(list)
    for result in eval_results:
        problems[result['problem']].append(result)
    
    print(f"Found {len(problems)} unique problems from {len(eval_results)} total runs.")

    # --- Step 2: Select the Oracle run for each problem ---
    oracle_selections = []
    for problem, runs in problems.items():
        correct_runs = [
            r for r in runs if r.get('xverify_evaluation', {}).get('Correctness') == 'Correct'
        ]

        if correct_runs:
            # If there are correct runs, choose the one with the minimum tokens.
            best_run = min(correct_runs, key=lambda r: r.get('total_generated_tokens', float('inf')))
        else:
            # If no run is correct, the problem is considered a failure for the Oracle.
            # We still select the one with the minimum tokens for statistical purposes.
            best_run = min(runs, key=lambda r: r.get('total_generated_tokens', float('inf')))
        
        oracle_selections.append(best_run)

    # --- Step 3: Calculate and print statistics ---
    if not oracle_selections:
        print("No data to process. Exiting.")
        return

    num_problems = len(problems)
    num_correct = sum(
        1 for r in oracle_selections if r.get('xverify_evaluation', {}).get('Correctness') == 'Correct'
    )
    total_tokens = sum(r.get('total_generated_tokens', 0) for r in oracle_selections)
    
    accuracy = (num_correct / num_problems) * 100 if num_problems > 0 else 0
    avg_tokens = total_tokens / num_problems if num_problems > 0 else 0
    
    strategy_counts = defaultdict(int)
    for r in oracle_selections:
        strategy_counts[r['strategy']] += 1

    print("\n--- Oracle Performance Statistics ---")
    print(f"Total Unique Problems: {num_problems}")
    print(f"Problems Solved (Oracle): {num_correct}")
    print(f"Oracle Accuracy (Pass@1): {accuracy:.2f}%")
    print(f"Oracle Average Tokens per Problem: {avg_tokens:.2f}")
    
    strategy_stats = defaultdict(lambda: {'count': 0, 'total_tokens': 0, 'correct_count': 0})
    for r in oracle_selections:
        strategy = r['strategy']
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['total_tokens'] += r.get('total_generated_tokens', 0)
        if r.get('xverify_evaluation', {}).get('Correctness') == 'Correct':
            strategy_stats[strategy]['correct_count'] += 1

    print("\n--- Oracle Strategy Distribution & Performance ---")
    for strategy, stats in sorted(strategy_stats.items()):
        count = stats['count']
        percentage_dist = (count / num_problems) * 100 if num_problems > 0 else 0
        
        correct_count = stats['correct_count']
        success_rate = (correct_count / count) * 100 if count > 0 else 0
        
        total_tokens = stats['total_tokens']
        avg_tokens_strat = total_tokens / count if count > 0 else 0
        
        print(f"  - Strategy: {strategy}")
        print(f"    - Chosen for: {count} problems ({percentage_dist:.2f}%)")
        print(f"    - Success Rate: {success_rate:.2f}% ({correct_count}/{count} correct)")
        print(f"    - Avg Tokens: {avg_tokens_strat:.2f}")
    print("--------------------------------------------------\n")

    # --- Step 4: Save the Oracle selections to a new file ---
    save_jsonl(oracle_selections, args.output_file)
    print(f"Saved {len(oracle_selections)} Oracle selections to: {args.output_file}")


if __name__ == "__main__":
    main()
