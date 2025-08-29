#!/usr/bin/env python3
"""
Oracle Strategy Selector

This script reads JSONL data containing multiple prompt strategies for each question
and selects the "oracle" strategy based on:
1. First priority: correct answers
2. Second priority: shortest think tokens

For each question, it extracts the difficulty label (easy/normal/hard) from the 
oracle strategy's prompt and creates a dataset with oracle labels.
"""

import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import pandas as pd


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
        print(f"Successfully loaded {len(data)} entries from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def extract_difficulty_from_strategy(strategy: str) -> str:
    """
    Extract difficulty level from strategy name.
    Assumes strategy contains 'easy', 'normal', or 'hard' keywords.
    """
    strategy_lower = strategy.lower()
    if 'easy' in strategy_lower:
        return 'easy'
    elif 'normal' in strategy_lower:
        return 'normal'
    elif 'hard' in strategy_lower:
        return 'hard'
    else:
        # If no clear difficulty indicator, return the strategy name
        return strategy


def is_correct_answer(entry: Dict[str, Any]) -> bool:
    """
    Determine if the answer is correct based on xverify_evaluation.
    Returns True if the answer is correct, False otherwise.
    """
    xverify = entry.get('xverify_evaluation', {})
    if isinstance(xverify, dict):
        correctness = xverify.get('Correctness', '').lower()
        return correctness == 'correct'
    return False


def get_think_tokens(entry: Dict[str, Any]) -> int:
    """Get the number of think tokens from an entry."""
    return entry.get('think_tokens', 0)


def group_by_question(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group entries by question text."""
    question_groups = defaultdict(list)
    for entry in data:
        question = entry.get('question', '')
        if question:
            question_groups[question].append(entry)
    return dict(question_groups)


def select_oracle_strategy(question_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the oracle strategy for a question based on:
    1. First priority: correct answers
    2. Second priority: shortest think tokens
    """
    if not question_entries:
        return None
    
    # Separate correct and incorrect answers
    correct_answers = [entry for entry in question_entries if is_correct_answer(entry)]
    incorrect_answers = [entry for entry in question_entries if not is_correct_answer(entry)]
    
    oracle_entry = None
    
    if correct_answers:
        # If there are correct answers, choose the one with shortest think tokens
        oracle_entry = min(correct_answers, key=get_think_tokens)
        print(f"  Selected correct answer with {get_think_tokens(oracle_entry)} think tokens")
    else:
        # If no correct answers, choose the one with shortest think tokens
        oracle_entry = min(question_entries, key=get_think_tokens)
        print(f"  No correct answers found, selected shortest think tokens: {get_think_tokens(oracle_entry)}")
    
    return oracle_entry


def process_oracle_selection(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process the data to select oracle strategies for each question.
    """
    question_groups = group_by_question(data)
    oracle_results = []
    
    print(f"\nProcessing {len(question_groups)} unique questions...")
    
    for question_idx, (question, entries) in enumerate(question_groups.items(), 1):
        print(f"\nQuestion {question_idx}/{len(question_groups)}: {question[:100]}...")
        print(f"  Found {len(entries)} strategy attempts")
        
        # Show strategy distribution
        strategy_counts = defaultdict(int)
        for entry in entries:
            strategy_counts[entry.get('strategy', 'unknown')] += 1
        print(f"  Strategy distribution: {dict(strategy_counts)}")
        
        # Select oracle strategy
        oracle_entry = select_oracle_strategy(entries)
        
        if oracle_entry:
            # Extract difficulty from oracle strategy
            oracle_strategy = oracle_entry.get('strategy', '')
            difficulty_label = extract_difficulty_from_strategy(oracle_strategy)
            
            # Create result entry with oracle information
            result_entry = {
                'question': question,
                'oracle_strategy': oracle_strategy,
                'difficulty_label': difficulty_label,
                'oracle_correct': is_correct_answer(oracle_entry),
                'oracle_think_tokens': get_think_tokens(oracle_entry),
                'oracle_final_answer': oracle_entry.get('final_answer', ''),
                'total_attempts': len(entries),
                'correct_attempts': sum(1 for e in entries if is_correct_answer(e)),
                # Include full oracle entry for reference
                'oracle_entry': oracle_entry
            }
            
            oracle_results.append(result_entry)
            print(f"  Oracle: {oracle_strategy} -> {difficulty_label} (correct: {result_entry['oracle_correct']})")
    
    return oracle_results


def save_results(oracle_results: List[Dict[str, Any]], output_file: str):
    """Save oracle selection results to file."""
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSONL
        if output_file.endswith('.jsonl'):
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in oracle_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"\nOracle results saved to {output_file}")
        
        # Also save a summary CSV
        csv_file = output_file.replace('.jsonl', '_summary.csv')
        summary_data = []
        for result in oracle_results:
            summary_data.append({
                'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                'oracle_strategy': result['oracle_strategy'],
                'difficulty_label': result['difficulty_label'],
                'oracle_correct': result['oracle_correct'],
                'oracle_think_tokens': result['oracle_think_tokens'],
                'total_attempts': result['total_attempts'],
                'correct_attempts': result['correct_attempts']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        print(f"Summary CSV saved to {csv_file}")
        
        # Print statistics
        print(f"\n=== Oracle Selection Statistics ===")
        print(f"Total questions processed: {len(oracle_results)}")
        print(f"Oracle strategies distribution:")
        strategy_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        correct_count = 0
        
        for result in oracle_results:
            strategy_counts[result['oracle_strategy']] += 1
            difficulty_counts[result['difficulty_label']] += 1
            if result['oracle_correct']:
                correct_count += 1
        
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}")
        
        print(f"\nDifficulty label distribution:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count}")
        
        print(f"\nOracle correctness: {correct_count}/{len(oracle_results)} ({100*correct_count/len(oracle_results):.1f}%)")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Select oracle strategies from multi-strategy JSONL data")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file containing multi-strategy data")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save oracle selection results (JSONL format)")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl_data(args.input_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Processing oracle selection...")
    oracle_results = process_oracle_selection(data)
    
    if oracle_results:
        save_results(oracle_results, args.output_file)
    else:
        print("No oracle results generated.")


if __name__ == "__main__":
    main()
