import argparse
import json
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(path: str) -> list:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {path}")
    return data

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare baseline and probe-based inference results across multiple benchmarks."
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        required=True,
        help="Directory containing JSONL evaluation results from the probe-based runs."
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Directory containing JSONL evaluation results from the baseline runs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="Directory to save the merged comparison data and plots."
    )
    return parser.parse_args()


def analyze_benchmark(probe_results_file: str, baseline_results_file: str):
    """
    Analyzes a single benchmark by comparing probe and baseline results.
    Returns a detailed comparison DataFrame and a summary analysis DataFrame.
    """
    # --- 1. Load Data ---
    print(f"Loading files:\n  Probe: {probe_results_file}\n  Baseline: {baseline_results_file}")
    try:
        probe_data = load_jsonl(probe_results_file)
        baseline_data = load_jsonl(baseline_results_file)
        print(f"Loaded {len(probe_data)} records from probe results.")
        print(f"Loaded {len(baseline_data)} records from baseline results.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Skipping this benchmark.")
        return None, None

    # --- 2. Index Baseline Data ---
    baseline_map = {item['data_index']: item for item in baseline_data if 'data_index' in item}
    print("Indexed baseline results by 'data_index'.")

    # --- 3. Merge Data ---
    print("Merging probe and baseline results...")
    comparison_data = []
    for probe_item in probe_data:
        data_index = probe_item.get('data_index')
        if data_index is None:
            continue

        baseline_item = baseline_map.get(data_index)
        if baseline_item:
            probe_correct = 1 if probe_item.get('xverify_evaluation', {}).get('Correctness') == 'Correct' else 0
            baseline_correct = 1 if baseline_item.get('xverify_evaluation', {}).get('Correctness') == 'Correct' else 0

            comparison_result = 'Tie'
            if probe_correct == 1 and baseline_correct == 0:
                comparison_result = 'Probe Wins'
            elif probe_correct == 0 and baseline_correct == 1:
                comparison_result = 'Baseline Wins'

            record = {
                'data_index': data_index,
                'problem': probe_item.get('problem', ''),
                'probe_strategy': probe_item.get('strategy', 'unknown'),
                'probe_correct': probe_correct,
                'probe_tokens': probe_item.get('total_generated_tokens', 0),
                'final_confidence': probe_item.get('final_confidence', 0.0),
                'baseline_correct': baseline_correct,
                'baseline_tokens': baseline_item.get('total_generated_tokens', 0),
                'comparison': comparison_result,
            }
            comparison_data.append(record)

    if not comparison_data:
        print("No matching data found between probe and baseline results based on 'data_index'. Skipping benchmark.")
        return None, None
        
    print(f"Successfully merged {len(comparison_data)} records.")
    
    df = pd.DataFrame(comparison_data)

    # --- 4. Analyze Data ---
    print("\n--- Analysis of Baseline Performance on Probe-Categorized Problems ---")

    # Group by the strategy assigned by the probe
    analysis = df.groupby('probe_strategy').agg(
        problem_count=('data_index', 'count'),
        probe_accuracy=('probe_correct', 'mean'),
        probe_avg_tokens=('probe_tokens', 'mean'),
        probe_median_tokens=('probe_tokens', 'median'),
        baseline_accuracy=('baseline_correct', 'mean'),
        baseline_avg_tokens=('baseline_tokens', 'mean'),
        baseline_median_tokens=('baseline_tokens', 'median'),
    ).reset_index()

    # Calculate win/loss counts and merge
    comparison_counts = pd.crosstab(df['probe_strategy'], df['comparison'])
    if not comparison_counts.empty:
        analysis = pd.merge(analysis, comparison_counts, on='probe_strategy', how='left').fillna(0)

    # Calculate additional comparison metrics
    analysis['accuracy_gain_pt'] = (analysis['probe_accuracy'] - analysis['baseline_accuracy']) * 100
    analysis['token_savings_avg'] = analysis['baseline_avg_tokens'] - analysis['probe_avg_tokens']
    # Handle potential division by zero if baseline tokens are 0
    analysis['token_savings_percent'] = (
        (analysis['token_savings_avg'] / analysis['baseline_avg_tokens'].replace(0, float('nan'))) * 100
    )

    # Formatting for better readability
    analysis['probe_accuracy'] = (analysis['probe_accuracy'] * 100).round(2)
    analysis['baseline_accuracy'] = (analysis['baseline_accuracy'] * 100).round(2)
    analysis['accuracy_gain_pt'] = analysis['accuracy_gain_pt'].round(2)
    analysis['probe_avg_tokens'] = analysis['probe_avg_tokens'].round(2)
    analysis['baseline_avg_tokens'] = analysis['baseline_avg_tokens'].round(2)
    analysis['token_savings_avg'] = analysis['token_savings_avg'].round(2)
    analysis['token_savings_percent'] = analysis['token_savings_percent'].round(2)

    # Reorder columns for logical presentation
    core_metrics = [
        'probe_strategy', 'problem_count',
        'probe_accuracy', 'baseline_accuracy', 'accuracy_gain_pt',
    ]
    # Add win/loss columns if they exist
    win_loss_cols = ['Probe Wins', 'Baseline Wins', 'Tie']
    for col in win_loss_cols:
        if col in analysis.columns:
            core_metrics.append(col)

    token_metrics = [
        'probe_avg_tokens', 'baseline_avg_tokens', 'token_savings_avg', 'token_savings_percent',
        'probe_median_tokens', 'baseline_median_tokens'
    ]
    
    # Ensure all expected columns exist before trying to reorder
    final_columns = core_metrics + [col for col in token_metrics if col in analysis.columns]
    existing_columns = [col for col in final_columns if col in analysis.columns]
    analysis = analysis[existing_columns]

    return df, analysis


def plot_results(summary_df: pd.DataFrame, output_dir: str):
    """
    Generates and saves plots based on the summary analysis DataFrame.
    """
    if summary_df.empty:
        print("Summary dataframe is empty, skipping plotting.")
        return

    print("\n--- Generating Plots ---")
    
    # Aggregate summary by benchmark
    benchmark_summary = summary_df.groupby('benchmark').agg({
        'accuracy_gain_pt': 'mean',
        'token_savings_percent': 'mean',
        'probe_accuracy': 'mean',
        'baseline_accuracy': 'mean'
    }).reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Accuracy Gain ---
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='benchmark', y='accuracy_gain_pt', data=benchmark_summary, ax=ax, palette='viridis', hue='benchmark', legend=False)
    ax.set_title('Average Accuracy Gain (Probe vs. Baseline) by Benchmark', fontsize=16)
    ax.set_ylabel('Accuracy Gain (Percentage Points)', fontsize=12)
    ax.set_xlabel('Benchmark', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'accuracy_gain_by_benchmark.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved accuracy gain plot to: {plot_path}")

    # --- Plot 2: Token Savings ---
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='benchmark', y='token_savings_percent', data=benchmark_summary, ax=ax, palette='plasma', hue='benchmark', legend=False)
    ax.set_title('Average Token Savings (Probe vs. Baseline) by Benchmark', fontsize=16)
    ax.set_ylabel('Token Savings (%)', fontsize=12)
    ax.set_xlabel('Benchmark', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'token_savings_by_benchmark.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved token savings plot to: {plot_path}")


def main():
    """
    Main function to load, merge, analyze, and save the results.
    """
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Find matching benchmark files ---
    baseline_files = sorted(glob.glob(os.path.join(args.baseline_dir, "*_evaluation_results.jsonl")))
    if not baseline_files:
        print(f"No '*_evaluation_results.jsonl' files found in baseline directory: {args.baseline_dir}")
        return
    
    print(f"Found {len(baseline_files)} potential baseline benchmarks.")

    all_detailed_dfs = []
    all_summary_dfs = []

    for baseline_file in baseline_files:
        filename = os.path.basename(baseline_file)
        probe_file = os.path.join(args.probe_dir, filename)

        if os.path.exists(probe_file):
            benchmark_name = filename.replace("_evaluation_results.jsonl", "")
            print(f"\n{'='*20} Analyzing Benchmark: {benchmark_name} {'='*20}")

            detailed_df, summary_df = analyze_benchmark(probe_file, baseline_file)

            if detailed_df is not None and summary_df is not None:
                detailed_df['benchmark'] = benchmark_name
                summary_df['benchmark'] = benchmark_name
                
                all_detailed_dfs.append(detailed_df)
                all_summary_dfs.append(summary_df)
                
                print(f"\nSummary for {benchmark_name}:")
                print(summary_df.to_string())
        else:
            print(f"Warning: Probe results file not found for {filename}, skipping.")
            print(f"  (Expected at: {probe_file})")

    if not all_summary_dfs:
        print("\nNo benchmarks were successfully analyzed. Exiting.")
        return

    # --- Aggregate and Save Results ---
    aggregated_detailed_df = pd.concat(all_detailed_dfs, ignore_index=True)
    aggregated_summary_df = pd.concat(all_summary_dfs, ignore_index=True)

    detailed_csv_path = os.path.join(args.output_dir, "all_benchmarks_detailed_comparison.csv")
    summary_csv_path = os.path.join(args.output_dir, "all_benchmarks_summary.csv")

    aggregated_detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8')
    aggregated_summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"\nSaved aggregated detailed results to: {detailed_csv_path}")
    print(f"Saved aggregated summary results to: {summary_csv_path}")

    # --- Overall Summary ---
    print("\n--- Overall Aggregated Summary (Grouped by Strategy) ---")
    overall_summary = aggregated_summary_df.groupby('probe_strategy').agg({
        'problem_count': 'sum',
        'probe_accuracy': 'mean',
        'baseline_accuracy': 'mean',
        'probe_avg_tokens': 'mean',
        'baseline_avg_tokens': 'mean',
        'accuracy_gain_pt': 'mean',
        'token_savings_percent': 'mean',
        'Probe Wins': 'sum',
        'Baseline Wins': 'sum',
        'Tie': 'sum'
    }).round(2)
    print(overall_summary.to_string())

    # --- Plotting ---
    plot_results(aggregated_summary_df, args.output_dir)
    # --- Reasoning Boundary Plotting ---
    plot_reasoning_boundaries(aggregated_detailed_df, args.output_dir)

def plot_reasoning_boundaries(detailed_df: pd.DataFrame, output_dir: str):
    """
    Generates and saves reasoning boundary scatter plots for each benchmark.
    """
    if detailed_df.empty:
        print("Detailed dataframe is empty, skipping reasoning boundary plots.")
        return

    benchmarks = detailed_df['benchmark'].unique()

    for benchmark in benchmarks:
        print(f"\n--- Generating Reasoning Boundary Plot for: {benchmark} ---")
        benchmark_df = detailed_df[detailed_df['benchmark'] == benchmark].copy()

        if 'final_confidence' not in benchmark_df.columns or 'baseline_tokens' not in benchmark_df.columns:
            print(f"Skipping {benchmark}: missing 'final_confidence' or 'baseline_tokens' columns.")
            continue

        # Create a numeric 'correctness' column for coloring
        benchmark_df['correctness'] = benchmark_df['probe_correct'].apply(lambda x: 'Correct' if x == 1 else 'Incorrect')

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define the "reasoning boundary" area
        too_hard_df = benchmark_df[benchmark_df['probe_strategy'] == 'too_hard']
        solvable_df = benchmark_df[benchmark_df['probe_strategy'] != 'too_hard']
        
        # Plot the area for problems deemed 'too_hard' (the red zone)
        if not too_hard_df.empty:
            sns.kdeplot(
                data=too_hard_df, x='final_confidence', y='baseline_tokens',
                fill=True, thresh=0.05, levels=5, cmap="Reds", ax=ax,
                alpha=0.3, label='Reasoning Boundary (Too Hard)'
            )

        # Plot the area for problems deemed solvable (the green zone)
        if not solvable_df.empty:
            sns.kdeplot(
                data=solvable_df, x='final_confidence', y='baseline_tokens',
                fill=True, thresh=0.05, levels=5, cmap="Greens", ax=ax,
                alpha=0.3, label='Reasoning Boundary (Solvable)'
            )
        
        # Scatter plot of all the individual problems
        sns.scatterplot(
            data=benchmark_df,
            x='final_confidence',
            y='baseline_tokens',
            hue='correctness',
            palette={'Correct': 'blue', 'Incorrect': 'red'},
            style='probe_strategy',
            markers={'normal': 'o', 'too_easy': 's', 'too_hard': 'X', 'unknown': 'D'},
            s=80,
            ax=ax
        )

        ax.set_title(f'Reasoning Boundary for {benchmark}', fontsize=16)
        ax.set_xlabel('Final Confidence', fontsize=12)
        ax.set_ylabel('Baseline Generated Tokens', fontsize=12)
        ax.legend(title='Probe Result')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'reasoning_boundary_{benchmark}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved reasoning boundary plot to: {plot_path}")

if __name__ == "__main__":
    main()
