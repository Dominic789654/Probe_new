import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

def create_comparison_plot(grouped_data, plot_path):
    """
    Creates and saves a comparison plot using seaborn's catplot.
    """
    if not ('matplotlib' in sys.modules and 'seaborn' in sys.modules):
        print("\nWarning: Plotting libraries (matplotlib, seaborn) not found. Skipping plot generation.")
        print("Please install them to enable this feature: pip install matplotlib seaborn pandas")
        return

    if grouped_data.empty:
        print("No data available for plotting.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # Select key metrics for visualization
    metrics_to_plot = [
        'correctness_rate',
        'reasoning_similarity',
        'average_entropy',
        'final_answer_entropy',
        'avg_gen_tokens'
    ]
    
    # Prettify metric names for plot titles
    metric_names = {
        'correctness_rate': 'Correctness Rate',
        'reasoning_similarity': 'Reasoning Similarity',
        'average_entropy': 'Average Generation Entropy',
        'final_answer_entropy': 'Final Answer Entropy',
        'avg_gen_tokens': 'Average Generated Tokens'
    }

    # Filter out metrics that might not be in the dataframe
    metrics_to_plot = [m for m in metrics_to_plot if m in grouped_data.columns]
    if not metrics_to_plot:
        print("None of the specified metrics for plotting are present in the data. Skipping plot generation.")
        return

    plot_df = pd.melt(grouped_data, 
                      id_vars=['difficulty', 'model_stage'], 
                      value_vars=metrics_to_plot,
                      var_name='metric', 
                      value_name='value')
    
    plot_df['metric'] = plot_df['metric'].map(metric_names)

    # Use catplot for a faceted bar plot view
    g = sns.catplot(
        data=plot_df, 
        x='difficulty', 
        y='value', 
        hue='model_stage', 
        col='metric',
        kind='bar', 
        sharey=False, 
        col_wrap=2, 
        height=6, 
        aspect=1.2,
        palette='viridis'
    )

    g.fig.suptitle('Model Performance Comparison vs. Problem Difficulty', fontsize=24, y=1.05)
    g.set_titles("{col_name}", size=18)
    g.set_axis_labels("Problem Difficulty", "Mean Value", size=14)
    
    # Move and format the legend to the top
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(0.5, 0.99), 
        ncol=3, 
        title='Model Stage', 
        frameon=True,
        fontsize=14
    )
    if g.legend:
        g.legend.get_title().set_fontsize('16')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"\nComparison plot saved to: {plot_path}")
    except IOError as e:
        print(f"Error saving plot to {plot_path}: {e}")

def main():
    """
    Main function to load multiple CSVs, aggregate data, and generate comparisons.
    """
    parser = argparse.ArgumentParser(description="Compare metrics from multiple evaluation CSV files.")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, 
                        help="Paths to the CSV files to compare.")
    parser.add_argument("--labels", type=str, nargs='+', required=True, 
                        help="Labels for each CSV file/model stage (e.g., 'Pre-training', 'Distillation', 'RL').")
    parser.add_argument("--output_csv", type=str, required=True, 
                        help="Path to save the combined and aggregated statistics in a CSV file.")
    parser.add_argument("--output_plot", type=str, required=True, 
                        help="Path to save the comparison plot as a PDF.")
    args = parser.parse_args()

    if len(args.csv_files) != len(args.labels):
        print("Error: The number of CSV files must match the number of labels.")
        return

    all_dfs = []
    for csv_file, label in zip(args.csv_files, args.labels):
        try:
            df = pd.read_csv(csv_file)
            df['model_stage'] = label
            all_dfs.append(df)
            print(f"Loaded {csv_file} with label '{label}'")
        except FileNotFoundError:
            print(f"Warning: File not found at {csv_file}, skipping.")
    
    if not all_dfs:
        print("No valid data loaded. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- Data Preprocessing ---
    combined_df['difficulty'] = pd.to_numeric(combined_df['difficulty'], errors='coerce')
    combined_df.dropna(subset=['difficulty'], inplace=True)
    combined_df = combined_df[combined_df['difficulty'] != -1].copy()
    combined_df['difficulty'] = combined_df['difficulty'].astype(int)

    # --- Determine order for model stages based on correctness rate ---
    if 'correctness_rate' in combined_df.columns:
        stage_order = combined_df.groupby('model_stage')['correctness_rate'].mean().sort_values().index.tolist()
        print(f"Reordering model stages based on mean correctness: {stage_order}")
        combined_df['model_stage'] = pd.Categorical(combined_df['model_stage'], categories=stage_order, ordered=True)
    else:
        print("Warning: 'correctness_rate' column not found. Using order from --labels argument.")
        combined_df['model_stage'] = pd.Categorical(combined_df['model_stage'], categories=args.labels, ordered=True)

    # --- Aggregate Data ---
    agg_cols = {
        'correctness_rate': 'mean', 'average_entropy': 'mean', 'avg_think_entropy': 'mean',
        'avg_answer_entropy': 'mean', 'final_answer_entropy': 'mean', 'reasoning_similarity': 'mean',
        'avg_gen_tokens': 'mean', 'avg_think_tokens': 'mean', 'avg_answer_tokens': 'mean'
    }
    # Filter out columns that don't exist in the DataFrame
    existing_agg_cols = {k: v for k, v in agg_cols.items() if k in combined_df.columns}
    grouped_data = combined_df.groupby(['difficulty', 'model_stage']).agg(existing_agg_cols).reset_index()

    # --- Save Aggregated Data ---
    try:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        grouped_data.to_csv(args.output_csv, index=False)
        print(f"\nAggregated comparison data saved to: {args.output_csv}")
    except IOError as e:
        print(f"Error saving aggregated CSV to {args.output_csv}: {e}")

    # --- Create and Save Plot ---
    if args.output_plot:
        try:
            output_dir = os.path.dirname(args.output_plot)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            create_comparison_plot(grouped_data, args.output_plot)
        except Exception as e:
            print(f"An error occurred during plot generation: {e}")

if __name__ == "__main__":
    main()
