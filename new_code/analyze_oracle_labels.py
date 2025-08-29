#!/usr/bin/env python3
"""
Oracle Label Analysis

This script analyzes the relationship between the new oracle labels and
the original difficulty/entropy metrics from the data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os


def load_oracle_data(oracle_file):
    """Load oracle selection results."""
    oracle_data = []
    with open(oracle_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                oracle_data.append(data)
    
    print(f"Loaded {len(oracle_data)} oracle entries")
    return oracle_data


def extract_oracle_metrics(oracle_data):
    """Extract relevant metrics from oracle data."""
    extracted_data = []
    
    for entry in oracle_data:
        oracle_entry = entry.get('oracle_entry', {})
        
        extracted = {
            'question': entry['question'][:100] + '...' if len(entry['question']) > 100 else entry['question'],
            'oracle_strategy': entry['oracle_strategy'],
            'difficulty_label': entry['difficulty_label'],
            'oracle_correct': entry['oracle_correct'],
            'oracle_think_tokens': entry['oracle_think_tokens'],
            'total_attempts': entry['total_attempts'],
            'correct_attempts': entry['correct_attempts'],
            # Extract metrics from oracle entry
            'original_difficulty': oracle_entry.get('difficulty', None),
            'total_entropy': oracle_entry.get('total_entropy', None),
            'average_entropy': oracle_entry.get('average_entropy', None),
            'think_total_entropy': oracle_entry.get('think_total_entropy', None),
            'think_average_entropy': oracle_entry.get('think_average_entropy', None),
            'answer_total_entropy': oracle_entry.get('answer_total_entropy', None),
            'answer_average_entropy': oracle_entry.get('answer_average_entropy', None),
            'generated_tokens': oracle_entry.get('generated_tokens', None),
            'think_tokens': oracle_entry.get('think_tokens', None),
            'answer_tokens': oracle_entry.get('answer_tokens', None),
        }
        extracted_data.append(extracted)
    
    return pd.DataFrame(extracted_data)


def create_label_difficulty_analysis(df, output_dir):
    """Analyze relationship between oracle labels and original difficulty."""
    # Clean difficulty data
    df_clean = df.dropna(subset=['original_difficulty'])
    df_clean = df_clean[df_clean['original_difficulty'] != -1]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Oracle Labels vs Original Difficulty Analysis', fontsize=16, y=0.95)
    
    # 1. Distribution of original difficulty by oracle label
    ax1 = axes[0, 0]
    df_clean.boxplot(column='original_difficulty', by='difficulty_label', ax=ax1)
    ax1.set_title('Original Difficulty Distribution by Oracle Label')
    ax1.set_xlabel('Oracle Label')
    ax1.set_ylabel('Original Difficulty')
    plt.suptitle('')  # Remove automatic title
    
    # 2. Heatmap of label vs difficulty
    ax2 = axes[0, 1]
    crosstab = pd.crosstab(df_clean['difficulty_label'], df_clean['original_difficulty'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Count: Oracle Label vs Original Difficulty')
    ax2.set_xlabel('Original Difficulty')
    ax2.set_ylabel('Oracle Label')
    
    # 3. Average original difficulty by oracle label
    ax3 = axes[1, 0]
    avg_diff = df_clean.groupby('difficulty_label')['original_difficulty'].mean().sort_values()
    avg_diff.plot(kind='bar', ax=ax3, color=['lightcoral', 'skyblue', 'lightgreen'])
    ax3.set_title('Average Original Difficulty by Oracle Label')
    ax3.set_xlabel('Oracle Label')
    ax3.set_ylabel('Average Original Difficulty')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Correctness rate by oracle label and difficulty
    ax4 = axes[1, 1]
    correctness = df_clean.groupby(['difficulty_label', 'original_difficulty'])['oracle_correct'].mean().unstack()
    sns.heatmap(correctness, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, vmin=0, vmax=1)
    ax4.set_title('Oracle Correctness Rate by Label and Difficulty')
    ax4.set_xlabel('Original Difficulty')
    ax4.set_ylabel('Oracle Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oracle_labels_vs_difficulty.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'oracle_labels_vs_difficulty.png'), dpi=300, bbox_inches='tight')
    print(f"Saved difficulty analysis plots to {output_dir}")
    
    # Print summary statistics
    print("\n=== Oracle Label vs Original Difficulty Summary ===")
    summary = df_clean.groupby('difficulty_label')['original_difficulty'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary)
    
    return fig


def create_label_entropy_analysis(df, output_dir):
    """Analyze relationship between oracle labels and entropy metrics."""
    # Clean entropy data
    entropy_cols = ['total_entropy', 'average_entropy', 'think_total_entropy', 
                   'think_average_entropy', 'answer_total_entropy', 'answer_average_entropy']
    df_clean = df.dropna(subset=entropy_cols)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Oracle Labels vs Entropy Metrics Analysis', fontsize=16, y=0.95)
    
    axes = axes.flatten()
    
    # Create box plots for each entropy metric
    for i, entropy_col in enumerate(entropy_cols):
        ax = axes[i]
        df_clean.boxplot(column=entropy_col, by='difficulty_label', ax=ax)
        ax.set_title(f'{entropy_col.replace("_", " ").title()} by Oracle Label')
        ax.set_xlabel('Oracle Label')
        ax.set_ylabel(entropy_col.replace('_', ' ').title())
        plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oracle_labels_vs_entropy.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'oracle_labels_vs_entropy.png'), dpi=300, bbox_inches='tight')
    print(f"Saved entropy analysis plots to {output_dir}")
    
    # Print summary statistics for entropy
    print("\n=== Oracle Label vs Entropy Summary ===")
    for entropy_col in entropy_cols:
        print(f"\n{entropy_col}:")
        summary = df_clean.groupby('difficulty_label')[entropy_col].agg(['count', 'mean', 'std'])
        print(summary)
    
    return fig


def create_comprehensive_analysis(df, output_dir):
    """Create comprehensive analysis combining multiple metrics."""
    # Clean data
    df_clean = df.dropna(subset=['original_difficulty', 'average_entropy'])
    df_clean = df_clean[df_clean['original_difficulty'] != -1]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Oracle Label Analysis', fontsize=16, y=0.95)
    
    # 1. Scatter plot: Original difficulty vs Average entropy, colored by oracle label
    ax1 = axes[0, 0]
    for label in df_clean['difficulty_label'].unique():
        mask = df_clean['difficulty_label'] == label
        ax1.scatter(df_clean[mask]['original_difficulty'], 
                   df_clean[mask]['average_entropy'],
                   label=label, alpha=0.6, s=30)
    ax1.set_xlabel('Original Difficulty')
    ax1.set_ylabel('Average Entropy')
    ax1.set_title('Original Difficulty vs Average Entropy by Oracle Label')
    ax1.legend()
    
    # 2. Think tokens vs Oracle label
    ax2 = axes[0, 1]
    df_clean.boxplot(column='oracle_think_tokens', by='difficulty_label', ax=ax2)
    ax2.set_title('Oracle Think Tokens by Label')
    ax2.set_xlabel('Oracle Label')
    ax2.set_ylabel('Think Tokens')
    plt.suptitle('')
    
    # 3. Correctness rate by oracle label
    ax3 = axes[1, 0]
    correctness_by_label = df_clean.groupby('difficulty_label')['oracle_correct'].mean()
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    bars = ax3.bar(correctness_by_label.index, correctness_by_label.values, color=colors)
    ax3.set_title('Oracle Correctness Rate by Label')
    ax3.set_xlabel('Oracle Label')
    ax3.set_ylabel('Correctness Rate')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 4. Strategy distribution
    ax4 = axes[1, 1]
    strategy_counts = df_clean['oracle_strategy'].value_counts()
    wedges, texts, autotexts = ax4.pie(strategy_counts.values, labels=strategy_counts.index, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Oracle Strategy Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_oracle_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comprehensive_oracle_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis plots to {output_dir}")
    
    return fig


def create_correlation_analysis(df, output_dir):
    """Create correlation analysis between metrics."""
    # Select numeric columns for correlation
    numeric_cols = ['original_difficulty', 'total_entropy', 'average_entropy', 
                   'think_total_entropy', 'think_average_entropy', 
                   'answer_total_entropy', 'answer_average_entropy',
                   'oracle_think_tokens', 'generated_tokens', 'think_tokens', 'answer_tokens']
    
    df_numeric = df[numeric_cols].dropna()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
    
    ax.set_title('Correlation Matrix: Oracle Metrics vs Original Metrics', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved correlation analysis plot to {output_dir}")
    
    # Print strong correlations
    print("\n=== Strong Correlations (|r| > 0.5) ===")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze oracle labels vs difficulty and entropy")
    parser.add_argument("--oracle_file", type=str, required=True,
                        help="Path to oracle selection results JSONL file")
    parser.add_argument("--output_dir", type=str, default="analysis_plots",
                        help="Directory to save analysis plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading oracle data from: {args.oracle_file}")
    oracle_data = load_oracle_data(args.oracle_file)
    
    print("Extracting metrics...")
    df = extract_oracle_metrics(oracle_data)
    
    print(f"Analyzing {len(df)} entries...")
    print(f"Oracle label distribution:\n{df['difficulty_label'].value_counts()}")
    print(f"Oracle strategy distribution:\n{df['oracle_strategy'].value_counts()}")
    
    # Create analyses
    print("\nCreating label vs difficulty analysis...")
    create_label_difficulty_analysis(df, args.output_dir)
    
    print("\nCreating label vs entropy analysis...")
    create_label_entropy_analysis(df, args.output_dir)
    
    print("\nCreating comprehensive analysis...")
    create_comprehensive_analysis(df, args.output_dir)
    
    print("\nCreating correlation analysis...")
    create_correlation_analysis(df, args.output_dir)
    
    # Save processed data
    df_summary = df[['oracle_strategy', 'difficulty_label', 'oracle_correct', 
                    'oracle_think_tokens', 'original_difficulty', 'average_entropy',
                    'think_average_entropy', 'answer_average_entropy']].copy()
    
    csv_file = os.path.join(args.output_dir, 'oracle_analysis_data.csv')
    df_summary.to_csv(csv_file, index=False)
    print(f"\nSaved analysis data to: {csv_file}")
    
    print(f"\nAll plots saved to: {args.output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
