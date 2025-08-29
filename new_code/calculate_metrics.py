import json
import argparse
import collections
import numpy as np
import math
import csv
import os
import re

# --- Optional plotting imports ---
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# --- Optional sentence-transformers import for semantic similarity ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SIMILARITY_AVAILABLE = True
    # Lazy load the model so it doesn't slow down startup if not used.
    similarity_model = None
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False
    similarity_model = None

# --- Optional vLLM import for accelerated embedding ---
try:
    import torch
    import vllm
    VLLM_AVAILABLE = True
    vllm_similarity_model = None
except ImportError:
    VLLM_AVAILABLE = False
    vllm_similarity_model = None

def estimate_pass_at_k(n, c, k):
    """
    Calculates pass@k for a single problem.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def assign_budget_label(correctness_rate, average_entropy):
    """
    Assigns a budget control label based on heuristic rules.
    
    :param correctness_rate: The correctness rate for the problem.
    :param average_entropy: The average generation entropy for the problem.
    :return: A string label: "Normal", "Too Hard", or "Too Easy".
    """
    # Rule 1: Identify the "Normal" (sweet spot) cases first.
    # These are high-accuracy, high-confidence (low entropy) problems.
    if correctness_rate > 0.85 and average_entropy < 0.35:
        return "Normal"  # Normal difficulty, requires thinking.

    # Rule 2: Identify the "Too Hard" cases.
    # These are problems where the model consistently fails.
    elif correctness_rate < 0.6:
        return "Too Hard"  # Too difficult, can abandon thinking.

    # Rule 3: Everything else is "Too Easy".
    # These are typically high-accuracy but also higher-entropy, or other edge cases.
    else:
        return "Too Easy"  # Too easy, doesn't require deep thinking.

def extract_final_answer(text):
    """
    Extracts the content from the first \boxed{} in the given text.
    """
    if not isinstance(text, str):
        return None
        
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None # Return None if no box is found

def calculate_distribution_entropy(items):
    """
    Calculates the Shannon entropy over a list of items (e.g., final answers).
    The items can contain None, which will be counted as a separate category.
    """
    if not items:
        return 0.0

    counts = collections.Counter(items)
    total_items = len(items)
    
    if total_items == 0:
        return 0.0
        
    probabilities = [count / total_items for count in counts.values()]
    
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    
    return entropy


def calculate_reasoning_similarity(texts, model_name='all-MiniLM-L6-v2', backend='sentence-transformers', max_len=1024):
    """
    Calculates the average pairwise cosine similarity of reasoning paths (texts).
    Supports both 'sentence-transformers' and 'vllm' backends.
    """
    # 1. Filter out None or empty strings
    valid_texts = [text for text in texts if text and text.strip()]

    # 2. Truncate texts if max_len is provided and positive
    if max_len and max_len > 0:
        proc_texts = [text[:max_len] for text in valid_texts]
    else:
        proc_texts = valid_texts
    
    # 3. If less than 2 valid paths, similarity is not meaningful or perfect.
    if len(proc_texts) < 2:
        return 1.0

    # 4. Dispatch to the appropriate backend
    if backend == 'vllm':
        return _calculate_similarity_vllm(proc_texts, model_name)
    else: # Default to sentence-transformers
        return _calculate_similarity_st(proc_texts, model_name)

def _get_vllm_embeddings(texts, model_name):
    """Helper function to get embeddings using vLLM."""
    global vllm_similarity_model, VLLM_AVAILABLE
    if not VLLM_AVAILABLE:
        print("\nError: vLLM backend was selected, but the vllm library is not installed.")
        print("Please install it or choose the 'sentence-transformers' backend.")
        return None

    if vllm_similarity_model is None:
        print(f"\nLoading vLLM embedding model '{model_name}' for the first time...")
        try:
            # For embedding tasks, vLLM recommends a larger GPU memory utilization
            vllm_similarity_model = vllm.LLM(model=model_name, task="embed", gpu_memory_utilization=0.7,
            dtype="bfloat16", enable_prefix_caching=True,tensor_parallel_size=1)
            print("vLLM model loaded successfully.")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            VLLM_AVAILABLE = False # Avoid retrying
            return None
            
    outputs = vllm_similarity_model.embed(texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    return embeddings

def _get_st_embeddings(texts, model_name):
    """Helper function to get embeddings using sentence-transformers."""
    global similarity_model, SEMANTIC_SIMILARITY_AVAILABLE
    if not SEMANTIC_SIMILARITY_AVAILABLE:
        print("\nError: sentence-transformers library not installed.")
        return None

    if similarity_model is None:
        print(f"\nLoading sentence-transformer model '{model_name}' for the first time...")
        try:
            similarity_model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence-transformer model: {e}")
            SEMANTIC_SIMILARITY_AVAILABLE = False
            return None
            
    return similarity_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def _calculate_similarity_vllm(texts, model_name):
    """Similarity calculation logic using vLLM."""
    embeddings = _get_vllm_embeddings(texts, model_name)
    if embeddings is None:
        return 0.0

    sim_matrix = cosine_similarity(embeddings.cpu().numpy())
    upper_triangle_indices = np.triu_indices(len(texts), k=1)
    pairwise_similarities = sim_matrix[upper_triangle_indices]
    
    return float(np.mean(pairwise_similarities)) if len(pairwise_similarities) > 0 else 1.0

def _calculate_similarity_st(texts, model_name):
    """Similarity calculation logic using sentence-transformers."""
    embeddings = _get_st_embeddings(texts, model_name)
    if embeddings is None:
        return 0.0

    sim_matrix = cosine_similarity(embeddings.cpu().numpy())
    upper_triangle_indices = np.triu_indices(len(texts), k=1)
    pairwise_similarities = sim_matrix[upper_triangle_indices]

    return float(np.mean(pairwise_similarities)) if len(pairwise_similarities) > 0 else 1.0


def create_and_save_plot(stats_data, all_results, plot_path):
    """
    Creates and saves a plot with multiple subplots showing metrics vs. difficulty.
    """
    if not PLOTTING_AVAILABLE:
        print("\nWarning: Plotting libraries (matplotlib, pandas, seaborn) not found. Skipping plot generation.")
        print("Please install them to enable this feature: pip install matplotlib pandas seaborn")
        return

    if not stats_data:
        print("No data available for plotting.")
        return

    df = pd.DataFrame(stats_data)
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
    df.dropna(subset=['difficulty'], inplace=True)
    df = df[df['difficulty'] != -1].copy()

    if df.empty:
        print("No data with valid difficulty values available for plotting after filtering.")
        return

    # Group per-problem stats for subplots 1 and 3
    grouped_data = df.groupby('difficulty').agg({
        'correctness_rate': 'mean',
        'average_entropy': 'mean',
        'avg_think_entropy': 'mean',
        'avg_answer_entropy': 'mean',
        'avg_top_5_think_entropy': 'mean',
        'avg_top_5_answer_entropy': 'mean',
        'reasoning_similarity': 'mean',
        'final_answer_entropy': 'mean',
        'avg_gen_tokens': 'mean',
        'avg_think_tokens': 'mean',
        'avg_answer_tokens': 'mean'
    }).sort_index().reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    # Create 5 subplots, sharing the x-axis
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(18, 50), sharex=True)
    fig.suptitle('Metrics vs. Problem Difficulty', fontsize=24, y=0.95)

    # --- Subplot 1: Correctness, Similarity, and Answer Entropy ---
    ax1.set_title('Correctness, Similarity & Answer Entropy', fontsize=20)
    ax1.set_ylabel('Correctness Rate', fontsize=16)
    
    # Bar plot for correctness
    sns.barplot(data=grouped_data, x='difficulty', y='correctness_rate', ax=ax1, color='skyblue', alpha=0.7, label='Correctness Rate')
    
    # Line plots on a secondary y-axis
    ax1b = ax1.twinx()
    ax1b.plot(grouped_data.index, grouped_data['reasoning_similarity'], color='darkblue', marker='D', linestyle='-', markersize=8, label='Reasoning Similarity')
    ax1b.plot(grouped_data.index, grouped_data['final_answer_entropy'], color='mediumseagreen', marker='s', linestyle=':', markersize=8, label='Final Answer Dist. Entropy')
    ax1b.set_ylabel('Similarity / Entropy', fontsize=16)
    
    # Combine legends for ax1 and ax1b
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=18)
    ax1.set_ylim(0, 1.05)

    # --- Subplot 2: Generation Entropies (Overall) ---
    ax2.set_title('Generation Entropy Metrics (Overall)', fontsize=20)
    ax2.set_ylabel('Average Entropy', fontsize=16)
    
    ax2.plot(grouped_data.index, grouped_data['average_entropy'], color='tomato', marker='o', linestyle='--', markersize=8, label='Avg. Total Gen. Entropy')
    ax2.plot(grouped_data.index, grouped_data['avg_think_entropy'], color='darkorange', marker='^', linestyle='-.', markersize=8, label='Avg. Think Stage Entropy')
    ax2.plot(grouped_data.index, grouped_data['avg_answer_entropy'], color='purple', marker='v', linestyle='-.', markersize=8, label='Avg. Answer Stage Entropy')
    ax2.plot(grouped_data.index, grouped_data['avg_top_5_think_entropy'], color='brown', marker='s', linestyle=':', markersize=6, label='Avg. Top-5 Think Token Entropy')
    ax2.plot(grouped_data.index, grouped_data['avg_top_5_answer_entropy'], color='deeppink', marker='P', linestyle=':', markersize=6, label='Avg. Top-5 Answer Token Entropy')
    ax2.legend(loc='upper left', fontsize=18)

    # --- Subplot 3: Generation Entropies (Split by Correctness) ---
    ax3.set_title('Generation Entropy Metrics (by Correctness)', fontsize=20)
    ax3.set_ylabel('Average Entropy', fontsize=16)

    # Process per-sample results for entropy plot
    df_samples = pd.DataFrame(all_results)
    df_samples['difficulty'] = pd.to_numeric(df_samples['difficulty'], errors='coerce')
    df_samples.dropna(subset=['difficulty'], inplace=True)
    df_samples = df_samples[df_samples['difficulty'] != -1].copy()
    
    def is_correct(row):
        correctness = row.get('xverify_evaluation', {}).get('Correctness')
        return correctness is True or str(correctness).lower() == 'correct' or correctness == 1
    
    df_samples['is_correct'] = df_samples.apply(is_correct, axis=1)

    entropy_grouped = df_samples.groupby(['difficulty', 'is_correct']).agg({
        'average_entropy': 'mean',
        'think_average_entropy': 'mean',
        'answer_average_entropy': 'mean'
    }).rename(columns={
        'average_entropy': 'Avg. Total Gen. Entropy',
        'think_average_entropy': 'Avg. Think Stage Entropy',
        'answer_average_entropy': 'Avg. Answer Stage Entropy'
    }).reset_index()

    difficulty_map = {difficulty: i for i, difficulty in enumerate(grouped_data['difficulty'])}
    
    # Define plotting aesthetics
    styles = {
        'Correct': {'marker': 'o', 'linestyle': '-'},
        'Incorrect': {'marker': 'x', 'linestyle': '--'}
    }
    colors = {
        'Avg. Total Gen. Entropy': 'tomato',
        'Avg. Think Stage Entropy': 'darkorange',
        'Avg. Answer Stage Entropy': 'purple'
    }

    for status, style in styles.items():
        correct_bool = (status == 'Correct')
        subset = entropy_grouped[entropy_grouped['is_correct'] == correct_bool]
        if subset.empty:
            continue
            
        x_values = subset['difficulty'].map(difficulty_map)
        
        for metric in colors:
            ax3.plot(x_values, subset[metric], 
                     color=colors[metric], 
                     marker=style['marker'], 
                     linestyle=style['linestyle'],
                     markersize=8, 
                     label=f'{metric} ({status})')

    ax3.legend(loc='upper left', fontsize=14)

    # --- Subplot 4: Token Lengths (Overall) ---
    ax4.set_title('Token Length Metrics (Overall)', fontsize=20)
    ax4.set_ylabel('Average Token Length', fontsize=16)
    
    ax4.plot(grouped_data.index, grouped_data['avg_gen_tokens'], color='gold', marker='P', linestyle=':', markersize=8, label='Avg. Total Tokens')
    ax4.plot(grouped_data.index, grouped_data['avg_think_tokens'], color='darkcyan', marker='X', linestyle=':', markersize=8, label='Avg. Think Tokens')
    ax4.plot(grouped_data.index, grouped_data['avg_answer_tokens'], color='magenta', marker='H', linestyle=':', markersize=8, label='Avg. Answer Tokens')
    ax4.legend(loc='upper left', fontsize=18)
    
    # --- Subplot 5: Token Lengths (by Correctness) ---
    ax5.set_title('Token Length Metrics (by Correctness)', fontsize=20)
    ax5.set_ylabel('Average Token Length', fontsize=16)

    df_samples['total_tokens'] = df_samples.get('think_tokens', 0) + df_samples.get('answer_tokens', 0)
    
    token_grouped = df_samples.groupby(['difficulty', 'is_correct']).agg({
        'total_tokens': 'mean',
        'think_tokens': 'mean',
        'answer_tokens': 'mean'
    }).rename(columns={
        'total_tokens': 'Avg. Total Tokens',
        'think_tokens': 'Avg. Think Tokens',
        'answer_tokens': 'Avg. Answer Tokens'
    }).reset_index()

    token_colors = {
        'Avg. Total Tokens': 'gold',
        'Avg. Think Tokens': 'darkcyan',
        'Avg. Answer Tokens': 'magenta'
    }

    for status, style in styles.items():
        correct_bool = (status == 'Correct')
        subset = token_grouped[token_grouped['is_correct'] == correct_bool]
        if subset.empty:
            continue
            
        x_values = subset['difficulty'].map(difficulty_map)
        
        for metric in token_colors:
            ax5.plot(x_values, subset[metric],
                     color=token_colors[metric],
                     marker=style['marker'],
                     linestyle=style['linestyle'],
                     markersize=8,
                     label=f'{metric} ({status})')

    ax5.legend(loc='upper left', fontsize=14)

    # Set shared x-axis label
    ax5.set_xlabel('Problem Difficulty', fontsize=18)
    # Since x-axis is shared, we can set the ticks for the last plot
    ax5.set_xticks(grouped_data.index)
    ax5.set_xticklabels(grouped_data['difficulty'].astype(int), fontsize=12)


    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap

    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"\nPlot saved to: {plot_path}")
    except IOError as e:
        print(f"Error saving plot to {plot_path}: {e}")

def create_confidence_distribution_plot(all_results, group_by_key, final_answer_entropies, budget_labels, plot_path):
    """
    Creates and saves multi-dimensional metric distribution plots.
    The distribution is broken down by correctness and budget label (difficulty).
    """
    if not PLOTTING_AVAILABLE:
        print("\nWarning: Plotting libraries not found. Skipping confidence distribution plot.")
        return

    plot_data = []
    for r in all_results:
        correctness = r.get('xverify_evaluation', {}).get('Correctness')
        is_correct = correctness is True or str(correctness).lower() == 'correct' or correctness == 1
        problem_key = r.get(group_by_key)

        plot_data.append({
            'Correctness': 'Correct' if is_correct else 'Incorrect',
            'Mean Confidence': -r.get('average_entropy', 0),
            'Think Confidence': -r.get('think_average_entropy', 0),
            'Answer Confidence': -r.get('answer_average_entropy', 0),
            'Total Tokens': r.get('think_tokens', 0) + r.get('answer_tokens', 0),
            'Final Answer Dist. Entropy': final_answer_entropies.get(problem_key, 0),
            'Budget Label': budget_labels.get(problem_key, 'N/A'),
            'Bottom 10% Confidence': 0  # Placeholder, calculated next
        })
        
        # Calculate Bottom 10% Confidence separately
        all_tokens = r.get('think_tokens_with_entropy', []) + r.get('answer_tokens_with_entropy', [])
        if all_tokens:
            all_tokens.sort(key=lambda x: x['entropy'], reverse=True)
            num_bottom = max(1, int(len(all_tokens) * 0.10))
            highest_entropies = [t['entropy'] for t in all_tokens[:num_bottom]]
            if highest_entropies:
                plot_data[-1]['Bottom 10% Confidence'] = -np.mean(highest_entropies)

    if not plot_data:
        print("No data available for confidence distribution plot.")
        return

    df = pd.DataFrame(plot_data)
    
    # --- Create the new composite hue group ---
    df['Distribution Group'] = df['Correctness'] + ' - ' + df['Budget Label'].replace({
        'Too Easy': 'Easy', 'Too Hard': 'Hard', 'Normal': 'Normal'
    })

    # --- Define a clear color palette and order for the legend ---
    palette = {
        'Correct - Easy': '#8fbc8f',       # darkseagreen
        'Correct - Normal': '#3cb371',     # mediumseagreen
        'Correct - Hard': '#006400',       # darkgreen
        'Incorrect - Easy': '#f4a460',     # sandybrown
        'Incorrect - Normal': '#d2691e',   # chocolate
        'Incorrect - Hard': '#a52a2a'      # brown
    }
    hue_order = [
        'Correct - Easy', 'Incorrect - Easy', 
        'Correct - Normal', 'Incorrect - Normal', 
        'Correct - Hard', 'Incorrect - Hard'
    ]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('Metric Distributions by Correctness and Difficulty', fontsize=26, y=1.0)

    metrics_to_plot = [
        'Mean Confidence', 'Think Confidence', 'Answer Confidence',
        'Bottom 10% Confidence', 'Final Answer Dist. Entropy', 'Total Tokens'
    ]
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Plot histogram with the new composite hue
        sns.histplot(data=df, x=metric, hue='Distribution Group', hue_order=hue_order, 
                     ax=ax, palette=palette, alpha=0.75, bins=50, multiple="layer", legend=False)
        
        ax.set_title(metric, fontsize=20)
        
        if "Tokens" in metric:
            ax.set_xlabel('Token Count', fontsize=14)
        elif "Entropy" in metric:
            ax.set_xlabel('Entropy', fontsize=14)
        else:
            ax.set_xlabel('Confidence (-Entropy)', fontsize=14)
            
        ax.set_ylabel('Frequency', fontsize=14)
        
        # --- Manually create and add a guaranteed legend ---
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=palette[label], alpha=0.75, label=label) for label in hue_order if label in df['Distribution Group'].unique()]
        ax.legend(handles=legend_elements, title='Category', loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive metric distribution plot saved to: {plot_path}")
    except IOError as e:
        print(f"Error saving confidence plot to {plot_path}: {e}")

def main():
    """
    Main function to load results, calculate metrics, and generate labeled data for probing.
    """
    parser = argparse.ArgumentParser(description="Calculate metrics and generate labeled data from xVerify results.")
    parser.add_argument("--input_file", type=str, 
                        required=True,
                        help="Path to the JSONL file containing evaluation results.")
    parser.add_argument("--group_by_key", type=str, default="question",
                        help="The key in the JSON object to group problems by.")
    parser.add_argument("--output_csv", type=str,
                        help="Optional path to save detailed per-problem statistics in a CSV file.")
    parser.add_argument("--plot_file", type=str,
                        help="Optional path to save a dual-axis plot of metrics vs. difficulty.")
    parser.add_argument("--confidence_plot_file", type=str,
                        help="Optional path to save confidence distribution plots.")
    parser.add_argument("--labeled_csv_path", type=str,
                        help="Path to save the final data with budget control labels for probe training.")
    parser.add_argument("--disable_similarity", action='store_true',
                        help="Disable the calculation of reasoning path similarity to save time and resources.")
    parser.add_argument("--similarity_model_path", type=str, default="all-MiniLM-L6-v2",
                        help="Path or name of the sentence-transformer model to use for similarity calculation.")
    parser.add_argument("--embedding_backend", type=str, choices=['sentence-transformers', 'vllm'], 
                        default='sentence-transformers',
                        help="The backend to use for generating embeddings.")
    parser.add_argument("--similarity_max_len", type=int, default=1024,
                        help="Maximum length of reasoning path text to use for similarity calculation. Set to 0 or negative for no limit.")
    
    args = parser.parse_args()

    results_by_problem = collections.defaultdict(list)
    all_results = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                all_results.append(data)
                problem_key = data.get(args.group_by_key)
                if problem_key:
                    results_by_problem[problem_key].append(data)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_file}. Make sure it is a valid JSONL file.")
        return

    if not results_by_problem:
        print("No data found to analyze.")
        return

    print(f"Loaded results for {len(results_by_problem)} unique problems from {args.input_file}")

    problem_stats = {}
    max_samples = 0
    detailed_stats_list = []
    overall_think_tokens_with_entropy = []
    overall_answer_tokens_with_entropy = []

    for problem_key, results in results_by_problem.items():
        n_samples = len(results)
        max_samples = max(max_samples, n_samples)
        
        n_correct = 0
        total_avg_entropy, total_avg_think_entropy, total_avg_answer_entropy = 0, 0, 0
        total_think_tokens, total_answer_tokens, total_gen_tokens = 0, 0, 0
        final_answers = []
        reasoning_paths = []
        difficulty = results[0].get('difficulty', 'N/A') if results else 'N/A'

        # For high-entropy token analysis
        all_think_tokens_with_entropy = []
        all_answer_tokens_with_entropy = []
        sample_top_5_think_entropies = []
        sample_top_5_answer_entropies = []

        for r in results:
            correctness = r.get('xverify_evaluation', {}).get('Correctness')
            if correctness is True or str(correctness).lower() == 'correct' or correctness == 1:
                n_correct += 1
            
            generated_text = r.get('generated_text')
            reasoning_paths.append(generated_text)
            final_answers.append(extract_final_answer(generated_text))

            # Accumulate metrics from each sample
            total_avg_entropy += r.get('average_entropy', 0)
            total_avg_think_entropy += r.get('think_average_entropy', 0)
            total_avg_answer_entropy += r.get('answer_average_entropy', 0)
            total_think_tokens += r.get('think_tokens', 0)
            total_answer_tokens += r.get('answer_tokens', 0)
            total_gen_tokens += r.get('think_tokens', 0) + r.get('answer_tokens', 0)

            # Analyze per-token entropies
            think_tokens = r.get('think_tokens_with_entropy', [])
            answer_tokens = r.get('answer_tokens_with_entropy', [])
            
            if think_tokens:
                all_think_tokens_with_entropy.extend(think_tokens)
                think_tokens.sort(key=lambda x: x['entropy'], reverse=True)
                top_5 = [t['entropy'] for t in think_tokens[:5]]
                if top_5:
                    sample_top_5_think_entropies.append(np.mean(top_5))
            
            if answer_tokens:
                all_answer_tokens_with_entropy.extend(answer_tokens)
                answer_tokens.sort(key=lambda x: x['entropy'], reverse=True)
                top_5 = [t['entropy'] for t in answer_tokens[:5]]
                if top_5:
                    sample_top_5_answer_entropies.append(np.mean(top_5))

        # Calculate averages over all samples for the problem
        correctness_rate = n_correct / n_samples if n_samples > 0 else 0
        average_entropy = total_avg_entropy / n_samples if n_samples > 0 else 0
        avg_think_entropy = total_avg_think_entropy / n_samples if n_samples > 0 else 0
        avg_answer_entropy = total_avg_answer_entropy / n_samples if n_samples > 0 else 0
        avg_think_tokens = total_think_tokens / n_samples if n_samples > 0 else 0
        avg_answer_tokens = total_answer_tokens / n_samples if n_samples > 0 else 0
        avg_gen_tokens = total_gen_tokens / n_samples if n_samples > 0 else 0
        final_answer_entropy = calculate_distribution_entropy(final_answers)
        
        # New stats for high-entropy tokens
        avg_top_5_think_entropy = np.mean(sample_top_5_think_entropies) if sample_top_5_think_entropies else 0
        avg_top_5_answer_entropy = np.mean(sample_top_5_answer_entropies) if sample_top_5_answer_entropies else 0

        all_think_tokens_with_entropy.sort(key=lambda x: x['entropy'], reverse=True)
        top_30_think_tokens = all_think_tokens_with_entropy[:30]

        all_answer_tokens_with_entropy.sort(key=lambda x: x['entropy'], reverse=True)
        top_30_answer_tokens = all_answer_tokens_with_entropy[:30]

        overall_think_tokens_with_entropy.extend(all_think_tokens_with_entropy)
        overall_answer_tokens_with_entropy.extend(all_answer_tokens_with_entropy)

        # Calculate reasoning similarity if not disabled
        reasoning_similarity = 0.0
        if not args.disable_similarity:
            reasoning_similarity = calculate_reasoning_similarity(
                reasoning_paths, 
                model_name=args.similarity_model_path,
                backend=args.embedding_backend,
                max_len=args.similarity_max_len
            )

        # Assign the budget control label
        budget_label = assign_budget_label(correctness_rate, average_entropy)

        problem_stats[problem_key] = {'n': n_samples, 'c': n_correct}
        detailed_stats_list.append({
            'problem_key': problem_key,
            'difficulty': difficulty,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'correctness_rate': correctness_rate,
            'average_entropy': average_entropy,
            'avg_think_entropy': avg_think_entropy,
            'avg_answer_entropy': avg_answer_entropy,
            'final_answer_entropy': final_answer_entropy,
            'reasoning_similarity': reasoning_similarity,
            'avg_think_tokens': avg_think_tokens,
            'avg_answer_tokens': avg_answer_tokens,
            'avg_gen_tokens': avg_gen_tokens,
            'budget_label': budget_label,
            'avg_top_5_think_entropy': avg_top_5_think_entropy,
            'avg_top_5_answer_entropy': avg_top_5_answer_entropy,
            'top_30_think_tokens': json.dumps(top_30_think_tokens),
            'top_30_answer_tokens': json.dumps(top_30_answer_tokens)
        })

    total_problems = len(problem_stats)
    
    final_answer_entropies = {
        stats['problem_key']: stats['final_answer_entropy']
        for stats in detailed_stats_list
    }
    budget_labels = {
        stats['problem_key']: stats['budget_label']
        for stats in detailed_stats_list
    }
    
    # --- Print overall high-entropy tokens ---
    overall_think_tokens_with_entropy.sort(key=lambda x: x['entropy'], reverse=True)
    overall_answer_tokens_with_entropy.sort(key=lambda x: x['entropy'], reverse=True)

    print("\n--- Top 30 High-Entropy Think Tokens (Overall) ---")
    for token_info in overall_think_tokens_with_entropy[:30]:
        print(f"Token: {token_info['token']!r:<25} Entropy: {token_info['entropy']:.4f}")

    print("\n--- Top 30 High-Entropy Answer Tokens (Overall) ---")
    for token_info in overall_answer_tokens_with_entropy[:30]:
        print(f"Token: {token_info['token']!r:<25} Entropy: {token_info['entropy']:.4f}")
        
    pass_at_k = collections.defaultdict(float)
    k_values = list(range(1, max_samples + 1)) if max_samples > 0 else []

    for stats in problem_stats.values():
        n = stats['n']
        c = stats['c']
        for k in k_values:
            if k <= n :
                pass_at_k[k] += estimate_pass_at_k(n, c, k)

    print("\n--- Pass@k Results ---")
    passk_to_print = [1, 5, 10, 20, 40]
    for k in passk_to_print:
        if k in pass_at_k:
            pass_at_k_value = pass_at_k[k] / total_problems if total_problems > 0 else 0
            print(f"Pass@{k}: {pass_at_k_value:.4f}")
    
    total_samples = sum(s['n'] for s in problem_stats.values())
    total_correct = sum(s['c'] for s in problem_stats.values())
    if total_samples > 0:
        accuracy = total_correct / total_samples
        print(f"\nOverall sample-level accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    # --- Save detailed stats and labeled data ---
    if args.output_csv:
        try:
            detailed_stats_list.sort(key=lambda x: (x.get('difficulty', 0), x['correctness_rate']), reverse=True)
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as f_out:
                if detailed_stats_list:
                    fieldnames = [
                        'problem_key', 'difficulty', 'n_samples', 'n_correct', 'correctness_rate', 
                        'average_entropy', 'avg_think_entropy', 'avg_answer_entropy', 
                        'final_answer_entropy', 'reasoning_similarity', 
                        'avg_think_tokens', 'avg_answer_tokens', 'avg_gen_tokens',
                        'budget_label',
                        'avg_top_5_think_entropy', 'avg_top_5_answer_entropy',
                        'top_30_think_tokens', 'top_30_answer_tokens'
                    ]
                    writer = csv.DictWriter(f_out, fieldnames=[k for k in fieldnames if k in detailed_stats_list[0]])
                    writer.writeheader()
                    writer.writerows(detailed_stats_list)
            print(f"\nDetailed per-problem statistics saved to: {args.output_csv}")
        except IOError as e:
            print(f"Error writing to CSV file {args.output_csv}: {e}")
    
    if args.labeled_csv_path:
        try:
            # You might want a different sorting for the probe training data, e.g., by problem key
            detailed_stats_list.sort(key=lambda x: x['problem_key'])
            with open(args.labeled_csv_path, 'w', newline='', encoding='utf-8') as f_out:
                if detailed_stats_list:
                    # For probe training, you might only need the question and the label
                    probe_fieldnames = ['problem_key', 'budget_label']
                    writer = csv.DictWriter(f_out, fieldnames=probe_fieldnames)
                    writer.writeheader()
                    # Write only the necessary columns
                    writer.writerows([{'problem_key': d['problem_key'], 'budget_label': d['budget_label']} for d in detailed_stats_list])
            print(f"Labeled data for probe training saved to: {args.labeled_csv_path}")
        except IOError as e:
            print(f"Error writing to labeled CSV file {args.labeled_csv_path}: {e}")
            
    if args.plot_file:
        create_and_save_plot(detailed_stats_list, all_results, args.plot_file)

    if args.confidence_plot_file:
        create_confidence_distribution_plot(all_results, args.group_by_key, final_answer_entropies, budget_labels, args.confidence_plot_file)


if __name__ == "__main__":
    main()
