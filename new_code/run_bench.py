import json
import argparse
import os
import tempfile
import shutil
from transformers import AutoTokenizer
import math
from collections import defaultdict

# It's good practice to set the workspace for the library if needed.
# This assumes the script is run from the `probe-data` directory.
import sys
sys.path.append('/home/zhtang/workspace/xVerify')

# Now we can import the library modules
from src.xVerify.model import Model
from src.xVerify.eval import Evaluator

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate LLM outputs using xVerify with batch processing.")
    
    # Paths
    parser.add_argument("--model_path", type=str, 
                        default="/home/zhtang/hf_models/IAAR-Shanghai/xVerify-0.5B-I",
                        help="Path to the xVerify model used for evaluation.")
    parser.add_argument("--input_file", type=str, 
                        default="new_results/ds_r1_7b_inference_results.json",
                        help="Path to the JSON file containing generation results to be evaluated.")
    parser.add_argument("--output_file", type=str,
                        default="new_results/ds_r1_7b_evaluation_results.jsonl",
                        help="Path to save the final evaluation results as a JSONL file.")
    parser.add_argument("--max_num_samples", type=int,
                        default=99999999,
                        help="Maximum number of samples to evaluate.")
    parser.add_argument("--pass_k", type=str, default="1",
                        help="Comma-separated list of k values for pass@k calculation (e.g., '1,5').")
    
    return parser.parse_args()

# --- Utility for GSM8K ---
def extract_gsm8k_answer(answer_str: str) -> str:
    """Extracts the final numerical answer from a GSM8K solution string."""
    if not isinstance(answer_str, str):
        return ""
    if '####' in answer_str:
        return answer_str.split('####')[-1].strip().replace(',', '')
    return answer_str

def pass_at_k(n, c, k):
    """
    Calculates pass@k.
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# --- Main Evaluation Logic ---
def main():
    """
    Loads generated results, prepares them for batch evaluation, runs the evaluator,
    merges the results, and saves them to a final file.
    """
    args = parse_args()

    # --- Tokenizer for stats ---
    tokenizer = AutoTokenizer.from_pretrained("/home/zhtang/hf_models/DeepSeek-R1-Distill-Qwen-7B")


    temp_dir = None
    try:
        # --- Step 1: Initialize the xVerify Model and Evaluator ---
        print(f"Initializing xVerify evaluator with model from: {args.model_path}")
        model = Model(
            model_name='xVerify-0.5B-I',
            model_path_or_url=args.model_path,
            inference_mode='local',
            api_key=None
        )
        evaluator = Evaluator(model=model)
        print("Evaluator initialized successfully.")

        # --- Step 2: Load original data and prepare for batch evaluation ---
        original_data = []
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                original_data.append(json.loads(line))
        
        if not original_data:
            print("No data to evaluate. Exiting.")
            return

        data_to_process = original_data[:args.max_num_samples]
        print(f"Loaded {len(original_data)} entries, processing the first {len(data_to_process)}.")

        formatted_data_for_eval = []
        for item in data_to_process:
            generated_text = item.get('response', '')
            if '</think>' in generated_text:
                llm_output = generated_text.split('</think>', 1)[-1].strip()
            else:
                llm_output = generated_text[:-1000]
            
            question = item.get('problem', '')
            correct_answer_raw = item.get('ground_truth_answer', '')
            correct_answer = extract_gsm8k_answer(correct_answer_raw)

            formatted_data_for_eval.append({
                "question": question,
                "llm_output": llm_output,
                "correct_answer": correct_answer
            })

        temp_dir = tempfile.mkdtemp()
        temp_input_path = os.path.join(temp_dir, "eval_input.json")
        temp_output_dir_path = os.path.join(temp_dir, "eval_output")
        
        with open(temp_input_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data_for_eval, f, indent=4)
        
        print(f"Prepared temporary data for batch evaluation at: {temp_input_path}")

        # --- Step 3: Run the batch evaluation ---
        print(f"Starting batch evaluation...")
        evaluator.evaluate(
            data_path=temp_input_path,
            data_size=len(formatted_data_for_eval),
            output_path=temp_output_dir_path
        )
        print("Batch evaluation complete.")

        # --- Step 4: Find, parse, and merge the results ---
        result_files = [f for f in os.listdir(temp_output_dir_path) if f.endswith('.json')]
        if not result_files:
            raise FileNotFoundError("No result file found in the xVerify output directory.")
        
        actual_output_file_path = os.path.join(temp_output_dir_path, result_files[0])
        print(f"Found and reading evaluation results from: {actual_output_file_path}")

        with open(actual_output_file_path, 'r', encoding='utf-8') as f_eval:
            eval_data_dict = json.load(f_eval)
            eval_results_list = eval_data_dict.get('results', [])

        if len(data_to_process) != len(eval_results_list):
            print(f"Warning: Mismatch between processed items ({len(data_to_process)}) and evaluated items ({len(eval_results_list)}). Merging based on index.")
        
        print(f"Merging {len(eval_results_list)} evaluation results with original data...")
        final_results = []
        for i, original in enumerate(data_to_process):
            if i >= len(eval_results_list):
                break
            
            evaluation = eval_results_list[i]
            merged = original.copy()
            
            judgment_key = f"{model.model_name}_judgment_result"
            judgment_result = evaluation.get(judgment_key)

            correctness, reasoning = None, None
            if isinstance(judgment_result, dict):
                correctness = judgment_result.get('Correctness')
                reasoning = judgment_result.get('Reasoning')
            elif isinstance(judgment_result, str):
                correctness = judgment_result

            merged['xverify_evaluation'] = {
                'Correctness': correctness,
                'Reasoning': reasoning
            }
            merged['processed_llm_output'] = evaluation.get('llm_output')
            final_results.append(merged)
        
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for result in final_results:
                f_out.write(json.dumps(result) + '\n')
        
        print(f"All {len(final_results)} merged results saved to {args.output_file}")

        # --- Step 5: Calculate and Print Statistics ---
        k_values = [int(k) for k in args.pass_k.split(',')]
        
        # Group results by problem
        problems = defaultdict(list)
        for r in final_results:
            problems[r['problem']].append(r)
        
        print("\n--- Evaluation Statistics ---")
        
        # Calculate pass@k and token stats per problem, then aggregate
        total_correct_runs = 0
        total_runs = len(final_results)
        
        strategy_stats = defaultdict(lambda: {'problems': defaultdict(list)})

        for problem, runs in problems.items():
            strategy = runs[0]['strategy']
            strategy_stats[strategy]['problems'][problem].extend(runs)

        print("\n--- Accuracy and Token Counts by Strategy ---")
        
        overall_problem_count = len(problems)
        overall_pass_at_k = {k: 0 for k in k_values}
        overall_avg_tokens = 0
        overall_total_correct_runs = 0
        overall_total_runs = 0

        for strategy, s_stats in sorted(strategy_stats.items()):
            num_problems = len(s_stats['problems'])
            strategy_pass_at_k = {k: 0 for k in k_values}
            strategy_total_tokens = 0
            strategy_total_correct_runs = 0
            strategy_total_runs = 0
            
            for problem, runs in s_stats['problems'].items():
                n_runs = len(runs)
                c_correct = sum(1 for r in runs if r.get('xverify_evaluation', {}).get('Correctness') == 'Correct')
                strategy_total_correct_runs += c_correct
                strategy_total_runs += n_runs
                
                for k in k_values:
                    strategy_pass_at_k[k] += pass_at_k(n_runs, c_correct, k)
                
                strategy_total_tokens += sum(r.get('total_generated_tokens', 0) for r in runs)

            overall_total_correct_runs += strategy_total_correct_runs
            overall_total_runs += strategy_total_runs

            avg_tokens_per_problem = (strategy_total_tokens / num_problems) if num_problems > 0 else 0
            avg_tokens_per_sample = (strategy_total_tokens / strategy_total_runs) if strategy_total_runs > 0 else 0
            
            print(f"Strategy: {strategy} ({num_problems} problems)")
            strategy_avg_pass_1 = (strategy_total_correct_runs / strategy_total_runs) * 100 if strategy_total_runs > 0 else 0
            print(f"  Avg Pass@1: {strategy_avg_pass_1:.2f}%")
            for k in k_values:
                pass_k_avg = (strategy_pass_at_k[k] / num_problems) * 100 if num_problems > 0 else 0
                print(f"  Pass@{k}: {pass_k_avg:.2f}%")
                overall_pass_at_k[k] += strategy_pass_at_k[k]
            
            print(f"  Average Tokens per Problem: {avg_tokens_per_problem:.2f}")
            print(f"  Average Tokens per Sample: {avg_tokens_per_sample:.2f}")
            overall_avg_tokens += strategy_total_tokens

        print("\n--- Overall Statistics ---")
        overall_avg_pass_1 = (overall_total_correct_runs / overall_total_runs) * 100 if overall_total_runs > 0 else 0
        print(f"Overall Avg Pass@1: {overall_avg_pass_1:.2f}%")
        for k in k_values:
            overall_pass_k_avg = (overall_pass_at_k[k] / overall_problem_count) * 100 if overall_problem_count > 0 else 0
            print(f"Overall Pass@{k}: {overall_pass_k_avg:.2f}%")

        grand_total_avg_tokens = overall_avg_tokens / overall_problem_count if overall_problem_count > 0 else 0
        print(f"Overall Average Tokens per Problem: {grand_total_avg_tokens:.2f}")

        grand_total_avg_tokens_per_sample = overall_avg_tokens / overall_total_runs if overall_total_runs > 0 else 0
        print(f"Overall Average Tokens per Sample: {grand_total_avg_tokens_per_sample:.2f}")


    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # --- Step 6: Clean up temporary directory and its contents ---
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()
