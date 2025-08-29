import json
import argparse
import os
import tempfile
import shutil

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
                        default="new_data/deepmath_sampled_by_original_difficulty_entropy.jsonl",
                        help="Path to the JSONL file containing generation results to be evaluated.")
    parser.add_argument("--output_file", type=str,
                        default="new_data/evaluation_results.jsonl",
                        help="Path to save the final evaluation results.")
    parser.add_argument("--max_num_samples", type=int,
                        default=99999999,
                        help="Maximum number of samples to evaluate.")
    
    return parser.parse_args()

# --- Main Evaluation Logic ---
def main():
    """
    Loads generated results, prepares them for batch evaluation, runs the evaluator,
    merges the results, and saves them to a final file.
    """
    args = parse_args()

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
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            original_data = [json.loads(line) for line in f_in]
        
        if not original_data:
            print("No data to evaluate. Exiting.")
            return

        data_to_process = original_data[:args.max_num_samples]
        print(f"Loaded {len(original_data)} entries, processing the first {len(data_to_process)}.")

        formatted_data_for_eval = []
        for item in data_to_process:
            generated_text = item.get('generated_text', '')
            if '</think>' in generated_text:
                llm_output = generated_text.split('</think>', 1)[-1].strip()
            else:
                llm_output = generated_text.strip()[:1000]
            formatted_data_for_eval.append({
                "question": item.get('question', ''),
                "llm_output": llm_output,
                "correct_answer": item.get('final_answer', '')
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
            print(f"Error: Mismatch between processed items ({len(data_to_process)}) and evaluated items ({len(eval_results_list)}).")
            return

        print(f"Merging {len(eval_results_list)} evaluation results with original data...")
        final_results = []
        for original, evaluation in zip(data_to_process, eval_results_list):
            merged = original.copy()
            
            # Dynamically get the judgment result key
            judgment_key = f"{model.model_name}_judgment_result"
            judgment_result = evaluation.get(judgment_key)

            # Safely parse the judgment result, whether it's a string or a dict
            correctness, reasoning = None, None
            if isinstance(judgment_result, dict):
                correctness = judgment_result.get('Correctness')
                reasoning = judgment_result.get('Reasoning')
            elif isinstance(judgment_result, str):
                correctness = judgment_result # The result is just a string like "Correct"

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

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # --- Step 5: Clean up temporary directory and its contents ---
        if temp_dir:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()
