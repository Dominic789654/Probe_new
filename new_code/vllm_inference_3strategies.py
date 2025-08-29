from vllm import LLM, SamplingParams
import torch
import numpy as np
from transformers import AutoTokenizer
import json
import argparse
from tqdm import tqdm

# --- Argument Parsing ---
def parse_args():
    """
    Parses command-line arguments for the inference script.
    """
    parser = argparse.ArgumentParser(description="Run vLLM inference with three different prompt strategies for each question.")
    
    # Model, Data, and Output paths
    parser.add_argument("--model_path", type=str, 
                        default="/home/zhtang/hf_models/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path to the Hugging Face model.")
    parser.add_argument("--data_file_path", type=str, 
                        default="deepmath/deepmath_sampled_by_original_difficulty.jsonl",
                        help="Path to the JSONL data file containing prompts.")
    parser.add_argument("--output_file_path", type=str,
                        default="entropy_results.jsonl",
                        help="Path to save the output JSONL file.")
    parser.add_argument("--max_num_samples", type=int, default=99999999,
                        help="Maximum number of samples to process from the data file.")

    # Hardware configuration
    parser.add_argument("--tensor_parallel_size", type=int, 
                        default=torch.cuda.device_count(),
                        help="Number of GPUs to use for tensor parallelism.")

    # Sampling parameters
    parser.add_argument("--n", type=int, default=1, 
                        help="Number of different samples to generate for each prompt (for pass@k).")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling probability.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate.")
    parser.add_argument("--logprobs", type=int, default=10, 
                        help="Number of top log probabilities to return for entropy calculation.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference to control memory usage.")
                        
    return parser.parse_args()

# --- Data Loading ---
def load_data_from_file(file_path):
    """
    Loads full data entries (including question, answer, difficulty) from a JSONL file.
    """
    data_entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'question' in data:
                        data_entries.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return []
    print(f"Loaded {len(data_entries)} data entries from {file_path}")
    return data_entries

# --- Helper Function for Entropy Calculation ---
def calculate_entropy_from_logprobs(logprobs_list):
    """Calculates total, average, and per-token entropy from a list of logprobs."""
    total_entropy = 0.0
    token_entropies = []
    if not logprobs_list:
        return 0.0, 0.0, 0, []
    
    num_tokens = len(logprobs_list)
    for token_logprobs in logprobs_list:
        # vLLM returns a dictionary of top-k logprobs for each token.
        # The values are Logprob objects.
        step_logprobs = [p.logprob for p in token_logprobs.values()]
        step_probs = np.exp(step_logprobs)
        step_probs /= np.sum(step_probs)
        # Calculate entropy for this token step
        step_entropy = -np.sum(step_probs * np.log(step_probs))
        token_entropies.append(step_entropy)
        total_entropy += step_entropy
    
    average_entropy = total_entropy / num_tokens if num_tokens > 0 else 0.0
    return total_entropy, average_entropy, num_tokens, token_entropies


# --- Main Inference Logic ---
def main():
    """
    Initializes the vLLM engine, runs inference on prompts from a file,
    generates n samples for each prompt, and saves the results to a JSONL file.
    """
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Using {args.tensor_parallel_size} GPU(s).")
    
    # Load the full data entries
    original_data = load_data_from_file(args.data_file_path)
    if not original_data:
        print("No data to process. Exiting.")
        return
    original_data = original_data[:args.max_num_samples]
    
    # Extract just the questions for prompt template creation
    prompts = [item['question'] for item in original_data]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Tokenize "</think>" to find the split point later
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    all_prompts_to_generate = []
    sys = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for item in original_data:
        prompt = item['question']
        base_template = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt+sys}],
            tokenize=False,
            add_generation_prompt=True
        )

        # Strategy 1: Normal
        normal_prompt = base_template + "<think>\nThis problem seems to be of normal difficulty. I will proceed with a step-by-step solution."
        all_prompts_to_generate.append({
            'original_data': item,
            'strategy': 'normal',
            'prompt': normal_prompt
        })

        # Strategy 2: Too Hard
        hard_prompt = base_template + "<think>\nThis problem appears to be quite challenging. To conserve budget, I will outline the key steps and avoid getting stuck in lengthy calculations."
        all_prompts_to_generate.append({
            'original_data': item,
            'strategy': 'too_hard',
            'prompt': hard_prompt
        })

        # Strategy 3: Too Easy
        easy_prompt = base_template + "<think>\nThis problem seems to be too easy. I will provide the solution directly.</think>"
        all_prompts_to_generate.append({
            'original_data': item,
            'strategy': 'too_easy',
            'prompt': easy_prompt
        })


    print(f"Prepared {len(all_prompts_to_generate)} prompts for generation ({len(original_data)} questions x 3 strategies).")
    
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )

    print(f"\nStarting inference with batch size {args.batch_size}, generating {args.n} samples per prompt...")
    
    with open(args.output_file_path, 'w', encoding='utf-8') as f:
        # Process data in batches
        for i in tqdm(range(0, len(all_prompts_to_generate), args.batch_size), desc="Processing Batches"):
            batch_items = all_prompts_to_generate[i:i+args.batch_size]
            batch_prompts = [item['prompt'] for item in batch_items]

            # Generate outputs for the current batch
            batch_outputs = llm.generate(batch_prompts, sampling_params)

            # Process and save outputs for the batch
            for j, output in enumerate(batch_outputs):
                original_item_info = batch_items[j]
                original_item = original_item_info['original_data']
                strategy = original_item_info['strategy']

                for sample_index, generated_sample in enumerate(output.outputs):
                    generated_text = generated_sample.text
                    logprobs = generated_sample.logprobs
                    token_ids = generated_sample.token_ids

                    think_logprobs = []
                    answer_logprobs = []
                    think_token_ids = []
                    answer_token_ids = []

                    if logprobs:
                        split_index = -1
                        # Search for the token sequence of '</think>'
                        if think_end_tokens: # Make sure it's not empty
                            for k in range(len(token_ids) - len(think_end_tokens) + 1):
                                if token_ids[k:k+len(think_end_tokens)] == think_end_tokens:
                                    split_index = k + len(think_end_tokens) - 1
                                    break
                        
                        if split_index != -1:
                            think_logprobs = logprobs[:split_index + 1]
                            answer_logprobs = logprobs[split_index + 1:]
                            think_token_ids = token_ids[:split_index + 1]
                            answer_token_ids = token_ids[split_index + 1:]
                        else:
                            # If '</think>' is not found, consider the whole generation as 'think' stage
                            think_logprobs = logprobs
                            answer_logprobs = []
                            think_token_ids = token_ids
                            answer_token_ids = []

                    think_total_entropy, think_average_entropy, num_think_tokens, think_token_entropies = calculate_entropy_from_logprobs(think_logprobs)
                    answer_total_entropy, answer_average_entropy, num_answer_tokens, answer_token_entropies = calculate_entropy_from_logprobs(answer_logprobs)

                    think_tokens_with_entropy = [
                        {"token": tokenizer.decode(token_id), "entropy": round(entropy, 4)}
                        for token_id, entropy in zip(think_token_ids, think_token_entropies)
                    ]
                    answer_tokens_with_entropy = [
                        {"token": tokenizer.decode(token_id), "entropy": round(entropy, 4)}
                        for token_id, entropy in zip(answer_token_ids, answer_token_entropies)
                    ]

                    total_entropy = think_total_entropy + answer_total_entropy
                    num_generated_tokens = num_think_tokens + num_answer_tokens
                    average_entropy = total_entropy / num_generated_tokens if num_generated_tokens > 0 else 0.0

                    result = {
                        'question': original_item.get('question'),
                        'final_answer': original_item.get('final_answer'),
                        'difficulty': original_item.get('difficulty'),
                        'strategy': strategy,
                        'sample_index': sample_index,
                        'generated_text': generated_text,
                        
                        'total_entropy': round(total_entropy, 4),
                        'average_entropy': round(average_entropy, 4),
                        'generated_tokens': num_generated_tokens,
                        
                        'think_total_entropy': round(think_total_entropy, 4),
                        'think_average_entropy': round(think_average_entropy, 4),
                        'think_tokens': num_think_tokens,
                        'think_tokens_with_entropy': think_tokens_with_entropy,

                        'answer_total_entropy': round(answer_total_entropy, 4),
                        'answer_average_entropy': round(answer_average_entropy, 4),
                        'answer_tokens': num_answer_tokens,
                        'answer_tokens_with_entropy': answer_tokens_with_entropy,
                    }
                    
                    f.write(json.dumps(result) + '\n')

    print(f"\nAll results have been saved to {args.output_file_path}.")


if __name__ == "__main__":
    main()
