"""
This script is for performing inference using a trained budget probe.
It combines probe prediction with conditional generation using vLLM.

Functionality:
1.  Load a dataset.
2.  Use a HiddenStatesExtractor to get embeddings from a base LLM.
3.  Use a trained probe (Linear, MLP, or Enhanced) to predict the required generation strategy (e.g., 'no_think', 'think', 'too_hard').
4.  Based on the predicted strategy, use a vLLM or HuggingFace generator to produce a final answer.
5.  Evaluate the results and save them.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from typing import Dict, Any, List, Tuple
import argparse
from tqdm import tqdm
from datasets import load_dataset
import re
import numpy as np
import gc
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter, defaultdict
import multiprocessing
import tempfile
import shutil
import math
import sys
from probe_utils import (
    FocalLoss,
    ClassBalancedLoss,
    S1TextDataset,
    EnhancedProbe,
    MLPProbe,
)

# Attempt to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

# Attempt to import xVerify
try:
    sys.path.append('/home/zhtang/workspace/xVerify')
    from src.xVerify.model import Model
    from src.xVerify.eval import Evaluator
    XVERIFY_AVAILABLE = True
except ImportError:
    Model = None
    Evaluator = None
    XVERIFY_AVAILABLE = False

# --- Utility functions from other files ---

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_answer(text: str) -> str:
    # Placeholder for answer extraction logic
    match = re.search(r'\\boxed\{(.*)\}', text)
    return match.group(1) if match else ""

def check_is_correct(pred, gt):
    # Placeholder for correctness checking
    return str(pred).strip() == str(gt).strip() if gt is not None else False

def load_data(dataset_name: str, split: str, data_dir: str) -> List[Dict]:
    file_path = os.path.join(data_dir, dataset_name, f"{split}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return list(load_jsonl(file_path))


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
        if not token_logprobs:
            continue
        step_logprobs = [p.logprob for p in token_logprobs.values()]
        step_probs = np.exp(step_logprobs)
        step_probs /= np.sum(step_probs)
        # Calculate entropy for this token step
        step_entropy = -np.sum(step_probs * np.log(step_probs))
        token_entropies.append(step_entropy)
        total_entropy += step_entropy
    
    average_entropy = total_entropy / len(token_entropies) if token_entropies else 0.0
    return total_entropy, average_entropy, num_tokens, token_entropies


# --- 2. Hidden States Extractor ---

class HiddenStatesExtractor:
    """Extracts hidden states from a model for a batch of texts."""
    def __init__(self, model_path: str, layer_idx: int = -1, device=None):
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model from {model_path}...")
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.layer_idx = layer_idx
        self.hidden_dim = self.model.config.hidden_size
        print(f"Model loaded. Hidden dimension: {self.hidden_dim}, Device: {self.device}")

    def create_prompt_for_probe(self, question: str) -> str:
        """Applies chat template and adds the <think> token."""
        template = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question+"\n\nLet's reason step by step, and put your final answer within \boxed{}."}
             ],
            tokenize=False,
            add_generation_prompt=True
        )
        template += "<think>\n\nI need to analyze the difficulty of the problem first, and then give a budget control strategy. I think the difficulty of this question is"
        return template

    def extract_batch_hidden_states(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Tokenizes texts, gets hidden states, and returns the last token's embedding."""
        prompts = [self.create_prompt_for_probe(text) for text in texts]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer_idx]
            # For left-padded sequences, the last token is at the end.
            last_token_hidden_states = hidden_states[:, -1, :]
            return last_token_hidden_states.cpu().to(torch.float32)

# --- 3. Probe Classifier ---

class ProbeClassifier:
    """Uses a trained probe to classify texts."""
    def __init__(self, probe_path: str, extractor: HiddenStatesExtractor, probe_type: str = 'mlp', num_classes: int = 4):
        self.extractor = extractor
        print(f"Loading {num_classes}-class {probe_type} probe from {probe_path}...")

        probe_map = {
            'mlp': MLPProbe,
            'enhanced': EnhancedProbe
        }
        if probe_type not in probe_map:
            raise ValueError(f"Unknown probe_type: {probe_type}. Supported types are {list(probe_map.keys())}")

        self.probe = probe_map[probe_type](extractor.hidden_dim, num_classes)
        self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        self.probe.eval()
        print("Probe classifier loaded successfully!")

    def predict_batch(self, texts: List[str], max_length: int = 512,
                        too_easy_bonus: float = 0.0, normal_penalty: float = 0.0, too_hard_penalty: float = 0.0) -> List[Dict[str, Any]]:
        """Performs batch prediction."""
        hidden_states = self.extractor.extract_batch_hidden_states(texts, max_length)
        results = []

        with torch.no_grad():
            outputs = self.probe(hidden_states)

            # Adjust logits based on penalties and bonuses to influence strategy selection.
            if outputs.shape[1] > 2:
                outputs[:, 0] += too_easy_bonus
                outputs[:, 1] -= normal_penalty
                outputs[:, 2] -= too_hard_penalty

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Corresponds to easy, medium, hard, fail
            strategy_map = {
                0: "too_easy",
                1: "normal",
                2: "too_hard"
            }

            for i in range(len(texts)):
                prediction = preds[i]
                confidence = float(probs[i][prediction])
                strategy = strategy_map.get(prediction, "normal") # Default to 'normal'

                results.append({
                    'final_prediction': int(prediction),
                    'strategy': strategy,
                    'final_confidence': confidence,
                    'probabilities': probs[i].tolist()
                })
        return results

# --- 4. Generation Solvers ---

class VLLMMathProblemSolver:
    """Math problem solver using vLLM for fast generation."""
    def __init__(self, model_path: str, max_length: int = 2048, tp_size: int = 1, prompt_pool_config: Dict[str, int] = None):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Please install it via 'pip install vllm'")

        self.model_path = model_path
        self.max_model_len = max_length
        print(f"Initializing vLLM Engine with model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.engine = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.9,
            max_model_len=max_length,
            tensor_parallel_size=tp_size,
        )
        
        # Initialize prompt pools
        self.prompt_pools = self._initialize_prompt_pools()
        self.prompt_config = prompt_pool_config or {"normal": 0, "too_hard": 0, "too_easy": 0}
        print("vLLM Engine initialized successfully")
        print(f"Prompt pool configuration: {self.prompt_config}")
    
    def _initialize_prompt_pools(self):
        """
        Initialize optimized prompt pools V2 based on experimental results.
        
        V2 improvements:
        - Enhanced successful mixed strategy patterns from V1 experiments
        - Better reasoning guidance for cross-dataset performance  
        - Optimized balance between accuracy and efficiency
        - Based on analysis of 92% MATH, 93.5% GSM8K, 66.7% AIME performance
        """
        return {
            "normal": [
                # 0: Conservative but systematic (best for GSM8K: 93.5%, good for MATH)
                "<think>\nLet me work through this systematically, ensuring I understand each step before proceeding.",
                
                # 1: Structured approach (enhanced from V1 good performers)  
                "<think>\nI'll break this down into clear, logical steps and solve methodically.",
                
                # 2: Analytical focus (balanced approach)
                "<think>\nLet me analyze the problem structure and identify the most direct solution path.",
                
                # 3: Efficient systematic (inspired by (3,1,4) success: 78.7% avg)
                "<think>\nI need to approach this efficiently while maintaining accuracy. Let me identify the key steps.",
                
                # 4: Direct but thorough (aggressive but careful)
                "<think>\nI can see the solution approach. Let me work through this directly but verify each step."
            ],
            "too_hard": [
                # 0: Conservative resource management
                "<think>\nThis is complex. I'll focus on the essential approach and avoid getting stuck in lengthy details.",
                
                # 1: Strategic efficiency (enhanced from mixed_efficient success)
                "<think>\nThis looks challenging. Let me identify the core strategy and key insights to solve efficiently.",
                
                # 2: Balanced complexity handling
                "<think>\nThis appears intricate. I'll outline the main method while being mindful of computational resources.",
                
                # 3: Insight-focused approach
                "<think>\nThis seems demanding. Let me focus on the fundamental insights and critical breakthrough points.",
                
                # 4: Aggressive efficiency (best for AIME, inspired by (0,4,2) success: 92% MATH)
                "<think>\nThis is sophisticated. I'll sketch the essential solution path and key steps without extensive computation."
            ],
            "too_easy": [
                # 0: Quick but verified (best for GSM8K: 93.5%)
                "<think>\nThis looks straightforward. Let me solve it directly while double-checking my approach.</think>",
                
                # 1: Efficient with validation (inspired by (2,4,1) efficiency)
                "<think>\nI can see the solution immediately. Let me apply it efficiently and verify the result.</think>",
                
                # 2: Streamlined approach
                "<think>\nThe method is clear. I'll execute this directly with a quick accuracy check.</think>",
                
                # 3: Confident but careful
                "<think>\nThis has an obvious solution path. Let me implement it while ensuring correctness.</think>",
                
                # 4: Direct execution (best for AIME: 66.7%, aggressive approach)
                "<think>\nI can identify the solution approach immediately. Let me execute it directly.</think>"
            ]
        }

    def batch_generate_responses(self, questions: List[str], strategies: List[str], 
                                 n_runs: int = 1,
                                 system_prompt: str = None,
                                 max_tokens: int = 2048, temperature: float = 0.6, top_p: float = 0.9,
                                 continuation_prompt: str = "\nNow the correct answer is \\boxed{",
                                 continuation_max_tokens: int = 128,
                                 enable_continuation: bool = True) -> List[List[Dict[str, Any]]]:
        """Generates responses for a batch of questions based on individual strategies."""

        # --- Define Sampling Strategies ---
        normal_params = SamplingParams(n=n_runs, temperature=0.8, top_p=0.95, max_tokens=max_tokens, logprobs=10)
        # limited_params = SamplingParams(n=n_runs, temperature=temperature, top_p=top_p, max_tokens=max_tokens // 4, logprobs=10)
        limited_params = SamplingParams(n=n_runs, temperature=0.5, max_tokens=int(max_tokens*0.4) , logprobs=10)
        hard_params = SamplingParams(n=n_runs, temperature=0.4, max_tokens=int(max_tokens*0.5) , logprobs=10 )
        # hard_params = SamplingParams(n=n_runs, temperature=temperature, top_p=top_p, max_tokens=max_tokens // 2, logprobs=10, repetition_penalty=1.1)
        continuation_params = SamplingParams(n=1, temperature=0.6, top_p=1.0, max_tokens=continuation_max_tokens, stop=["}"], logprobs=10)

        # --- First Generation Step ---
        first_pass_prompts = []
        first_pass_params = []
        original_indices = list(range(len(questions)))

        for question, strategy in zip(questions, strategies):
            messages = [{"role": "user", "content": question + "\n\nLet's reason step by step, and put your final answer within \\boxed{}."}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            template = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False, add_generation_prompt=True
            )
            
            # Select prompt from pool based on strategy and configuration
            if strategy in self.prompt_pools and strategy in self.prompt_config:
                pool_idx = self.prompt_config[strategy]
                selected_prompt = self.prompt_pools[strategy][pool_idx]
                prompt = template + selected_prompt
            else:
                # Fallback to original prompts if strategy not found
                if strategy == "simple_baseline":
                    # For simple baseline, use only the basic template without any <think> additions
                    prompt = template
                elif strategy == "normal":
                    prompt = template + "<think>\nThis problem seems to be of normal difficulty. I will proceed with a step-by-step solution." 
                elif strategy == "too_hard":
                    prompt = template + "<think>\nThis problem appears to be quite challenging. To conserve budget, I will outline the key steps and avoid getting stuck in lengthy calculations."
                else:  # too_easy
                    prompt = template + "<think>\nThis problem seems to be too easy. I will provide the solution directly.</think>"
            
            # Set generation parameters based on strategy
            if strategy == "simple_baseline":
                params = normal_params  # Use normal parameters (temp=0.6) for simple baseline
            elif strategy == "normal":
                params = normal_params
            elif strategy == "too_hard":
                params = hard_params
            else:  # too_easy
                params = limited_params
            
            first_pass_prompts.append(prompt)
            first_pass_params.append(params)

        first_pass_outputs = self.engine.generate(first_pass_prompts, use_tqdm=False, sampling_params=first_pass_params)

        # --- Process and Handle Truncated Outputs ---
        final_results = {}
        continuation_prompts = []
        continuation_indices = []
        continuation_contexts = []

        max_model_len = self.max_model_len

        for i, output in enumerate(first_pass_outputs):
            original_idx = original_indices[i]
            
            # Since we now generate `n_runs` samples per prompt, we process each one.
            for sample_idx, generated_sample in enumerate(output.outputs):
                result_key = (original_idx, sample_idx)

                if generated_sample.finish_reason == 'length' and enable_continuation:
                    continuation_prompt_text = output.prompt + generated_sample.text + continuation_prompt

                    prompt_token_ids = self.tokenizer.encode(continuation_prompt_text)
                    new_max_len = len(prompt_token_ids)

                    if len(prompt_token_ids) >= max_model_len:
                        # Truncate from the left to keep the most recent context
                        # Leave space for continuation_max_tokens
                        new_max_len = max_model_len - continuation_params.max_tokens - 5  # 5 for safety

                        if new_max_len <= 0:
                            # Not enough space for continuation, so we can't do the second step.
                            # Use the first-pass output as is.
                            print(f"Warning: Cannot perform continuation for sample {original_idx} because prompt is too long after first generation.")
                            response_text = "<think>" + output.prompt.split("<think>")[-1] + generated_sample.text
                            total_entropy, average_entropy, _, _ = calculate_entropy_from_logprobs(generated_sample.logprobs)
                            result_details = {
                                'response': response_text,
                                'total_generated_tokens': len(generated_sample.token_ids),
                                'total_entropy': total_entropy,
                                'average_entropy': average_entropy
                            }
                            final_results[result_key] = result_details
                            continue  # Skip to next item in loop

                    truncated_token_ids = prompt_token_ids[-new_max_len:]
                    continuation_prompt_text = self.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
                
                    continuation_prompts.append(continuation_prompt_text)
                    continuation_indices.append(result_key)
                    continuation_contexts.append(generated_sample)
                elif generated_sample.finish_reason == 'length' and not enable_continuation:
                    # When continuation is disabled, just use the truncated output as is
                    response_text = "<think>" + output.prompt.split("<think>")[-1] + generated_sample.text
                    total_entropy, average_entropy, _, _ = calculate_entropy_from_logprobs(generated_sample.logprobs)
                    result_details = {
                        'response': response_text,
                        'total_generated_tokens': len(generated_sample.token_ids),
                        'total_entropy': total_entropy,
                        'average_entropy': average_entropy
                    }
                    final_results[result_key] = result_details
                else:
                    response_text = "<think>" + output.prompt.split("<think>")[-1] + generated_sample.text
                    total_entropy, average_entropy, _, _ = calculate_entropy_from_logprobs(generated_sample.logprobs)
                    result_details = {
                        'response': response_text,
                        'total_generated_tokens': len(generated_sample.token_ids),
                        'total_entropy': total_entropy,
                        'average_entropy': average_entropy
                    }
                    final_results[result_key] = result_details
        
        # --- Second Generation Step (if needed) ---
        if continuation_prompts:
            continuation_outputs = self.engine.generate(continuation_prompts, continuation_params, use_tqdm=False)
            
            for i, second_gen_output in enumerate(continuation_outputs):
                result_key = continuation_indices[i]
                original_idx, sample_idx = result_key
                
                # Find the corresponding first-pass output object
                first_gen_output_obj = first_pass_outputs[original_idx]
                first_sample = continuation_contexts[i]
                second_sample = second_gen_output.outputs[0]

                combined_text = first_sample.text + continuation_prompt + second_sample.text
                if not combined_text.endswith("}"):
                    combined_text += "}"

                final_response = "<think>" + first_gen_output_obj.prompt.split("<think>")[-1] + combined_text
                
                # Entropy calculation for two-part generation
                first_total_entropy, _, first_num_tokens, _ = calculate_entropy_from_logprobs(first_sample.logprobs)
                second_total_entropy, _, second_num_tokens, _ = calculate_entropy_from_logprobs(second_sample.logprobs)
                
                total_entropy = first_total_entropy + second_total_entropy
                
                # Calculate total tokens carefully
                continuation_tokens_count = len(self.tokenizer.encode(continuation_prompt, add_special_tokens=False))
                total_tokens = first_num_tokens + continuation_tokens_count + second_num_tokens
                
                average_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0

                result_details = {
                    'response': final_response,
                    'total_generated_tokens': total_tokens,
                    'total_entropy': total_entropy,
                    'average_entropy': average_entropy
                }
                final_results[result_key] = result_details

        # Reformat results to be a list of lists
        output_results = [[] for _ in range(len(questions))]
        for (original_idx, sample_idx), result in final_results.items():
            output_results[original_idx].append(result)
            
        return output_results

    def shutdown(self):
        """Shuts down the vLLM engine."""
        if hasattr(self, 'engine'):
            del self.engine
            self.engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("vLLM engine shutdown.")

class HFMathProblemSolver:
    """Fallback solver using HuggingFace's generate method."""
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.eval()

    def generate_response(self, question: str, strategy: str = "normal", max_length: int = 2048, temperature: float = 0.0, system_prompt: str = None) -> Tuple[str, Dict[str, int]]:
        messages = [
            {"role": "user", "content": question + "\n\nLet's reason step by step, and put your final answer within \\boxed{}."}
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )

        if strategy == "simple_baseline":
            # For simple baseline, use only the basic template without any <think> additions
            prompt = template
        elif strategy == "think":
            prompt = template + "<think>"
        elif strategy == "too_hard":
            prompt = template + "<think>This problem is too difficult for me. I will give up thinking to save tokens.</think>"
        else: # no_think
            prompt = template + "<think>Okay, I think I have finished thinking.</think>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_length, "pad_token_id": self.tokenizer.pad_token_id}
            if temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_part = "<think>" + response.split("<think>")[-1]
        token_stats = {'total_generated_tokens': outputs.shape[1] - input_length}
        return response_part, token_stats

    def shutdown(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# --- 5. Main Workflow ---

def parse_args():
    parser = argparse.ArgumentParser(description="Budget Probe Inference with Optional vLLM Generation")
    # Model and Probe paths
    parser.add_argument("--llm_model", type=str, required=True, help="Path to base LLM model for feature extraction and generation.")
    parser.add_argument("--probe_path", type=str, default=None, help="Path to the trained probe model. Required if not in baseline mode.")
    parser.add_argument("--probe_type", type=str, default="mlp", choices=["mlp", "enhanced"], help="Type of probe architecture.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of output classes for the probe.")
    parser.add_argument("--extract_layer", type=int, default=-1, help="Layer index to extract hidden states from.")

    # Strategy adjustment arguments
    parser.add_argument("--too_easy_bonus", type=float, default=0.0, help="Bonus to add to the 'too_easy' logit to increase its selection frequency.")
    parser.add_argument("--normal_penalty", type=float, default=0.0, help="Penalty to subtract from the 'normal' logit to reduce its selection frequency.")
    parser.add_argument("--too_hard_penalty", type=float, default=0.0, help="Penalty to subtract from the 'too_hard' logit to reduce its selection frequency.")

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="deepmath", help="Dataset name (e.g., gsm8k, math, deepmath).")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for datasets like 'math'.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a specific .jsonl data file (overrides other data args).")

    # Execution arguments
    parser.add_argument("--output_path", type=str, default="./results/budget_probe_results.json", help="Path to save results.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation (if available).")
    parser.add_argument("--baseline_mode", action="store_true", help="Run in baseline mode, using 'normal' strategy for all samples.")
    parser.add_argument("--full_baseline_mode", action="store_true", help="Run in full baseline mode, using all three strategies for each sample.")
    parser.add_argument("--tale_ep_baseline", action="store_true", help="Run in TALE-EP baseline mode.")
    parser.add_argument("--concise_cot_baseline", action="store_true", help="Run in ConciseCOT baseline mode.")
    parser.add_argument("--random_baseline_mode", action="store_true", help="Run in random baseline mode, using random strategy for each sample.")
    parser.add_argument("--only_probe_prediction", action="store_true", help="Only run probe predictions and skip generation.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length for tokenizer.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation.")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--stage2_n_runs", type=int, default=1, help="Number of runs for each problem in stage 2 for stability.")

    # Continuation prompt settings
    parser.add_argument("--continuation_prompt", type=str, default="\\nNow the correct answer is \\boxed{", 
                        help="Prompt to use for the second generation step if the first is truncated.")
    parser.add_argument("--continuation_max_tokens", type=int, default=1024,
                        help="Max tokens for the continuation generation step.")
    parser.add_argument("--offline_entropy_selection", action="store_true", help="Enable offline entropy selection for vLLM generation.")
    
    # Prompt pool settings
    parser.add_argument("--prompt_normal_idx", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Index of prompt to use for 'normal' strategy (0-4).")
    parser.add_argument("--prompt_too_hard_idx", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Index of prompt to use for 'too_hard' strategy (0-4).")
    parser.add_argument("--prompt_too_easy_idx", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Index of prompt to use for 'too_easy' strategy (0-4).")
    parser.add_argument("--search_prompts", action="store_true", 
                        help="Enable prompt search mode: test all prompt combinations and save results with prompt info.")

    return parser.parse_args()


def run_probe_predictions(args, data, classifier):
    """Run probe predictions over the dataset."""
    probe_results = []
    print("Running probe predictions...")
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch_data = data[i:i + args.batch_size]
        batch_questions = [item['problem'] for item in batch_data]
        batch_results = classifier.predict_batch(
            batch_questions,
            max_length=args.max_length,
            too_easy_bonus=args.too_easy_bonus,
            normal_penalty=args.normal_penalty,
            too_hard_penalty=args.too_hard_penalty
        )

        for j, (data_item, result) in enumerate(zip(batch_data, batch_results)):
            result['data_index'] = i + j
            result['problem'] = data_item['problem']
            result['ground_truth_answer'] = data_item.get('answer')
            if 'difficulty' in data_item:
                result['ground_truth_difficulty'] = data_item['difficulty']
            probe_results.append(result)
    # print probe results statistics
    probe_results_df = pd.DataFrame(probe_results)
    print(probe_results_df.groupby('strategy').size())
    return probe_results

def run_generation_stage(args, probe_results, is_baseline_mode=False):
    """Run the generation stage based on probe predictions."""
    print("Loading math problem solver for generation...")
    
    # Prepare prompt pool configuration
    prompt_config = {
        "normal": args.prompt_normal_idx,
        "too_hard": args.prompt_too_hard_idx,
        "too_easy": args.prompt_too_easy_idx
    }
    
    if args.use_vllm and VLLM_AVAILABLE:
        solver = VLLMMathProblemSolver(args.llm_model, args.max_length, args.tp_size, prompt_config)
    else:
        if args.use_vllm:
            print("Warning: vLLM not available, falling back to HuggingFace generator.")
        solver = HFMathProblemSolver(args.llm_model)

    system_prompt = None
    if args.baseline_mode:
        system_prompt = None  # Use no system prompt for simple baseline
    elif args.tale_ep_baseline:
        system_prompt = "You are a helpful and harmless assistant. You should think step-by-step but use as few tokens burgets as possible without compromising performance."
    elif args.concise_cot_baseline:
        system_prompt = "You are a helpful and harmless assistant. You should think step-by-step. Please be concise throughout the reasoning process."
    else:
        system_prompt = "You are a helpful and harmless assistant. You should think step-by-step. Please be concise throughout the reasoning process."

    final_results = []
    print(f"\nProcessing {len(probe_results)} samples...")
    for i in tqdm(range(0, len(probe_results), args.batch_size)):
        batch_chunk = probe_results[i:i + args.batch_size]
        batch_questions = [res['problem'] for res in batch_chunk]
        batch_strategies = [res['strategy'] for res in batch_chunk]

        if isinstance(solver, VLLMMathProblemSolver):
            # Disable continuation for baseline modes, enable for probe modes
            enable_continuation = not is_baseline_mode
            batch_responses_list = solver.batch_generate_responses(
                batch_questions, 
                strategies=batch_strategies,
                n_runs=args.stage2_n_runs,
                max_tokens=args.max_new_tokens,
                continuation_prompt=args.continuation_prompt,
                continuation_max_tokens=args.continuation_max_tokens,
                system_prompt=system_prompt,
                enable_continuation=enable_continuation
            )
        else: # Fallback to single generation
            # Note: HF fallback does not support n_runs > 1 for simplicity.
            batch_responses_list = []
            for q, s in zip(batch_questions, batch_strategies):
                temp = 0.6 if s == "simple_baseline" else 0.0
                batch_responses_list.append([solver.generate_response(q, s, max_length=args.max_new_tokens, temperature=temp, system_prompt=system_prompt)])

        for j, responses in enumerate(batch_responses_list):
            probe_res = batch_chunk[j]
            # Store all responses for this problem
            for run_idx, response_details in enumerate(responses):
                result_item = {
                    **probe_res,
                    'run_id': run_idx,
                    **response_details
                }
                # Add prompt configuration info if using vLLM with prompt pools
                if isinstance(solver, VLLMMathProblemSolver) and hasattr(solver, 'prompt_config'):
                    result_item['prompt_config'] = solver.prompt_config.copy()
                final_results.append(result_item)


    if hasattr(solver, 'shutdown'):
        solver.shutdown()
    
    # Filter out any None entries if some batches failed
    processed_results = [res for res in final_results if res is not None]
    return {'results': processed_results}


def save_results(args, data_list):
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure the output is .jsonl for multiple runs
    if args.stage2_n_runs > 1 and not args.output_path.endswith('.jsonl'):
        print("Warning: Outputting multiple runs. Forcing output file to .jsonl format.")
        base, _ = os.path.splitext(args.output_path)
        args.output_path = base + ".jsonl"

    print(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        if isinstance(data_list, list):
            for item in data_list:
                f.write(json.dumps(item) + '\n')
        elif isinstance(data_list, dict) and 'results' in data_list:
             for item in data_list['results']:
                f.write(json.dumps(item) + '\n')
        else:
            # Fallback for old format, though the new logic should prevent this.
            json.dump(data_list, f, indent=2, ensure_ascii=False)

def run_offline_entropy_selection(args, data):
    """
    Runs generation for all strategies, then selects the best response based on entropy and length.
    """
    print("Running in offline entropy selection mode.")
    
    # 1. Prepare data for all strategies, similar to full_baseline_mode
    probe_output_results = []
    strategies = ["too_easy", "normal", "too_hard"]
    for i, item in enumerate(data):
        for strategy in strategies:
            probe_output_results.append({
                'data_index': i,
                'problem': item['problem'],
                'ground_truth_answer': item.get('answer'),
                'strategy': strategy,
            })
            
    # 2. Run generation for all prepared items
    # Offline entropy selection uses all strategies, enable continuation
    all_generated_results = run_generation_stage(args, probe_output_results, is_baseline_mode=False)['results']
    
    # 3. Group results by problem
    grouped_by_problem = {}
    for result in all_generated_results:
        idx = result['data_index']
        if idx not in grouped_by_problem:
            grouped_by_problem[idx] = []
        grouped_by_problem[idx].append(result)
        
    # 4. Select the best result for each problem
    final_selection = []
    for idx, candidates in grouped_by_problem.items():
        # Sort by average entropy (lower is better), then by generated tokens (lower is better)
        best_candidate = sorted(candidates, key=lambda x: (x['average_entropy'], x['total_generated_tokens']))[0]
        final_selection.append(best_candidate)
        
    print(f"Selected {len(final_selection)} best responses from {len(all_generated_results)} candidates.")
    return final_selection

def run_prompt_search(args, probe_results):
    """
    Run prompt search: test all combinations of prompts and save results.
    This will generate 5^3 = 125 different prompt combinations.
    """
    print("Running prompt search mode...")
    import itertools
    
    all_search_results = []
    prompt_combinations = list(itertools.product(range(5), range(5), range(5)))
    
    print(f"Testing {len(prompt_combinations)} prompt combinations...")
    
    for combo_idx, (normal_idx, too_hard_idx, too_easy_idx) in enumerate(prompt_combinations):
        print(f"Testing combination {combo_idx + 1}/{len(prompt_combinations)}: "
              f"normal={normal_idx}, too_hard={too_hard_idx}, too_easy={too_easy_idx}")
        
        # Create temporary args with current prompt configuration
        temp_args = argparse.Namespace(**vars(args))
        temp_args.prompt_normal_idx = normal_idx
        temp_args.prompt_too_hard_idx = too_hard_idx
        temp_args.prompt_too_easy_idx = too_easy_idx
        
        # Run generation with current prompt combination
        # Prompt search is typically used with probe mode, so enable continuation
        combo_results = run_generation_stage(temp_args, probe_results, is_baseline_mode=False)['results']
        
        # Add combination info to each result
        for result in combo_results:
            result['prompt_combination_id'] = combo_idx
            result['prompt_combination'] = {
                'normal': normal_idx,
                'too_hard': too_hard_idx,
                'too_easy': too_easy_idx
            }
        
        all_search_results.extend(combo_results)
    
    print(f"Prompt search completed. Generated {len(all_search_results)} results.")
    return all_search_results

def run_probe_stage(args, data, output_file):
    """
    This function runs the entire probe prediction stage in a self-contained manner.
    It loads the necessary models, performs predictions, and saves the results to a file.
    This is designed to be run in a separate process to ensure all memory is released upon completion.
    """
    try:
        print("[Probe Process] Initializing extractor and probe classifier...")
        extractor = HiddenStatesExtractor(args.llm_model, args.extract_layer)
        classifier = ProbeClassifier(args.probe_path, extractor, args.probe_type, args.num_classes)
        
        probe_output_results = run_probe_predictions(args, data, classifier)
        
        probe_output = {'results': probe_output_results}
        
        print(f"[Probe Process] Saving probe results to temporary file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(probe_output, f)
            
        print("[Probe Process] Probe stage finished.")

    except Exception as e:
        print(f"[Probe Process] An error occurred in the probe stage: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Explicitly delete models and clear cache to be thorough, although process exit is the main cleanup
        del extractor.model, extractor, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[Probe Process] Cleaned up resources.")


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

def run_bench_evaluation(input_file: str, model_path: str = "/home/zhtang/hf_models/IAAR-Shanghai/xVerify-0.5B-I", pass_k: str = "1,3") -> Dict[str, Any]:
    """
    Run bench evaluation on the generated results.
    Returns evaluation statistics.
    """
    if not XVERIFY_AVAILABLE:
        print("Warning: xVerify not available, skipping bench evaluation.")
        return {}
    
    print(f"Running bench evaluation on {input_file}")
    
    temp_dir = None
    try:
        # Initialize the xVerify Model and Evaluator
        print(f"Initializing xVerify evaluator with model from: {model_path}")
        model = Model(
            model_name='xVerify-0.5B-I',
            model_path_or_url=model_path,
            inference_mode='local',
            api_key=None
        )
        evaluator = Evaluator(model=model)
        print("Evaluator initialized successfully.")

        # Load original data
        original_data = []
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                original_data.append(json.loads(line))
        
        if not original_data:
            print("No data to evaluate.")
            return {}

        print(f"Loaded {len(original_data)} entries for evaluation.")

        # Prepare data for evaluation
        formatted_data_for_eval = []
        for item in original_data:
            generated_text = item.get('response', '')
            if '</think>' in generated_text:
                llm_output = generated_text.split('</think>', 1)[-1].strip()
            else:
                llm_output = generated_text[-1000:] if len(generated_text) > 1000 else generated_text
            
            question = item.get('problem', '')
            correct_answer_raw = item.get('ground_truth_answer', '')
            correct_answer = extract_gsm8k_answer(correct_answer_raw)

            formatted_data_for_eval.append({
                "question": question,
                "llm_output": llm_output,
                "correct_answer": correct_answer
            })

        # Create temporary files for evaluation
        temp_dir = tempfile.mkdtemp()
        temp_input_path = os.path.join(temp_dir, "eval_input.json")
        temp_output_dir_path = os.path.join(temp_dir, "eval_output")
        
        with open(temp_input_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data_for_eval, f, indent=4)
        
        print(f"Prepared temporary data for batch evaluation at: {temp_input_path}")

        # Run the batch evaluation
        print(f"Starting batch evaluation...")
        evaluator.evaluate(
            data_path=temp_input_path,
            data_size=len(formatted_data_for_eval),
            output_path=temp_output_dir_path
        )
        print("Batch evaluation complete.")

        # Parse results
        result_files = [f for f in os.listdir(temp_output_dir_path) if f.endswith('.json')]
        if not result_files:
            raise FileNotFoundError("No result file found in the xVerify output directory.")
        
        actual_output_file_path = os.path.join(temp_output_dir_path, result_files[0])
        print(f"Found and reading evaluation results from: {actual_output_file_path}")

        with open(actual_output_file_path, 'r', encoding='utf-8') as f_eval:
            eval_data_dict = json.load(f_eval)
            eval_results_list = eval_data_dict.get('results', [])

        if len(original_data) != len(eval_results_list):
            print(f"Warning: Mismatch between processed items ({len(original_data)}) and evaluated items ({len(eval_results_list)}). Merging based on index.")
        
        print(f"Merging {len(eval_results_list)} evaluation results with original data...")
        
        # Merge results and calculate statistics
        k_values = [int(k) for k in pass_k.split(',')]
        
        # Group results by problem
        problems = defaultdict(list)
        
        for i, original in enumerate(original_data):
            if i >= len(eval_results_list):
                break
            
            evaluation = eval_results_list[i]
            judgment_key = f"{model.model_name}_judgment_result"
            judgment_result = evaluation.get(judgment_key)

            correctness = None
            if isinstance(judgment_result, dict):
                correctness = judgment_result.get('Correctness')
            elif isinstance(judgment_result, str):
                correctness = judgment_result

            # Add evaluation result to original data
            original['xverify_evaluation'] = {
                'Correctness': correctness,
            }
            
            problems[original['problem']].append(original)
        
        # Calculate statistics
        strategy_stats = defaultdict(lambda: {'problems': defaultdict(list)})
        
        for problem, runs in problems.items():
            strategy = runs[0]['strategy']
            strategy_stats[strategy]['problems'][problem].extend(runs)

        print("\n--- Bench Evaluation Results ---")
        
        overall_problem_count = len(problems)
        overall_pass_at_k = {k: 0 for k in k_values}
        overall_avg_tokens = 0
        overall_total_correct_runs = 0
        overall_total_runs = 0
        
        results_summary = {}

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
            
            strategy_avg_pass_1 = (strategy_total_correct_runs / strategy_total_runs) * 100 if strategy_total_runs > 0 else 0
            
            print(f"Strategy: {strategy} ({num_problems} problems)")
            print(f"  Avg Pass@1: {strategy_avg_pass_1:.2f}%")
            
            strategy_results = {
                'num_problems': num_problems,
                'avg_pass_1': strategy_avg_pass_1,
                'avg_tokens_per_problem': avg_tokens_per_problem,
                'avg_tokens_per_sample': avg_tokens_per_sample,
                'pass_at_k': {}
            }
            
            for k in k_values:
                pass_k_avg = (strategy_pass_at_k[k] / num_problems) * 100 if num_problems > 0 else 0
                print(f"  Pass@{k}: {pass_k_avg:.2f}%")
                strategy_results['pass_at_k'][k] = pass_k_avg
                overall_pass_at_k[k] += strategy_pass_at_k[k]
            
            print(f"  Average Tokens per Problem: {avg_tokens_per_problem:.2f}")
            print(f"  Average Tokens per Sample: {avg_tokens_per_sample:.2f}")
            overall_avg_tokens += strategy_total_tokens
            
            results_summary[strategy] = strategy_results

        print("\n--- Overall Statistics ---")
        overall_avg_pass_1 = (overall_total_correct_runs / overall_total_runs) * 100 if overall_total_runs > 0 else 0
        print(f"Overall Avg Pass@1: {overall_avg_pass_1:.2f}%")
        
        overall_results = {
            'avg_pass_1': overall_avg_pass_1,
            'pass_at_k': {},
            'avg_tokens_per_problem': 0,
            'avg_tokens_per_sample': 0
        }
        
        for k in k_values:
            overall_pass_k_avg = (overall_pass_at_k[k] / overall_problem_count) * 100 if overall_problem_count > 0 else 0
            print(f"Overall Pass@{k}: {overall_pass_k_avg:.2f}%")
            overall_results['pass_at_k'][k] = overall_pass_k_avg

        grand_total_avg_tokens = overall_avg_tokens / overall_problem_count if overall_problem_count > 0 else 0
        print(f"Overall Average Tokens per Problem: {grand_total_avg_tokens:.2f}")
        overall_results['avg_tokens_per_problem'] = grand_total_avg_tokens

        grand_total_avg_tokens_per_sample = overall_avg_tokens / overall_total_runs if overall_total_runs > 0 else 0
        print(f"Overall Average Tokens per Sample: {grand_total_avg_tokens_per_sample:.2f}")
        overall_results['avg_tokens_per_sample'] = grand_total_avg_tokens_per_sample
        
        return {
            'overall': overall_results,
            'strategies': results_summary
        }

    except Exception as e:
        print(f"An error occurred during bench evaluation: {e}")
        return {}
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.baseline_mode and args.only_probe_prediction:
        raise ValueError("Cannot use --baseline_mode and --only_probe_prediction simultaneously.")
    if args.full_baseline_mode and (args.baseline_mode or args.only_probe_prediction):
        raise ValueError("Cannot use --full_baseline_mode with --baseline_mode or --only_probe_prediction.")
    if args.tale_ep_baseline and (args.baseline_mode or args.full_baseline_mode or args.only_probe_prediction or args.concise_cot_baseline):
        raise ValueError("Cannot use --tale_ep_baseline with other baseline modes.")
    if args.concise_cot_baseline and (args.baseline_mode or args.full_baseline_mode or args.only_probe_prediction or args.tale_ep_baseline):
        raise ValueError("Cannot use --concise_cot_baseline with other baseline modes.")
    if args.random_baseline_mode and (args.baseline_mode or args.full_baseline_mode or args.only_probe_prediction or args.tale_ep_baseline or args.concise_cot_baseline):
        raise ValueError("Cannot use --random_baseline_mode with other baseline modes.")
    if args.offline_entropy_selection and not args.use_vllm:
        raise ValueError("--offline_entropy_selection requires --use_vllm.")
    if args.offline_entropy_selection and (args.baseline_mode or args.full_baseline_mode or args.only_probe_prediction or args.tale_ep_baseline or args.concise_cot_baseline or args.random_baseline_mode):
        raise ValueError("Cannot use --offline_entropy_selection with other baseline modes.")
    if not args.baseline_mode and not args.full_baseline_mode and not args.tale_ep_baseline and not args.concise_cot_baseline and not args.random_baseline_mode and not args.probe_path and not args.offline_entropy_selection:
        raise ValueError("--probe_path is required when not in a baseline mode.")


    try:
        # Load Data
        print("Loading dataset...")
        if args.input_file:
            data = list(load_jsonl(args.input_file))
        elif args.dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main")
            data = [{'problem': item['question'], 'answer': item['answer']} for item in dataset[args.dataset_split]]
        else:
            data = load_data(args.dataset_name, args.dataset_split, args.data_dir)
        data = data[:min(len(data), args.max_samples)]
        print(f"Processing {len(data)} samples.")

        if args.baseline_mode or args.tale_ep_baseline or args.concise_cot_baseline:
            if args.baseline_mode:
                print("Running in baseline mode. All samples will use the 'simple_baseline' strategy.")
            elif args.tale_ep_baseline:
                print("Running in TALE-EP baseline mode. All samples will use the 'normal' strategy.")
            elif args.concise_cot_baseline:
                print("Running in ConciseCOT baseline mode. All samples will use the 'normal' strategy.")
            probe_output_results = []
            for i, item in enumerate(data):
                strategy = 'simple_baseline' if args.baseline_mode else 'normal'
                probe_output_results.append({
                    'data_index': i,
                    'problem': item['problem'],
                    'ground_truth_answer': item.get('answer'),
                    'strategy': strategy,
                    'final_prediction': 1,  # Corresponds to 'normal'
                    'final_confidence': 1.0,
                    'probabilities': [0.0, 1.0, 0.0] 
                })
        elif args.full_baseline_mode:
            print("Running in full baseline mode. Each sample will be run with all three strategies.")
            probe_output_results = []
            strategies = ["too_easy", "normal", "too_hard"]
            for i, item in enumerate(data):
                for strategy in strategies:
                    probe_output_results.append({
                        'data_index': i,
                        'problem': item['problem'],
                        'ground_truth_answer': item.get('answer'),
                        'strategy': strategy,
                        'final_prediction': strategies.index(strategy),
                        'final_confidence': 1.0,
                        'probabilities': [1.0 if s == strategy else 0.0 for s in strategies]
                    })
        elif args.random_baseline_mode:
            print("Running in random baseline mode. Each sample will be assigned a random strategy.")
            probe_output_results = []
            strategies = ["too_easy", "normal", "too_hard"]
            for i, item in enumerate(data):
                strategy = np.random.choice(strategies)
                probe_output_results.append({
                    'data_index': i,
                    'problem': item['problem'],
                    'ground_truth_answer': item.get('answer'),
                    'strategy': strategy,
                    'final_prediction': strategies.index(strategy),
                    'final_confidence': 1.0,
                    'probabilities': [1.0 if s == strategy else 0.0 for s in strategies]
                })
        elif args.offline_entropy_selection:
            # This mode handles its own generation and selection.
            # The result is the final list to be saved.
            results_to_save = run_offline_entropy_selection(args, data)
            save_results(args, results_to_save)
            print("Offline entropy selection completed successfully.")
            return # Exit early as we've completed the full workflow
        else:
            # --- Probe Prediction Stage (in a separate process) ---
            print("Starting probe prediction stage in a separate process...")
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
                temp_output_path = tmp_file.name

            ctx = multiprocessing.get_context("spawn")
            probe_process = ctx.Process(target=run_probe_stage, args=(args, data, temp_output_path))
            probe_process.start()
            probe_process.join() # Wait for the process to complete

            if probe_process.exitcode != 0:
                raise RuntimeError("The probe prediction subprocess failed.")

            print("Probe process finished. Loading results from temporary file.")
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                probe_output = json.load(f)
            
            os.remove(temp_output_path) # Clean up the temporary file
            
            probe_output_results = probe_output.get('results', [])
            if not probe_output_results:
                 raise RuntimeError("Probe process did not produce any results.")

        # At this point, the probe process has exited and its GPU memory is released.
        
        # Generation Stage
        if not args.only_probe_prediction:
            # Determine if we're in baseline mode
            is_baseline = args.baseline_mode or args.tale_ep_baseline or args.concise_cot_baseline or args.full_baseline_mode or args.random_baseline_mode
            
            if args.search_prompts:
                # Run prompt search mode
                results_to_save = run_prompt_search(args, probe_output_results)
            else:
                # Normal generation mode
                final_output = run_generation_stage(args, probe_output_results, is_baseline_mode=is_baseline)
                results_to_save = final_output['results']
        else:
            # Re-wrap the probe results in the expected format if only predicting
            results_to_save = probe_output_results


        save_results(args, results_to_save)
        print("Inference completed successfully.")
        
        # Run bench evaluation if not in probe-only mode
        if not args.only_probe_prediction:
            print("\n" + "="*60)
            print("Starting bench evaluation...")
            bench_results = run_bench_evaluation(args.output_path)
            if bench_results:
                print("Bench evaluation completed successfully.")
            else:
                print("Bench evaluation was skipped or failed.")
            print("="*60)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Set start method for multiprocessing to 'spawn' for CUDA safety
    multiprocessing.set_start_method("spawn", force=True)
    main()
