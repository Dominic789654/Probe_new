# -*- coding: utf-8 -*-
"""
train_budget_probe.py

基于 calculate_metrics.py 生成的带标签数据集，训练一个用于预算控制的Probing模型。
该模型旨在预测问题的难度类别 ("Too Easy", "Normal", "Too Hard")。
在训练过程中实时生成hidden states来训练探查器。

作者: Assistant
日期: 2025年8月7日
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import json
import os
import wandb
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict
from probe_utils import (
    FocalLoss,
    ClassBalancedLoss,
    S1TextDataset,
    EnhancedProbe,
    MLPProbe,
)

# Try to import Muon optimizer
try:
    from muon import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Warning: Muon optimizer not available. Install with: pip install muon-optimizer")

# --- 1. 参数解析 ---

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Probe Training for Budget Control")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate for AdamW (default: 0.001)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer type: adamw or muon (default: adamw)")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon optimizer on hidden weights (default: 0.02)")
    parser.add_argument("--muon_aux_lr", type=float, default=3e-4,
                        help="Learning rate for auxiliary AdamW in Muon (default: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.95,
                        help="Momentum for AdamW (default: 0.95)")
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"],
                        help="Type of probe to use: linear or mlp (default: linear)")
    parser.add_argument("--max_length", type=int, default=2048, 
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--loss_type", type=str, default="weighted_ce", 
                        choices=["ce", "weighted_ce", "focal", "class_balanced"],
                        help="Loss function type (default: weighted_ce)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss (default: 2.0)")
    parser.add_argument("--cb_beta", type=float, default=0.9999,
                        help="Beta parameter for class-balanced loss (default: 0.9999)")
    
    # LLM配置
    parser.add_argument("--llm_model", type=str, 
                        default="/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B",
                        help="Path to LLM model for hidden state extraction")
    parser.add_argument("--extract_layer", type=int, default=-1, 
                        help="Which layer to extract hidden states from (-1 for last layer, default: -1)")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="analysis/probe_training_data.csv",
                        help="Path to training data CSV file (default: analysis/probe_training_data.csv)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation set split ratio (default: 0.2)")
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N epochs (default: 1)")
    
    # WandB配置  
    parser.add_argument("--project_name", type=str, default="budget-control-probe",
                        help="WandB project name (default: budget-control-probe)")
    parser.add_argument("--run_name", type=str, default="probe-training",
                        help="WandB run name (default: probe-training)")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable WandB logging")
    
    # 其他配置
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save_path", type=str, default="probe_models",
                        help="Save path for the trained model")
    return parser.parse_args()

# 硬编码为三分类预算控制探针
LABEL_TO_ID = {"Too Easy": 0, "Normal": 1, "Too Hard": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
TRAINING_TYPE = "budget_control"
PROBE_NAME = "Budget Control Probe"


# --- 2. 数据加载和预处理 ---

def analyze_label_distribution(labels: List[int], dataset_name: str = "Dataset") -> Dict[str, Any]:
    """分析并打印label分布"""
    label_counter = Counter(labels)
    total_samples = len(labels)
    
    print(f"\n=== {dataset_name} Label Distribution ===")
    print(f"Total samples: {total_samples}")
    
    sorted_labels = sorted(label_counter.keys())
    
    for label in sorted_labels:
        count = label_counter[label]
        percentage = count / total_samples * 100
        class_name = ID_TO_LABEL.get(label, f"Class {label}")
        print(f"Label {label} ({class_name}): {count:5d} samples ({percentage:5.1f}%)")
    
    if len(sorted_labels) > 1:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.array(sorted_labels), 
            y=np.array(labels)
        )
        print(f"\nComputed class weights for balancing:")
        for i, label in enumerate(sorted_labels):
            class_name = ID_TO_LABEL.get(label, f"Class {label}")
            print(f"Label {label} ({class_name}): {class_weights[i]:.4f}")
        
        return {
            'label_counts': dict(label_counter),
            'class_weights': class_weights,
            'sorted_labels': sorted_labels,
            'total_samples': total_samples
        }
    else:
        return {'label_counts': dict(label_counter), 'class_weights': None, 'sorted_labels': sorted_labels, 'total_samples': total_samples}

def load_budget_data(data_path: str, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Loads the budget control labeled dataset from a CSV file or Oracle JSONL file."""
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run the labeling script first.")
    
    # Determine file type and load accordingly
    if data_path.endswith('.jsonl'):
        # Load Oracle JSONL format
        data_entries = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_entries.append(json.loads(line))
        
        print(f"Loaded {len(data_entries)} entries from Oracle JSONL file")
        
        # Create DataFrame for consistent processing
        df_data = {
            'problem_key': [entry['question'] for entry in data_entries],
            'budget_label': [entry['difficulty_label'] for entry in data_entries]
        }
        df = pd.DataFrame(df_data)
        
        # Map oracle labels to training labels
        # oracle: 'easy', 'normal', 'hard' -> training: 'Too Easy', 'Normal', 'Too Hard'
        oracle_to_training_label = {
            'easy': 'Too Easy',
            'normal': 'Normal', 
            'hard': 'Too Hard'
        }
        
        df['budget_label'] = df['budget_label'].map(oracle_to_training_label)
        
        # Check for unmapped labels
        if df['budget_label'].isnull().any():
            unknown_labels = df[df['budget_label'].isnull()].index.tolist()
            original_labels = [data_entries[i]['difficulty_label'] for i in unknown_labels]
            raise ValueError(f"Unknown oracle labels found: {set(original_labels)}. Expected: {list(oracle_to_training_label.keys())}")
        
        print(f"Oracle label distribution:")
        for oracle_label, training_label in oracle_to_training_label.items():
            count = sum(1 for entry in data_entries if entry['difficulty_label'] == oracle_label)
            print(f"  {oracle_label} -> {training_label}: {count} samples")
            
    else:
        # Load CSV format (original behavior)
        df = pd.read_csv(data_path)
        
        if 'problem_key' not in df.columns or 'budget_label' not in df.columns:
            raise ValueError("CSV file must contain 'problem_key' and 'budget_label' columns.")
    
    # Common processing for both formats
    df['label_id'] = df['budget_label'].map(LABEL_TO_ID)
    
    if df['label_id'].isnull().any():
        unknown_labels = df[df['label_id'].isnull()]['budget_label'].unique()
        raise ValueError(f"Unknown labels found in data: {unknown_labels}. Expected: {list(LABEL_TO_ID.keys())}")

    def map_text_to_template(text: str) -> str:
        template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": text+"\n\nLet's reason step by step, and put your final answer within \boxed{}."}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        template += "<think>\n\nI need to analyze the difficulty of the problem first, and then give a budget control strategy. I think the difficulty of this question is"
        # template += "<think>"
        # breakpoint()
        return template
    
    data = {
        'texts': [map_text_to_template(text) for text in df['problem_key'].tolist()],
        'labels': df['label_id'].astype(int).tolist()
    }
    
    print("Data loaded successfully!")
    print(f"Total samples: {len(data['texts'])}")
    
    return data

def create_val_split(all_data: Dict[str, Any], val_split: float = 0.2, random_seed: int = 42) -> Tuple[Dict, Dict]:
    """从总数据中划分出训练集和验证集"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    total_samples = len(all_data['texts'])
    indices = torch.randperm(total_samples).tolist()
    val_size = int(total_samples * val_split)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_data = {
        'texts': [all_data['texts'][i] for i in train_indices],
        'labels': [all_data['labels'][i] for i in train_indices]
    }
    val_data = {
        'texts': [all_data['texts'][i] for i in val_indices],
        'labels': [all_data['labels'][i] for i in val_indices]
    }
    
    print(f"\nDataset split created:")
    print(f"Training samples: {len(train_data['texts'])}")
    print(f"Validation samples: {len(val_data['texts'])}")
    
    return train_data, val_data

# --- 3. Loss函数创建函数 ---

def create_loss_function(loss_type: str, class_weights: np.ndarray = None, 
                       samples_per_class: List[int] = None, 
                       gamma: float = 2.0) -> nn.Module:
    """根据参数创建loss函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if loss_type == "weighted_ce":
        if class_weights is not None:
            weights = torch.FloatTensor(class_weights).to(device)
            print(f"Using Weighted CrossEntropyLoss with weights: {weights}")
            return nn.CrossEntropyLoss(weight=weights)
        else:
            print("Warning: No class weights provided, using standard CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    elif loss_type == "focal":
        alpha = None
        if class_weights is not None:
            alpha = torch.FloatTensor(class_weights).to(device)
        print(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == "class_balanced":
        if samples_per_class is not None:
            print(f"Using Class-Balanced Loss with samples_per_class={samples_per_class}, beta={gamma}")
            return ClassBalancedLoss(samples_per_class, beta=gamma)
        else:
            print("Warning: No samples_per_class provided, using standard CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    else:
        print("Using standard CrossEntropyLoss")
        return nn.CrossEntropyLoss()

# --- 4. Hidden States提取器 ---

class HiddenStatesExtractor:
    def __init__(self, model_path: str, layer_idx: int = -1):
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.layer_idx = layer_idx
        self.device = self.model.device
        self.hidden_dim = self.model.config.hidden_size
        print(f"Model loaded to {self.device}. Hidden dimension: {self.hidden_dim}")
    
    def extract_batch_hidden_states(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # breakpoint()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer_idx]
            # Use attention_mask to find the last non-padding token
            batch_hidden = hidden_states[:, -1]
            return batch_hidden.to(torch.float32)

# --- 5. 训练和评估函数 ---

def train_probe_with_llm(probe_name: str, probe_model: nn.Module, extractor: HiddenStatesExtractor,
                       train_loader: DataLoader, val_loader: DataLoader,
                       criterion: nn.Module, optimizer: optim.Optimizer, scheduler,
                       epochs: int, val_every_n_epochs: int, max_length: int = 2048, 
                       use_wandb: bool = True, save_path: str = None) -> Dict[str, Any]:
    print(f"\n=== Training {probe_name} with Real-time Hidden States ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_model.to(device)
    
    history = defaultdict(list)
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        probe_model.train()
        train_loss, train_preds, train_labels = 0, [], []
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for batch_texts, batch_labels in train_pbar:
            hidden_states = extractor.extract_batch_hidden_states(batch_texts, max_length).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = probe_model(hidden_states)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())
            train_pbar.set_postfix({'loss': loss.item()})
        
        probe_model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch_texts, batch_labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                hidden_states = extractor.extract_batch_hidden_states(batch_texts, max_length).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
                outputs = probe_model(hidden_states)
                val_loss += criterion(outputs, batch_labels).item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        scheduler.step()
        
        # Log metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        history['train_losses'].append(epoch_train_loss)
        history['val_losses'].append(epoch_val_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        history['train_f1s'].append(train_f1)
        history['val_f1s'].append(val_f1)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if use_wandb:
            wandb.log({'epoch': epoch + 1, 'train/loss': epoch_train_loss, 'train/accuracy': train_acc, 'train/f1': train_f1,
                       'val/loss': epoch_val_loss, 'val/accuracy': val_acc, 'val/f1': val_f1, 
                       'learning_rate': optimizer.param_groups[0]['lr']})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_filename = f"{save_path}/{TRAINING_TYPE}_probe_best.pt"
            torch.save(probe_model.state_dict(), model_filename)
            print(f"New best model saved to {model_filename} with F1: {best_val_f1:.4f}")
            if use_wandb:
                wandb.log({'best_val_f1': best_val_f1})

    print(f"\nTraining finished. Best validation F1: {best_val_f1:.4f}")
    history['best_val_f1'] = best_val_f1
    history['final_preds'] = val_preds
    history['final_labels'] = val_labels
    return history


def visualize_training_results(training_results: Dict, output_dir: str = "."):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    # 损失曲线
    axes[0, 0].plot(epochs, training_results['train_losses'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, training_results['val_losses'], 'r-', label='Val Loss')
    axes[0, 0].set_title(f'{PROBE_NAME} Loss')
    axes[0, 0].legend()
    
    # 准确率曲线
    axes[0, 1].plot(epochs, training_results['train_accuracies'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, training_results['val_accuracies'], 'r-', label='Val Acc')
    axes[0, 1].set_title(f'{PROBE_NAME} Accuracy')
    axes[0, 1].legend()

    # F1分数曲线
    axes[1, 0].plot(epochs, training_results['train_f1s'], 'b-', label='Train F1')
    axes[1, 0].plot(epochs, training_results['val_f1s'], 'r-', label='Val F1')
    axes[1, 0].set_title(f'{PROBE_NAME} F1 Score')
    axes[1, 0].legend()
    
    # 混淆矩阵
    cm = confusion_matrix(training_results['final_labels'], training_results['final_preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], 
                xticklabels=ID_TO_LABEL.values(), yticklabels=ID_TO_LABEL.values())
    axes[1, 1].set_title(f'{PROBE_NAME} Confusion Matrix (Val)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{TRAINING_TYPE}_training_results.png", dpi=300)
    plt.show()

# --- 6. 主训练流程 ---

def main():
    args = parse_args()
    print(f"=== {PROBE_NAME} Training ===")
    print(f"Configuration: {vars(args)}")
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.save_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    all_data = load_budget_data(args.data_path, tokenizer)
    train_data, val_data = create_val_split(all_data, val_split=args.val_split, random_seed=args.random_seed)
    
    print("\n" + "="*50 + "\nANALYZING LABEL DISTRIBUTIONS\n" + "="*50)
    train_dist = analyze_label_distribution(train_data['labels'], "Training Set")
    
    if not args.disable_wandb:
        wandb.init(project=args.project_name, name=f"{TRAINING_TYPE}-{args.run_name}", config=vars(args))
        wandb.config.update({
            'train_label_distribution': train_dist['label_counts'],
            'class_weights': train_dist['class_weights'].tolist() if train_dist['class_weights'] is not None else None
        })
    
    print("\n" + "="*50 + "\nINITIALIZING MODELS\n" + "="*50)
    extractor = HiddenStatesExtractor(args.llm_model, args.extract_layer)
    
    num_classes = len(ID_TO_LABEL)
    if args.probe_type == 'linear':
        probe_model = EnhancedProbe(extractor.hidden_dim, num_classes)
    else:
        probe_model = MLPProbe(extractor.hidden_dim, num_classes)
    
    train_dataset = S1TextDataset(train_data['texts'], train_data['labels'])
    val_dataset = S1TextDataset(val_data['texts'], val_data['labels'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    criterion = create_loss_function(
        loss_type=args.loss_type,
        class_weights=train_dist['class_weights'],
        samples_per_class=[train_dist['label_counts'][i] for i in train_dist['sorted_labels']],
        gamma=args.focal_gamma if args.loss_type == 'focal' else args.cb_beta
    )
    optimizer = optim.AdamW(probe_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    
    training_results = train_probe_with_llm(
        PROBE_NAME, probe_model, extractor, train_loader, val_loader, 
        criterion, optimizer, scheduler, args.epochs, args.max_length, 
        use_wandb=(not args.disable_wandb), save_path=args.save_path
    )
    
    print("\n" + "="*50 + "\nGENERATING VISUALIZATIONS\n" + "="*50)
    visualize_training_results(training_results, output_dir=".")
    
    if not args.disable_wandb:
        wandb.finish()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {args.save_path}/{TRAINING_TYPE}_probe_best.pt")

if __name__ == "__main__":
    main()
