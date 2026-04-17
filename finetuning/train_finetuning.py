import os, sys, random
import argparse
from datetime import datetime
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, T5Tokenizer 
from transformers import AutoModel, T5EncoderModel
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# re-import
from src.models import ModelForResidueClassification
from src.data_utils import load_biodl_dataset
from src.dataset_class import ResidueInterfaceDataset
from src.evaluation import compute_metrics

# -------------------------------
# Wrapper to guarantee Precision & Recall
# -------------------------------
def enhanced_compute_metrics(eval_pred):
    metrics = compute_metrics(eval_pred)
    
    if not any("precision" in k.lower() for k in metrics):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        # ✅ Mirror compute_metrics exactly: sigmoid + threshold, then flatten
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.7).astype(int)

        labels_flat = labels.flatten()
        preds_flat = preds.flatten()

        mask = labels_flat != -100
        labels_masked = labels_flat[mask]
        preds_masked = preds_flat[mask]

        metrics["precision"] = precision_score(labels_masked, preds_masked, average='macro', zero_division=0)
        metrics["recall"] = recall_score(labels_masked, preds_masked, average='macro', zero_division=0)
        
    return metrics

# -------------------------------
# Custom Trainer for Train Set Evaluation
# -------------------------------
class CustomTrainer(Trainer):
    """
    Custom Trainer that evaluates both the validation dataset and the training dataset
    at the end of each epoch to track train_set performances.
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # First evaluate the default eval_dataset (validation)
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # If this is the standard validation step, also evaluate the training dataset
        if metric_key_prefix == "eval" and self.train_dataset is not None:
            super().evaluate(self.train_dataset, ignore_keys, "train")
            
        return eval_results

# -------------------------------
# Custom Callback for CSV Logging
# -------------------------------
class CSVLoggerCallback(TrainerCallback):
    """
    A Custom Callback that groups log history by epoch and shapes it 
    into the specified structure continuously during training.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.expected_cols = [
            "epoch", "is_best", 
            "train_step_loss", # Training loss from standard gradient steps
            "train_loss", "train_mcc", "train_f1", "train_accuracy", "train_precision", "train_recall",
            "eval_loss", "eval_mcc", "eval_f1", "eval_accuracy", "eval_precision", "eval_recall",
            "test_loss", "test_mcc", "test_f1", "test_accuracy", "test_precision", "test_recall"
        ]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.log_history:
            return
            
        df_data = {}
        # Group logs by epoch
        for log in state.log_history:
            epoch = log.get("epoch")
            if epoch is None: 
                continue
            
            epoch = round(epoch, 4)
            if epoch not in df_data:
                df_data[epoch] = {"epoch": epoch}
            
            for k, v in log.items():
                if k == "loss":
                    # Distinct standard loss tracking vs full 'train_loss' evaluated on the train set
                    df_data[epoch]["train_step_loss"] = v
                elif k != "epoch" and not k.startswith("test_"):
                    df_data[epoch][k] = v
                    
        df = pd.DataFrame(list(df_data.values()))
        
        # Calculate the current best epoch based on eval_loss
        df["is_best"] = False
        if "eval_loss" in df.columns:
            best_idx = df["eval_loss"].idxmin()
            if pd.notna(best_idx):
                df.loc[best_idx, "is_best"] = True
                
        # Ensure all expected columns exist
        for col in self.expected_cols:
            if col not in df.columns:
                df[col] = None
                
        # Filter out extra Hugging Face default columns and sort
        df = df[[c for c in self.expected_cols if c in df.columns]]
        df.to_csv(self.file_path, index=False)

# -------------------------------
# Main
# -------------------------------
def main():
    # -------------------------------
    # Argument parser
    # -------------------------------
    parser = argparse.ArgumentParser(description="Fine-tune ESM2/Ankh on residue classification")

    parser.add_argument("--model_name", type=str, default="ElnaggarLab/ankh-large", help="Model name or path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--dataset_type", type=str, default="p", help="Dataset type (e.g., 'p')")
    parser.add_argument("--output_dir", type=str, default="./finetuning", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--gradient_batch", type=int, default=2, help="number of mini-batches for gradient accumulation ")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="L2 regularization strength")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()
    
    # -------------------------------
    # Determine Dataset Name and Setup Directories
    # -------------------------------
    # Extract filename like "dataset.csv" and strip the extension to get the dataset name
    train_file_name = os.path.basename(args.train_file)
    dataset_name = train_file_name.rsplit('.', 1)[0]
    
    # 1. Output dir for the saved model: output_dir + models/model_<dataset_name>
    model_output_dir = os.path.join(args.output_dir, "models", f"model_{dataset_name}")
    # 2. Output dir for performances: results_finetuning/performances_<dataset_name>
    
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -------------------------------
    # Load sequences and labels
    # -------------------------------
    train_seqs, train_labels = load_biodl_dataset(args.train_file, dataset_type=args.dataset_type)
    val_seqs, val_labels = load_biodl_dataset(args.val_file, dataset_type=args.dataset_type)
    test_seqs, test_labels = load_biodl_dataset(args.test_file, dataset_type=args.dataset_type)

    print(f"✅ Training samples: {len(train_seqs)}", flush=True)
    print(f"✅ Validation samples: {len(val_seqs)}", flush=True)
    print(f"✅ Test samples: {len(test_seqs)}", flush=True)

    # -------------------------------
    # Tokenizer & Datasets
    # -------------------------------
    if any(x in args.model_name for x in ["ankh", "T5", "t5"]):
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False, legacy=True)
    elif "esm3" in args.model_name:
        tokenizer = EsmSequenceTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = ResidueInterfaceDataset(train_seqs, train_labels, tokenizer)
    val_dataset = ResidueInterfaceDataset(val_seqs, val_labels, tokenizer)
    test_dataset = ResidueInterfaceDataset(test_seqs, test_labels, tokenizer)

    # -------------------------------
    # Compute positive weight for class imbalance
    # -------------------------------
    flat_labels = np.concatenate(train_labels)
    num_pos = (flat_labels == 1).sum()
    num_neg = (flat_labels == 0).sum()
    
    # Convert weight to standard Python float for JSON serialization in config
    pos_weight_val = [float(num_neg / (num_pos + 1e-6))]
    
    # -------------------------------
    # Model
    # -------------------------------
    model = ModelForResidueClassification(model_name=args.model_name, pos_weight=pos_weight_val)
    
    # -------------------------------
    # Training arguments
    # -------------------------------
    training_args_dict = {
        "output_dir": model_output_dir, # Checkpoints directly to new model dir
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_batch,
        "eval_strategy": "epoch", 
        "save_strategy": "epoch",
        "save_safetensors": False,
        "logging_dir": "./logs",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", 
        "greater_is_better": False,          
        "save_total_limit": 3,
        "fp16": True,
        "report_to": "none",
        "weight_decay": args.weight_decay,
        "learning_rate": args.lr,
    }
    
    training_args = TrainingArguments(**training_args_dict)

    # Metrics will be exported directly to performances directory
    metrics_csv_path = os.path.join("results", f"training_metrics_{dataset_name}.csv")
    csv_callback = CSVLoggerCallback(metrics_csv_path)
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=7, 
        early_stopping_threshold=0.1 
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=enhanced_compute_metrics,
        callbacks=[csv_callback, early_stopping]
    )

    train_result = trainer.train()

    print("🔍 Evaluating on test set...", flush=True)
    # Passed metric_key_prefix="test" to ensure compute_metrics outputs "test_mcc", "test_precision" etc.
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("📊 Test metrics:", test_metrics, flush=True)

    # -------------------------------
    # Save model locally (inside output_dir + models/model_<dataset_name>)
    # -------------------------------
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"✅ Model and Tokenizer saved to {model_output_dir}", flush=True)

    # -------------------------------
    # Finalize and Save Performance Metrics
    # -------------------------------
    # Re-build final clean CSV to accurately assign the test metrics to the best epoch
    metrics = trainer.state.log_history
    df_data = {}
    for log in metrics:
        epoch = log.get("epoch")
        if epoch is None: continue
        epoch = round(epoch, 4)
        if epoch not in df_data:
            df_data[epoch] = {"epoch": epoch}
            
        for k, v in log.items():
            if k == "loss":
                df_data[epoch]["train_step_loss"] = v
            elif k != "epoch" and not k.startswith("test_"):
                df_data[epoch][k] = v

    metrics_df = pd.DataFrame(list(df_data.values()))
    metrics_df["is_best"] = False

    if "eval_loss" in metrics_df.columns:
        best_idx = metrics_df["eval_loss"].idxmin()
        if pd.notna(best_idx):
            metrics_df.loc[best_idx, "is_best"] = True
            
            # Since load_best_model_at_end=True, the test metrics come from the best model.
            for k, v in test_metrics.items():
                if k in csv_callback.expected_cols:
                    metrics_df.loc[best_idx, k] = v

    # Add any missing required columns and enforce strict order
    for col in csv_callback.expected_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = None

    metrics_df = metrics_df[csv_callback.expected_cols]
    metrics_df.to_csv(metrics_csv_path, index=False)

    dict_to_save = {
        "training_args": training_args_dict,
        "training_metrics": metrics_df,
        "test_metrics": test_metrics,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "val_file": args.val_file 
    }

    # Save details parameters in the performances folder
    params_path = os.path.join("results", f"params_dict_{dataset_name}.pkl")
    with open(params_path , "wb") as f:
        pickle.dump(dict_to_save, f)
        
    print(f"✅ Performances and Metrics saved to results folder", flush=True)

# run
if __name__ == "__main__":
    main()