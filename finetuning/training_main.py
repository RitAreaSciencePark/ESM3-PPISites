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

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

# re-import
from src.models import ModelForResidueClassification
from src.data_utils import load_biodl_dataset
from src.dataset_class import ResidueInterfaceDataset
from src.evaluation import compute_metrics

# -------------------------------
# Custom Callback for CSV Logging
# -------------------------------
class CSVLoggerCallback(TrainerCallback):
    """
    A Custom Callback that saves the log history to a CSV file 
    every time the trainer logs metrics.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.log_history:
            df = pd.DataFrame(state.log_history)
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
    now = datetime.now()
    output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"), now.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(output_dir, exist_ok=True)

    training_args_dict = {
        "output_dir": output_dir,
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

    metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
    csv_callback = CSVLoggerCallback(metrics_csv_path)
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=7, 
        early_stopping_threshold=0.1 
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[csv_callback, early_stopping]
    )

    train_result = trainer.train()

    print("🔍 Evaluating on test set...", flush=True)
    test_metrics = trainer.evaluate(test_dataset)
    print("📊 Test metrics:", test_metrics, flush=True)

    # -------------------------------
    # Save model and metrics locally
    # -------------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and Tokenizer saved to {output_dir}", flush=True)

    metrics = trainer.state.log_history
    metrics_df = pd.DataFrame(metrics)

    dict_to_save = {
        "training_args": training_args_dict,
        "training_metrics": metrics_df,
        "test_metrics": test_metrics,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "val_file": args.val_file 
    }

    with open(os.path.join(output_dir, "params_dict.pkl") , "wb") as f:
        pickle.dump(dict_to_save, f)
    print("✅ Dictionary saved", flush=True)


# run
if __name__ == "__main__":
    main()