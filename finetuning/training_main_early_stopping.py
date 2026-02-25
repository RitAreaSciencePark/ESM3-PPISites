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
        # Check if there is history to save
        if state.log_history:
            # Create DataFrame from the entire history
            df = pd.DataFrame(state.log_history)
            # Save to CSV (overwrite to keep it updated)
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
    # --- NEW: Added argument for validation file ---
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV file")
    # -----------------------------------------------
    parser.add_argument("--test_file", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--dataset_type", type=str, default="p", help="Dataset type (e.g., 'p')")
    parser.add_argument("--output_dir", type=str, default="./finetuning", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--gradient_batch", type=int, default=2, help="number of mini-batches for gradient accumulation ")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="L2 regularization strength")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()
    #torch.cuda.empty_cache()
    # -------------------------------
    # Load sequences and labels
    # -------------------------------
    
    # --- PREVIOUS CODE COMMENTED OUT ---
    # all_train_seqs, all_train_labels = load_biodl_dataset(args.train_file, dataset_type=args.dataset_type)
    # test_seqs, test_labels = load_biodl_dataset(args.test_file, dataset_type=args.dataset_type)

    # # Split training data into 80% train and 20% validation
    # train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    #     all_train_seqs, all_train_labels, test_size=0.2, random_state=42, shuffle=True
    # )
    # -----------------------------------

    # --- NEW CODE: Load Train and Val separately ---
    train_seqs, train_labels = load_biodl_dataset(args.train_file, dataset_type=args.dataset_type)
    val_seqs, val_labels = load_biodl_dataset(args.val_file, dataset_type=args.dataset_type)
    test_seqs, test_labels = load_biodl_dataset(args.test_file, dataset_type=args.dataset_type)
    # -----------------------------------------------

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
    pos_weight = torch.tensor(num_neg / (num_pos + 1e-6), dtype=torch.float)

    # -------------------------------
    # Model
    # -------------------------------
    model = ModelForResidueClassification(args.model_name, pos_weight=pos_weight)
    
    # before_params = model.model.transformer.blocks[1].ffn[1]._parameters["weight"].clone()
    
    # -------------------------------
    # Training arguments
    # -------------------------------
    now = datetime.now()
    output_dir = os.path.join(args.output_dir, args.model_name, now.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(output_dir, exist_ok=True)

    training_args_dict = {
    "output_dir": output_dir,
    "num_train_epochs": args.num_train_epochs,
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "gradient_accumulation_steps": args.gradient_batch,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_safetensors": False,  # Needed for T5-style models
    "logging_dir": "./logs",
    "logging_steps": 10,
    "load_best_model_at_end": True,
    # --- MODIFIED: EARLY STOPPING METRIC ---
    #"metric_for_best_model": "eval_mcc", # PREVIOUS
    #"greater_is_better": True,           # PREVIOUS
    "metric_for_best_model": "eval_loss", # NEW: Monitor Validation Loss
    "greater_is_better": False,           # NEW: Lower loss is better
    # ---------------------------------------
    "save_total_limit": 3,
    "fp16": True,
    "report_to": "none",
    "weight_decay": args.weight_decay,
    "learning_rate": args.lr,
    }
    
    training_args = TrainingArguments(**training_args_dict)


    metrics_csv_path = "test_metrics.csv"
    csv_callback = CSVLoggerCallback(metrics_csv_path)
    
    # --- MODIFIED: EARLY STOPPING CALLBACK ---
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=7, # Stops if valid loss doesn't improve for 3 epochs
        early_stopping_threshold=0.1 # Any decrease in loss is considered an improvement
    )
    # -----------------------------------------

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[csv_callback,early_stopping]
    )

    train_result = trainer.train()

    print("🔍 Evaluating on test set...", flush=True)
    test_metrics = trainer.evaluate(test_dataset)
    print("📊 Test metrics:", test_metrics, flush=True)

    # -------------------------------
    # Save model and metrics
    # -------------------------------
    # after_params = model.model.transformer.blocks[1].ffn[1]._parameters["weight"].clone()
    # change = (before_params - after_params).sum()
    # print(f"Model weights changed by {change}")
    
    trainer.save_model(output_dir)
    print(f"✅ Model saved to {output_dir}", flush=True)

    #metrics 
    metrics = trainer.state.log_history
    metrics_df = pd.DataFrame(metrics)

    #save metrics and parameters
    dict_to_save = {
        "training_args": training_args_dict,
        "training_metrics": metrics_df,
        "test_metrics": test_metrics,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "val_file": args.val_file # Added to saved dict
    }

    with open( os.path.join(output_dir, "params_dict.pkl") , "wb") as f:
        pickle.dump(dict_to_save, f)

    print("✅ Dictionary saved", flush=True)

#run
if __name__ == "__main__":
    main()