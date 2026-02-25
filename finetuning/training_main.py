import os, sys, random
import argparse
from datetime import datetime
import json, pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Tokenizer, TrainingArguments, Trainer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# re-import from your source files
from src.models import ModelForResidueClassification
from src.data_utils import load_biodl_dataset
from src.dataset_class import ResidueInterfaceDataset
from src.evaluation import compute_metrics

# -------------------------------
# Custom Trainer to Log Val & Test every Epoch
# -------------------------------
class ValidationTestTrainer(Trainer):
    """
    Custom Trainer that:
      - Evaluates on both validation and test sets at the end of every epoch.
      - Saves combined metrics (train loss, val/test loss, MCC, F1, accuracy) to a CSV.
      - Tracks the best validation MCC and saves the best model manually to a
        dedicated 'best_model/' subfolder whenever a new best is found.
    """
    def __init__(self, test_dataset, log_csv_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataset = test_dataset
        self.log_csv_path = log_csv_path
        self.best_mcc = -float("inf")  # MCC ranges from -1 to +1
        self.best_model_dir = os.path.join(self.args.output_dir, "best_model")

    def _get_train_loss_for_epoch(self):
        """
        Pulls the most recent training loss from self.state.log_history.
        HuggingFace logs training entries with a 'loss' key (not 'eval_loss').
        """
        current_epoch = int(self.state.epoch) if self.state.epoch else 0
        for entry in reversed(self.state.log_history):
            entry_epoch = int(entry.get("epoch", -1))
            if entry_epoch == current_epoch and "loss" in entry and "eval_loss" not in entry:
                return {"train_loss": entry["loss"]}
        return {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 1. Validation set evaluation
        val_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="eval")

        # 2. Test set evaluation
        test_metrics = super().evaluate(self.test_dataset, ignore_keys, metric_key_prefix="test")

        # 3. Training loss from log history
        train_metrics = self._get_train_loss_for_epoch()

        # 4. Combine all metrics and tag with epoch
        combined_metrics = {**train_metrics, **val_metrics, **test_metrics}
        combined_metrics["epoch"] = int(self.state.epoch) if self.state.epoch else 0

        # 5. Check if this is the best model by validation MCC and save if so
        val_mcc = val_metrics.get("eval_mcc", None)
        if val_mcc is not None and val_mcc > self.best_mcc:
            self.best_mcc = val_mcc
            combined_metrics["is_best"] = True
            os.makedirs(self.best_model_dir, exist_ok=True)
            self.save_model(self.best_model_dir)
            print(f"\n⭐ New best model (MCC={val_mcc:.4f}) saved to {self.best_model_dir}")
        else:
            combined_metrics["is_best"] = False

        # 6. Save to CSV
        self.log_metrics_to_csv(combined_metrics)

        return val_metrics

    def log_metrics_to_csv(self, metrics):
        """Append per-epoch metrics to a CSV, with a sensible column order."""
        priority_cols = [
            "epoch", "is_best",
            # Training
            "train_loss",
            # Validation
            "eval_loss", "eval_mcc", "eval_f1", "eval_accuracy",
            # Test
            "test_loss", "test_mcc", "test_f1", "test_accuracy",
        ]

        df = pd.DataFrame([metrics])

        # Reorder: priority columns first (if present), then any extras
        cols = [c for c in priority_cols if c in df.columns] + \
               [c for c in df.columns if c not in priority_cols]
        df = df[cols]

        if not os.path.exists(self.log_csv_path):
            df.to_csv(self.log_csv_path, index=False)
        else:
            df.to_csv(self.log_csv_path, mode='a', header=False, index=False)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2/Ankh (No Early Stopping)")

    parser.add_argument("--model_name", type=str, default="ElnaggarLab/ankh-large", help="Model name or path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--dataset_type", type=str, default="p", help="Dataset type (e.g., 'p')")
    parser.add_argument("--output_dir", type=str, default="./finetuning_v2", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--gradient_batch", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="L2 regularization strength")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()

    # -------------------------------
    # Load Data
    # -------------------------------
    train_seqs, train_labels = load_biodl_dataset(args.train_file, dataset_type=args.dataset_type)
    val_seqs, val_labels = load_biodl_dataset(args.val_file, dataset_type=args.dataset_type)
    test_seqs, test_labels = load_biodl_dataset(args.test_file, dataset_type=args.dataset_type)

    print(f"✅ Training samples: {len(train_seqs)}")
    print(f"✅ Validation samples: {len(val_seqs)}")
    print(f"✅ Test samples: {len(test_seqs)}")

    # -------------------------------
    # Tokenizer
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
    # Class Balancing
    # -------------------------------
    flat_labels = np.concatenate(train_labels)
    num_pos = (flat_labels == 1).sum()
    num_neg = (flat_labels == 0).sum()
    pos_weight = torch.tensor(num_neg / (num_pos + 1e-6), dtype=torch.float)

    # -------------------------------
    # Model
    # -------------------------------
    model = ModelForResidueClassification(args.model_name, pos_weight=pos_weight)

    # -------------------------------
    # Paths & Args
    # -------------------------------
    now = datetime.now()
    run_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(args.output_dir, args.model_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the custom CSV log
    progress_csv_path = os.path.join(output_dir, args.train_file.split("/")[-1]+"training_progress.csv")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_batch,
        
        # --- CHANGES FOR INTEGER EPOCHS ---
        logging_strategy="epoch",    # Log to console only at end of epoch
        evaluation_strategy="epoch", # Evaluate at end of epoch
        save_strategy="epoch",       # Save checkpoint at end of epoch
        # ----------------------------------
        
        save_safetensors=False,
        logging_dir="./logs",
        
        # --- BEST MODEL TRACKED MANUALLY BY MCC IN CUSTOM TRAINER ---
        # load_best_model_at_end is False because we handle best-model saving
        # ourselves in ValidationTestTrainer based on validation MCC.
        load_best_model_at_end=False,
        metric_for_best_model="eval_mcc",
        greater_is_better=True,
        # --------------------------------------------------------------
        
        save_total_limit=3,
        fp16=True,
        report_to="none",
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
    )

    # -------------------------------
    # Initialize Custom Trainer
    # -------------------------------
    trainer = ValidationTestTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,       # Standard 'eval' set
        test_dataset=test_dataset,      # Custom arg for our custom trainer
        log_csv_path=progress_csv_path, # Where to save the per-epoch CSV
        compute_metrics=compute_metrics,
        callbacks=[]                    # No EarlyStoppingCallback
    )

    # -------------------------------
    # Train
    # -------------------------------
    print("🚀 Starting training (No Early Stopping)...")
    trainer.train()

    # -------------------------------
    # Final Save
    # -------------------------------
    trainer.save_model(output_dir)
    print(f"✅ Final model saved to {output_dir}")
    print(f"⭐ Best model (val MCC={trainer.best_mcc:.4f}) saved to {trainer.best_model_dir}")
    print(f"✅ Per-epoch metrics saved to {progress_csv_path}")

    # Save parameters dict for reproducibility
    dict_to_save = {
        "training_args": training_args.to_dict(),
        "train_file": args.train_file,
        "test_file": args.test_file,
        "val_file": args.val_file
    }

    with open(os.path.join(output_dir, "params_dict.pkl"), "wb") as f:
        pickle.dump(dict_to_save, f)

if __name__ == "__main__":
    main()