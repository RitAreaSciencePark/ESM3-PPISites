import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — make sure finetuning package is importable.
# Adjust if your repo layout differs.
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from transformers import AutoTokenizer, T5Tokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from finetuning.src.models import ModelForResidueClassification
from finetuning.src.dataset_class import ResidueInterfaceDataset
from finetuning.src.data_utils import load_biodl_dataset

# ===========================================================================
# CONFIGURATION — edit these to match your setup
# ===========================================================================

# --- Model sources: paths to the local fine-tuned checkpoints ---
CHECKPOINT_PATH_BIOLIP  = "../finetuning/results_finetuning/models/model_BioLiP-3693_train"
CHECKPOINT_PATH_PDBBIND = "../finetuning/results_finetuning/models/model_PDBbind-1409_train"

# Base ESM3 model name (used to build the architecture when loading locally)
BASE_MODEL_NAME = "EvolutionaryScale/esm3-sm-open-v1"

# --- Data ---
data_dir      = Path(__file__).resolve().parent.parent / "data"
TEST_CSV_PATH = str(data_dir / "zk448_test.csv")
DATASET_TYPE  = "p"           # must match what was used during training

# --- Output ---
results_dir = Path(__file__).resolve().parent / "results_finetuning"
results_dir.mkdir(parents=True, exist_ok=True)

dataset_tag  = "zk448_test"

# Separate output files for predictions
out_csv_biolip   = str(results_dir / f"biolip_{dataset_tag}_predictions.csv")
out_csv_pdbbind  = str(results_dir / f"pdbbind_{dataset_tag}_predictions.csv")
# Performance summary file (will be appended with 2 rows)
performance_csv  = str(results_dir / "performance.csv")

# --- Inference settings ---
BATCH_SIZE  = 1          # keep at 1 for variable-length sequences; increase if padding
THRESHOLD   = 0.7        # probability threshold for positive prediction
MAX_LENGTH  = 1024       # must match the value used in ResidueInterfaceDataset
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================================
print(f"Device       : {DEVICE}")
print(f"Threshold    : {THRESHOLD}")
print(f"Max length   : {MAX_LENGTH}")
print(f"BioLip Model : {CHECKPOINT_PATH_BIOLIP}")
print(f"PDBBind Model: {CHECKPOINT_PATH_PDBBIND}")


# ===========================================================================
# Helper utilities
# ===========================================================================

def hotspot_sites_from_binary_array(binary_array):
    """Return a comma-separated string of 1-based hotspot indices."""
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)


def probabilities_to_csv_string(prob_array, decimals=6):
    """Serialise a probability array to a comma-separated string."""
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)


def upsert_performance_csv(
    performance_csv_path,
    model_name,
    dataset,
    f1_value,
    mcc_value,
    auc_value,
    precision_value,
    recall_value,
):
    """Insert or update a row in the shared performance summary CSV."""
    new_row = pd.DataFrame([{
        "model":     model_name,
        "dataset":   dataset,
        "F1":        f1_value,
        "MCC":       mcc_value,
        "AUC":       auc_value,
        "Precision": precision_value,
        "Recall":    recall_value,
    }])

    if os.path.exists(performance_csv_path):
        perf_df = pd.read_csv(performance_csv_path)
        if not perf_df.empty and {"model", "dataset"}.issubset(perf_df.columns):
            perf_df = perf_df[
                ~(
                    (perf_df["model"].astype(str) == str(model_name))
                    & (perf_df["dataset"].astype(str) == str(dataset))
                )
            ]
            perf_df = pd.concat([perf_df, new_row], ignore_index=True)
        else:
            perf_df = new_row
    else:
        perf_df = new_row

    perf_df = perf_df[["model", "dataset", "F1", "MCC", "AUC", "Precision", "Recall"]]
    perf_df.to_csv(performance_csv_path, index=False)


# ===========================================================================
# Model / tokenizer loading
# ===========================================================================

def load_tokenizer_for_esm3():
    """ESM3 always uses EsmSequenceTokenizer."""
    print("🔹 Loading ESM3 tokenizer...")
    return EsmSequenceTokenizer()


def load_model_from_checkpoint(base_model_name, checkpoint_path, device):
    """
    Supports both pytorch_model.bin and model.safetensors checkpoint formats.
    """
    print(f"\n🔹 Loading model architecture: {base_model_name}")
    model = ModelForResidueClassification(base_model_name)

    bin_path  = os.path.join(checkpoint_path, "pytorch_model.bin")
    safe_path = os.path.join(checkpoint_path, "model.safetensors")

    if os.path.exists(bin_path):
        print(f"🔹 Loading weights from {bin_path}...")
        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict)
    elif os.path.exists(safe_path):
        print(f"🔹 Loading weights from {safe_path}...")
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"No 'pytorch_model.bin' or 'model.safetensors' found in {checkpoint_path}"
        )

    model.to(device)
    model.eval()
    return model


# ===========================================================================
# Inference loop
# ===========================================================================

def run_inference(model, dataloader, seqs, device, max_length=1024, threshold=0.7):
    """
    Run forward passes over the dataloader and collect per-residue predictions
    and probabilities.

    BOS token is at position 0; EOS token is at position max_length-1.
    Usable residue window: [1 : 1 + min(seq_len, max_length-2)].
    """
    max_residues = max_length - 2   # exclude BOS and EOS

    all_preds_binary = []   # list[np.ndarray] — one per sequence
    all_probs        = []   # list[np.ndarray] — one per sequence

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            logits  = outputs["logits"]

            probs_t = torch.sigmoid(logits)
            preds_t = (probs_t > threshold).int()

            probs_np = probs_t.cpu().numpy()
            preds_np = preds_t.cpu().numpy()

            batch_offset = len(all_preds_binary)
            for i in range(preds_np.shape[0]):
                global_idx   = batch_offset + i
                original_len = len(seqs[global_idx])
                effective_len = min(original_len, max_residues)

                start = 1
                end   = start + effective_len

                valid_preds = preds_np[i][start:end]
                valid_probs = probs_np[i][start:end]

                all_preds_binary.append(valid_preds)
                all_probs.append(valid_probs)

    return all_preds_binary, all_probs


# ===========================================================================
# Evaluation routine
# ===========================================================================

def evaluate_model(
    model,
    dataloader,
    test_seqs,
    test_labels,
    test_df_raw,
    device,
    threshold=0.7,
    max_length=1024,
):
    max_residues = max_length - 2
    id_col = "uniprot_id" if "uniprot_id" in test_df_raw.columns else None

    # Run Inference
    all_preds_binary, all_probs = run_inference(
        model, dataloader, test_seqs, device,
        max_length=max_length, threshold=threshold,
    )

    # ------------------------------------------------------------------
    # Aggregate flat arrays for global metrics
    # ------------------------------------------------------------------
    flat_true = []
    flat_pred = []
    flat_prob = []

    for i, (pred_arr, true_labels, probs_arr) in enumerate(
        zip(all_preds_binary, test_labels, all_probs)
    ):
        true_arr = np.array(true_labels[:max_residues], dtype=np.int32)
        if len(pred_arr) != len(true_arr):
            continue

        flat_true.extend(true_arr.tolist())
        flat_pred.extend(pred_arr.tolist())
        flat_prob.extend(probs_arr.tolist())

    flat_true = np.array(flat_true, dtype=np.int32)
    flat_pred = np.array(flat_pred, dtype=np.int32)
    flat_prob = np.array(flat_prob, dtype=np.float32)

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    global_f1        = f1_score(flat_true, flat_pred, pos_label=1, zero_division=0)
    global_mcc       = matthews_corrcoef(flat_true, flat_pred)
    global_precision = precision_score(flat_true, flat_pred, pos_label=1, zero_division=0)
    global_recall    = recall_score(flat_true, flat_pred, pos_label=1, zero_division=0)
    global_auc       = (
        roc_auc_score(flat_true, flat_prob)
        if np.unique(flat_true).size > 1
        else np.nan
    )
    global_accuracy  = accuracy_score(flat_true, flat_pred)
    global_f1_macro  = f1_score(flat_true, flat_pred, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    # Per-sequence results
    # ------------------------------------------------------------------
    results_list = []
    for i, (pred_arr, true_labels, probs_arr) in enumerate(
        zip(all_preds_binary, test_labels, all_probs)
    ):
        true_arr = np.array(true_labels[:max_residues], dtype=np.int32)

        if len(pred_arr) != len(true_arr):
            continue 

        uniprot_id = (
            str(test_df_raw.iloc[i][id_col])
            if id_col and i < len(test_df_raw)
            else str(i)
        )
        sequence = test_seqs[i][:max_residues]
        prot_len = len(true_arr)

        hotspot_true       = hotspot_sites_from_binary_array(true_arr)
        hotspot_pred       = hotspot_sites_from_binary_array(pred_arr)
        probability_vector = probabilities_to_csv_string(probs_arr)

        seq_f1        = f1_score(true_arr, pred_arr, pos_label=1, zero_division=0)
        seq_mcc       = (
            matthews_corrcoef(true_arr, pred_arr)
            if np.unique(true_arr).size > 1
            else np.nan
        )
        seq_precision = precision_score(true_arr, pred_arr, pos_label=1, zero_division=0)
        seq_recall    = recall_score(true_arr, pred_arr, pos_label=1, zero_division=0)
        seq_auc       = (
            roc_auc_score(true_arr, probs_arr)
            if np.unique(true_arr).size > 1
            else np.nan
        )
        seq_accuracy  = accuracy_score(true_arr, pred_arr)

        hotspot_count_true   = int(np.sum(true_arr))
        hotspot_density_true = hotspot_count_true / prot_len

        hotspot_count_pred   = int(np.sum(pred_arr))
        hotspot_density_pred = hotspot_count_pred / prot_len

        results_list.append({
            "uniprot_id":            uniprot_id,
            "sequence":              sequence,
            "hotspot_true":          hotspot_true,
            "hotspot_pred":          hotspot_pred,
            "probability_vector":    probability_vector,
            "f1_score":              seq_f1,
            "mcc":                   seq_mcc,
            "precision":             seq_precision,
            "recall":                seq_recall,
            "auc":                   seq_auc,
            "accuracy":              seq_accuracy,
            "length":                prot_len,
            "hotspot_density_true":  hotspot_density_true,
            "hotspot_density_pred":  hotspot_density_pred,
        })

    df_results = pd.DataFrame(results_list)

    return {
        "df_results":       df_results,
        "global_f1":        global_f1,
        "global_mcc":       global_mcc,
        "global_auc":       global_auc,
        "global_precision": global_precision,
        "global_recall":    global_recall,
        "global_accuracy":  global_accuracy,
        "global_f1_macro":  global_f1_macro,
        "residues_count":   len(flat_true)
    }

def print_metrics(model_name, outputs):
    print("=" * 45)
    print(f"     TEST SET METRICS: {model_name}")
    print("=" * 45)
    print(f"  Accuracy  : {outputs['global_accuracy']:.4f}")
    print(f"  MCC       : {outputs['global_mcc']:.4f}")
    print(f"  AUC       : {outputs['global_auc']:.4f}")
    print(f"  Precision : {outputs['global_precision']:.4f}")
    print(f"  Recall    : {outputs['global_recall']:.4f}")
    print(f"  F1 (pos)  : {outputs['global_f1']:.4f}")
    print(f"  F1 (macro): {outputs['global_f1_macro']:.4f}")
    print(f"  Residues  : {outputs['residues_count']}")
    print("=" * 45)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # 1. Load tokenizer & data
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer_for_esm3()

    print(f"\n🔹 Loading test set from {TEST_CSV_PATH}...")
    test_seqs, test_labels = load_biodl_dataset(TEST_CSV_PATH, dataset_type=DATASET_TYPE)
    test_df_raw = pd.read_csv(TEST_CSV_PATH)
    print(f"   Found {len(test_seqs)} sequences.")

    dataset    = ResidueInterfaceDataset(test_seqs, test_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 2. Evaluate BioLip Model
    # ------------------------------------------------------------------
    model_biolip = load_model_from_checkpoint(BASE_MODEL_NAME, CHECKPOINT_PATH_BIOLIP, DEVICE)
    print("\n🔹 Running inference for BioLip model...")
    outputs_biolip = evaluate_model(
        model_biolip, dataloader, test_seqs, test_labels, test_df_raw,
        DEVICE, THRESHOLD, MAX_LENGTH
    )
    # Free memory
    del model_biolip
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Evaluate PDBBind Model
    # ------------------------------------------------------------------
    model_pdbbind = load_model_from_checkpoint(BASE_MODEL_NAME, CHECKPOINT_PATH_PDBBIND, DEVICE)
    print("\n🔹 Running inference for PDBBind model...")
    outputs_pdbbind = evaluate_model(
        model_pdbbind, dataloader, test_seqs, test_labels, test_df_raw,
        DEVICE, THRESHOLD, MAX_LENGTH
    )
    # Free memory
    del model_pdbbind
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Save Predictions to Separate CSVs
    # ------------------------------------------------------------------
    print("\n🔹 Saving predictions to separate files...")
    
    outputs_biolip["df_results"].to_csv(out_csv_biolip, index=False)
    outputs_pdbbind["df_results"].to_csv(out_csv_pdbbind, index=False)

    # ------------------------------------------------------------------
    # 5. Update shared performance summary
    # ------------------------------------------------------------------
    upsert_performance_csv(
        performance_csv_path=performance_csv,
        model_name="BioLip_Finetuned",
        dataset=dataset_tag,
        f1_value=outputs_biolip["global_f1"],
        mcc_value=outputs_biolip["global_mcc"],
        auc_value=outputs_biolip["global_auc"],
        precision_value=outputs_biolip["global_precision"],
        recall_value=outputs_biolip["global_recall"],
    )

    upsert_performance_csv(
        performance_csv_path=performance_csv,
        model_name="PDBBind_Finetuned",
        dataset=dataset_tag,
        f1_value=outputs_pdbbind["global_f1"],
        mcc_value=outputs_pdbbind["global_mcc"],
        auc_value=outputs_pdbbind["global_auc"],
        precision_value=outputs_pdbbind["global_precision"],
        recall_value=outputs_pdbbind["global_recall"],
    )

    # ------------------------------------------------------------------
    # 6. Print summaries
    # ------------------------------------------------------------------
    print()
    print_metrics("BioLip Model", outputs_biolip)
    print_metrics("PDBBind Model", outputs_pdbbind)
    
    print(f"\n✅ BioLip predictions CSV saved to: {out_csv_biolip}")
    print(f"✅ PDBBind predictions CSV saved to: {out_csv_pdbbind}")
    print(f"✅ Performance summary updated with both models: {performance_csv}")


if __name__ == "__main__":
    main()