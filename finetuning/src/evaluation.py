import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    
    # Convert logits to probabilities using sigmoid
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    # Flatten arrays for masked evaluation
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    probs_flat = probs.flatten()

    # Mask out ignored tokens (-100)
    mask = labels_flat != -100
    labels_masked = labels_flat[mask]
    preds_masked = preds_flat[mask]
    probs_masked = probs_flat[mask]

    # Compute metrics
    acc = accuracy_score(labels_masked, preds_masked)
    f1 = f1_score(labels_masked, preds_masked, zero_division=0)
    mcc = matthews_corrcoef(labels_masked, preds_masked)
    
    # Compute ROC AUC only if both classes are present
    try:
        auc = roc_auc_score(labels_masked, probs_masked)
    except ValueError:
        auc = float('nan')  # Undefined if only one class present

    return {"accuracy": acc, "f1": f1, "mcc": mcc, "roc_auc": auc}