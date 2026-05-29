import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# DATASET & MODEL CONFIGURATION
parser = argparse.ArgumentParser(description="Evaluate baseline model")
parser.add_argument("--model_name", default="small_model", help="Model name (big_model or small_model)")
parser.add_argument("--dataset_name", default="BioLiP-3693", help="Dataset name (BioLiP-3693 or PDBbind-1409)")
args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(repo_root, "data") + os.sep
reps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esm3_data", model_name + "data") + os.sep
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_baseline")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_baseline")

id_col = "uniprot_id"
label_col = "p_interface"

# GLOBAL VARIABLES
BATCH = 96
THRESHOLD = 0.7
print(f"Threshold: {THRESHOLD}")

def resolve_checkpoint_path(base_dir, selected_model_name, selected_dataset_name):
    candidate_paths = [
        os.path.join(base_dir, f"{selected_model_name}_{selected_dataset_name}.pt"),
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    return candidate_paths[0]


checkpoint_path = resolve_checkpoint_path(models_dir, model_name, dataset_name)
emb_pt_path = os.path.join(reps_path, "zk448_test.pt")
test_csv = data_path + "zk448_test.csv"
out_csv = os.path.join(results_dir, f"{model_name}_{dataset_name}_evaluation_results.csv")
performance_csv = os.path.join(results_dir, "performance.csv")

os.makedirs(results_dir, exist_ok=True)

print(f"Model: {model_name}")
print(f"Dataset: {dataset_name}")
print(f"Batch size: {BATCH}")


class ProteinDataset(Dataset):
    def __init__(self, embeddings_list, labels_list):
        self.embeddings = embeddings_list
        self.labels = labels_list

        self.mapping = []
        for prot_idx, labels in enumerate(labels_list):
            for res_idx in range(len(labels)):
                self.mapping.append((prot_idx, res_idx))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        prot_idx, res_idx = self.mapping[idx]
        x = self.embeddings[prot_idx][res_idx].view(-1).float()
        y = torch.tensor(self.labels[prot_idx][res_idx]).float()
        return x, y


def hotspot_sites_from_binary_array(binary_array):
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)


def probabilities_to_csv_string(prob_array, decimals=6):
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)


def upsert_performance_csv(
    performance_csv_path,
    model,
    dataset,
    f1_value,
    mcc_value,
    auc_value,
    precision_value,
    recall_value,
):
    new_row = pd.DataFrame(
        [
            {
                "model": model,
                "dataset": dataset,
                "F1": f1_value,
                "MCC": mcc_value,
                "AUC": auc_value,
                "Precision": precision_value,
                "Recall": recall_value,
            }
        ]
    )

    if os.path.exists(performance_csv_path):
        performance_df = pd.read_csv(performance_csv_path)
        if not performance_df.empty and {"model", "dataset"}.issubset(performance_df.columns):
            performance_df = performance_df[
                ~(
                    (performance_df["model"].astype(str) == str(model))
                    & (performance_df["dataset"].astype(str) == str(dataset))
                )
            ]
            performance_df = pd.concat([performance_df, new_row], ignore_index=True)
        else:
            performance_df = new_row
    else:
        performance_df = new_row

    performance_df = performance_df[["model", "dataset", "F1", "MCC", "AUC", "Precision", "Recall"]]
    performance_df.to_csv(performance_csv_path, index=False)


def load_aligned_data(
    csv_path,
    pt_path,
    label_col="p_interface",
    id_col="uniprot_id",
):
    test_df = pd.read_csv(csv_path)
    emb_dict = torch.load(pt_path, map_location="cpu")

    id_set = set(emb_dict.keys())
    test_df = test_df[test_df[id_col].astype(str).isin(id_set)].copy()

    x_test = [
        F.normalize(emb_dict[str(u_id)], p=2, dim=1)
        for u_id in test_df[id_col]
    ]
    y_test = [
        np.fromstring(str(labels), dtype=np.int32, sep=",")
        for labels in test_df[label_col]
    ]

    if not x_test:
        raise ValueError(
            f"No aligned proteins found between {csv_path} and {pt_path} using id_col='{id_col}'."
        )

    return test_df, x_test, y_test


def run_inference(
    checkpoint_path,
    emb_pt_path,
    test_csv,
    batch_size=96,
    threshold=0.7,
    id_col="uniprot_id",
    label_col="p_interface",
):
    test_df, x_test, y_test = load_aligned_data(
        csv_path=test_csv,
        pt_path=emb_pt_path,
        label_col=label_col,
        id_col=id_col,
    )

    hidden_dim = x_test[0].shape[1]
    model = nn.Linear(hidden_dim, 1)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = ProteinDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(np.int32)
            y_prob_all.extend(probs)
            y_pred_all.extend(preds)
            y_true_all.extend(yb.numpy())

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all, dtype=np.float32)

    results_list = []
    current_idx = 0

    for i, y_single in enumerate(y_test):
        prot_len = len(y_single)
        y_true = y_true_all[current_idx: current_idx + prot_len]
        y_pred = y_pred_all[current_idx: current_idx + prot_len]
        y_prob = y_prob_all[current_idx: current_idx + prot_len]

        uniprot_id = test_df.iloc[i][id_col] if id_col in test_df.columns else i
        sequence = test_df.iloc[i]["sequence"] if "sequence" in test_df.columns else ""
        hotspot_true = hotspot_sites_from_binary_array(y_true)
        hotspot_pred = hotspot_sites_from_binary_array(y_pred)
        probability_vector = probabilities_to_csv_string(y_prob)

        f1_sing = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        mcc_sing = matthews_corrcoef(y_true, y_pred)
        precision_sing = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_sing = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        auc_sing = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else np.nan

        hotspot_count_true = np.sum(y_true)
        hotspot_density_true = hotspot_count_true / prot_len

        hotspot_count_pred = np.sum(y_pred)
        hotspot_density_pred = hotspot_count_pred / prot_len

        results_list.append(
            {
                "uniprot_id": uniprot_id,
                "sequence": sequence,
                "hotspot_true": hotspot_true,
                "hotspot_pred": hotspot_pred,
                "probability_vector": probability_vector,
                "f1_score": f1_sing,
                "mcc": mcc_sing,
                "precision": precision_sing,
                "recall": recall_sing,
                "auc": auc_sing,
                "length": prot_len,
                "hotspot_density_true": hotspot_density_true,
                "hotspot_density_pred": hotspot_density_pred,
            }
        )

        current_idx += prot_len

    df_results = pd.DataFrame(results_list)

    global_f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    global_mcc = matthews_corrcoef(y_true_all, y_pred_all)
    global_auc = roc_auc_score(y_true_all, y_prob_all) if np.unique(y_true_all).size > 1 else np.nan
    global_precision = precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    global_recall = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)

    return {
        "test_df": test_df,
        "df_results": df_results,
        "global_f1": global_f1,
        "global_mcc": global_mcc,
        "global_auc": global_auc,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "y_prob_all": y_prob_all,
    }


def main():
    outputs = run_inference(
        checkpoint_path=checkpoint_path,
        emb_pt_path=emb_pt_path,
        test_csv=test_csv,
        batch_size=BATCH,
        threshold=THRESHOLD,
        id_col=id_col,
        label_col=label_col,
    )

    outputs["df_results"].to_csv(out_csv, index=False)

    upsert_performance_csv(
        performance_csv_path=performance_csv,
        model=model_name,
        dataset=dataset_name,
        f1_value=outputs["global_f1"],
        mcc_value=outputs["global_mcc"],
        auc_value=outputs["global_auc"],
        precision_value=outputs["global_precision"],
        recall_value=outputs["global_recall"],
    )

    print(f"Global Test F1: {outputs['global_f1']:.4f}")
    print(f"Global Test MCC: {outputs['global_mcc']:.4f}")
    print(f"Global Test AUC: {outputs['global_auc']:.4f}")
    print(f"Global Test Precision: {outputs['global_precision']:.4f}")
    print(f"Global Test Recall: {outputs['global_recall']:.4f}")
    print(f"Saved per-protein metrics to: {out_csv}")
    print(f"Updated shared performance summary in: {performance_csv}")


if __name__ == "__main__":
    main()
