import os
import random
import argparse
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42

# --- ADDED SEED FUNCTION ---
def set_seed(seed=SEED):
    """Sets the seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Forces deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- ADDED ENVIRONMENT-LEVEL DETERMINISM ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

model_name = "big_model"
dataset_name = "BioLiP-3693"
models_dir = "/u/lade/divorad/scratch/projects/PPI/PPI-Reps/baseline/models_baseline"

# FOR GITHUB, COMMENT PREVIOUS AND UNCOMMENT FOLLOWING
# models_dir = "../baseline/models_baseline"

id_col = "id"
checkpoint_path = os.path.join(models_dir, f"{model_name}_{dataset_name}.pt")
emb_pt_path = "esm3_reps/db5_embeddings.pt"

# --- CHANGED OUTPUT DIRECTORY ---
out_dir = "inference_results/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# --- GET NODE (MACHINE) NAME ---
node_name = socket.gethostname()

def hotspot_sites_from_binary_array(binary_array):
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)

def probabilities_to_csv_string(prob_array, decimals=6):
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)

def normalize_data(pt_path):
    emb_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    processed_data = []

    # --- SORTED ITERATION FOR DETERMINISM ---
    for u_id in sorted(emb_dict.keys()):
        data = emb_dict[u_id]
        raw_tensor = data['tensor']
        complex_name = data['complex']

        normalized_tensor = F.normalize(raw_tensor, p=2, dim=1).float()

        processed_data.append({
            'id': u_id,
            'complex': complex_name,
            'tensor': normalized_tensor
        })

    return processed_data

def run_inference(checkpoint_path, emb_pt_path, seed):
    # --- SEED SET PER RUN ---
    set_seed(seed)

    processed_data = normalize_data(pt_path=emb_pt_path)
    hidden_dim = processed_data[0]['tensor'].shape[1]
    model = nn.Linear(hidden_dim, 1)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results_list = []

    with torch.no_grad():
        for item in processed_data:
            x_single = item['tensor']
            protein_id = item['id']
            complex_name = item['complex']
            seq_len = x_single.shape[0]

            logits = model(x_single).squeeze(-1)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            threshold = 0.7 if seq_len < 100 else 0.6
            y_pred = (y_prob > threshold).astype(np.int32)
            hotspot_pred = hotspot_sites_from_binary_array(y_pred)
            probability_vector = probabilities_to_csv_string(y_prob[y_pred == 1])

            results_list.append(
                {
                    "complex": complex_name,
                    "id": protein_id,
                    "hotspot_pred": hotspot_pred,
                    "prob": probability_vector,
                    "length": seq_len
                }
            )

    df_results = pd.DataFrame(results_list, columns=["complex", "id", "hotspot_pred", "prob", "length"])
    return {"df_results": df_results}

def main():
    # --- SEEDS TO RUN ---
    seeds = [42] ####, 123, 2024]

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running inference with seed: {seed}")
        print(f"{'='*50}")

        # --- OUTPUT FILE WITH SEED AND NODE SUFFIX ---
        out_csv = os.path.join(out_dir, f"db5_results.csv")

        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Node: {node_name}")
        print(f"Output: {out_csv}")

        outputs = run_inference(
            checkpoint_path=checkpoint_path,
            emb_pt_path=emb_pt_path,
            seed=seed,
        )

        outputs["df_results"].to_csv(out_csv, index=False)
        print(f"Saved inference results to: {out_csv}")

if __name__ == "__main__":
    main()
