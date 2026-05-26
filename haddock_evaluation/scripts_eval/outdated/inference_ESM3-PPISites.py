import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

model_name = "big_model"
dataset_name = "BioLiP-3693"
my_threshold = 0.7
models_dir = "/orfeo/scratch/lade/divorad/projects/PPI/PPI-Reps/baseline/models_baseline"
# FOR GITHUB, COMMENT PREVIOUS AND UNCOMMENT FOLLOWING
#models_dir = "../baseline/models_baseline"

id_col = "id"

checkpoint_path = os.path.join(models_dir, f"{model_name}_{dataset_name}.pt")
emb_pt_path = "esm3_reps/db5_embeddings.pt"

# Check if required files exist before creating directories or running anything else
missing_files = []
if not os.path.exists(checkpoint_path):
    missing_files.append(f"Checkpoint not found at: {checkpoint_path}")
if not os.path.exists(emb_pt_path):
    missing_files.append(f"Embeddings not found at: {emb_pt_path}")

if missing_files:
    raise SystemExit("Error: Required file(s) missing!\n" + "\n".join(missing_files))
# --- MINIMAL MODIFICATION END ---

out_dir = "inference_results/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_csv = out_dir + "db5_results.csv"

print(f"Model: {model_name}")
print(f"Dataset: {dataset_name}")


def hotspot_sites_from_binary_array(binary_array):
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)


def probabilities_to_csv_string(prob_array, decimals=6):
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)


def normalize_data(pt_path):
    emb_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    processed_data = []
    
    for u_id, data in emb_dict.items():
        raw_tensor = data['tensor']
        complex_name = data['complex']
        
        normalized_tensor = F.normalize(raw_tensor, p=2, dim=1).float()
        
        processed_data.append({
            'id': u_id,
            'complex': complex_name,
            'tensor': normalized_tensor
        })

    return processed_data

def run_inference(checkpoint_path, emb_pt_path, threshold=my_threshold):
    
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

            logits = model(x_single).squeeze(-1)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_prob > threshold).astype(np.int32)

            hotspot_pred = hotspot_sites_from_binary_array(y_pred)
            probability_vector = probabilities_to_csv_string(y_prob[y_pred == 1])

            results_list.append(
                {
                    "complex": complex_name,
                    "id": protein_id,
                    "hotspot_pred": hotspot_pred,
                    "prob": probability_vector,
                }
            )
    df_results = pd.DataFrame(results_list, columns=["complex", "id", "hotspot_pred", "prob"])

    return {"df_results": df_results}

def main():
    outputs = run_inference(
        checkpoint_path=checkpoint_path,
        emb_pt_path=emb_pt_path,
    )

    outputs["df_results"].to_csv(out_csv, index=False)
    print(f"Saved inference results to: {out_csv}")


if __name__ == "__main__":
    main()


