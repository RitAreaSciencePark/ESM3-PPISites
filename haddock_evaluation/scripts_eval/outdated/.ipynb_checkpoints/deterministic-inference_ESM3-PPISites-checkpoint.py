#!/usr/bin/env python3
import os
import re
import random
import argparse
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

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
out_csv_remapped = out_dir + "db5_results_remapped.csv"

print(f"Model: {model_name}")
print(f"Dataset: {dataset_name}")


def build_monomer_to_complex_mapping(base_dir="data/haddock_units"):
    """
    Build mapping from monomer residue positions to complex residue positions
    using alignment files in the 'aln' directory.
    
    Returns:
        Dictionary with key (complex_id, monomer_id) -> dict with 'chain_w' and 'chain_z'
        containing mapping from monomer position to complex position
    """
    mapping = defaultdict(lambda: {'chain_w': {}, 'chain_z': {}})
    haddock_dir = Path(base_dir)
    
    for complex_dir in sorted(haddock_dir.iterdir()):
        if not complex_dir.is_dir():
            continue
        
        complex_id = complex_dir.name
        aln_dir = complex_dir / "aln"
        
        if not aln_dir.exists():
            continue
        
        # Find alignment files
        aln_files = list(aln_dir.glob("*_W.tsv")) + list(aln_dir.glob("*_Z.tsv"))
        
        for aln_file in sorted(aln_files):
            # Determine chain type from filename
            chain_type = aln_file.name.split('_')[-1].replace('.tsv', '')  # 'W' or 'Z'
            chain_key = f'chain_{chain_type.lower()}'
            
            try:
                df_aln = pd.read_csv(aln_file, sep='\t')
            except Exception as e:
                print(f"Error reading {aln_file}: {e}")
                continue
            
            # Extract monomer ID from the alignment (e.g., "1QUP_W")
            monomer_ids = df_aln['pdb_chain'].unique()
            
            for monomer_id in sorted(monomer_ids, key=str):
                if pd.isna(monomer_id):
                    continue
                
                df_monomer = df_aln[df_aln['pdb_chain'] == monomer_id]
                
                # Create mapping: position_chain -> position_complex
                for _, row in df_monomer.iterrows():
                    pos_monomer = row['position_chain']
                    pos_complex = row['position_complex']
                    
                    # Only add mapping if both positions exist (no gaps in alignment)
                    if pd.notna(pos_monomer) and pd.notna(pos_complex):
                        try:
                            mapping[(complex_id, monomer_id)][chain_key][int(pos_monomer)] = int(pos_complex)
                        except (ValueError, TypeError):
                            pass
    
    return mapping


def map_monomer_to_complex(complex_id, monomer_id, chain_type, monomer_positions, mapping):
    """
    Map monomer positions to complex positions using alignment mapping.
    
    Args:
        complex_id: ID of the complex
        monomer_id: ID of the monomer (e.g., "1QUP_W")
        chain_type: 'W' or 'Z'
        monomer_positions: List of monomer residue indices (1-based)
        mapping: Dictionary with mapping information
    
    Returns:
        Comma-separated string of complex positions, or empty string if no mapping
    """
    chain_key = f'chain_{chain_type.lower()}'
    
    if (complex_id, monomer_id) not in mapping:
        return ""
    
    chain_mapping = mapping[(complex_id, monomer_id)][chain_key]
    
    complex_positions = []
    for pos in monomer_positions:
        if pos in chain_mapping:
            complex_positions.append(chain_mapping[pos])
    
    if complex_positions:
        return ",".join(map(str, sorted(set(complex_positions), key=int)))
    return ""


def hotspot_sites_from_binary_array(binary_array):
    return [idx + 1 for idx, value in enumerate(binary_array) if value == 1]


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


def run_inference(checkpoint_path, emb_pt_path, mapping, threshold=my_threshold):
    
    processed_data = normalize_data(pt_path=emb_pt_path)
    hidden_dim = processed_data[0]['tensor'].shape[1]
    model = nn.Linear(hidden_dim, 1)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    results_list = []
    results_remapped_list = []
    
    with torch.no_grad():
        for item in processed_data:
            x_single = item['tensor']
            protein_id = item['id']           
            complex_name = item['complex']
            seq_len = x_single.shape[0]  # <-- ADDED: sequence length for dynamic threshold
            
            logits = model(x_single).squeeze(-1)
            y_prob = torch.sigmoid(logits).cpu().numpy()
            
            # <-- CHANGED: length-dependent threshold from new-inference
            threshold = 0.7 if seq_len < 100 else 0.6
            y_pred = (y_prob > threshold).astype(np.int32)
            
            # Original monomer predictions
            hotspot_pred_monomer = hotspot_sites_from_binary_array(y_pred)
            hotspot_pred_monomer_str = ",".join(map(str, hotspot_pred_monomer))
            probability_vector = probabilities_to_csv_string(y_prob[y_pred == 1])
            
            results_list.append(
                {
                    "complex": complex_name,
                    "id": protein_id,
                    "hotspot_pred": hotspot_pred_monomer_str,
                    "prob": probability_vector,
                    "length": seq_len,
                }
            )
            
            # Extract chain type from protein_id (e.g., "1IJJ_W" -> chain_type="W")
            chain_type = protein_id.split('_')[-1] if '_' in protein_id else ""
            
            # Map to complex coordinates using alignment mapping
            hotspot_pred_complex = map_monomer_to_complex(
                complex_name,
                protein_id,
                chain_type, 
                hotspot_pred_monomer, 
                mapping
            )
            
            results_remapped_list.append(
                {
                    "complex": complex_name,
                    "id": protein_id,
                    "monomer_hotspot_pred": hotspot_pred_monomer_str,
                    "complex_hotspot_pred": hotspot_pred_complex,
                    "prob": probability_vector,
                    "length": seq_len,
                }
            )
    
    df_results = pd.DataFrame(results_list, columns=["complex", "id", "hotspot_pred", "prob", "length"])
    df_results_remapped = pd.DataFrame(results_remapped_list, columns=["complex", "id", "monomer_hotspot_pred", "complex_hotspot_pred", "prob", "length"])
    
    return {"df_results": df_results, "df_results_remapped": df_results_remapped}


def main():
    # --- SEED SET AT ENTRY POINT ---
    set_seed(SEED)

    print("Building monomer-to-complex mapping from alignment files...")
    mapping = build_monomer_to_complex_mapping()
    print(f"Mapping built for {len(mapping)} monomer-complex pairs")
    
    outputs = run_inference(
        checkpoint_path=checkpoint_path,
        emb_pt_path=emb_pt_path,
        mapping=mapping,
    )
    
    outputs["df_results"].to_csv(out_csv, index=False)
    print(f"Saved original monomer predictions to: {out_csv}")
    
    outputs["df_results_remapped"].to_csv(out_csv_remapped, index=False)
    print(f"Saved remapped predictions (monomer + complex) to: {out_csv_remapped}")


if __name__ == "__main__":
    # --- SEED SET AT SCRIPT START ---
    set_seed(SEED)
    main()