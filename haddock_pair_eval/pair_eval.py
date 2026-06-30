#!/usr/bin/env python3

import argparse
import csv
import os
import random
import socket
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import login

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetuning.src.models import ModelForResidueClassification


SEED = 42
OUT_DIR = "inference_results"
BASE_PATH = "data/haddock_units"
STRUCTURE_STEPS = 8
MAX_LENGTH = 1024

node_name = socket.gethostname()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ESM3 Residue Classification Inference")

    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")

    return parser.parse_args()


def clean_sequence(sequence):
    return str(sequence).replace(",", "").replace(" ", "").strip()


def load_input_pairs(input_csv):
    df = pd.read_csv(input_csv)

    required = {"complex", "id", "sequence"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["complex", "id", "sequence"]].copy()
    df["complex"] = df["complex"].astype(str).str.strip()
    df["original_id"] = df["id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[(df["complex"] != "") & (df["original_id"] != "") & (df["sequence"] != "")]

    rows = []
    discarded = []

    for complex_id, group in df.groupby("complex", sort=False):
        group = group.sort_values("original_id").reset_index(drop=True)

        if len(group) != 2:
            discarded.append((complex_id, len(group)))
            continue

        for idx, chain in enumerate(["W", "Z"]):
            row = group.iloc[idx]

            rows.append(
                {
                    "complex": complex_id,
                    "original_id": row["original_id"],
                    "id": f"{row['original_id']}_{chain}",
                    "chain": chain,
                    "sequence": row["sequence"],
                }
            )

    if discarded:
        print("Discarded complexes without exactly two ids:")
        for complex_id, n in discarded:
            print(f"{complex_id}: {n}")

    if not rows:
        raise ValueError("No valid complexes found. Each complex must contain exactly two ids.")

    return pd.DataFrame(rows)


def write_pdb_with_chain_id(pdb_path, chain_id):
    lines = []

    with open(pdb_path, "r") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM", "TER")) and len(line) > 21:
                line = line[:21] + chain_id + line[22:]
            lines.append(line)

    with open(pdb_path, "w") as handle:
        handle.writelines(lines)


def generate_structures(df_input, device):
    print("STEP 1 - STRUCTURE GENERATION")

    struct_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    struct_model.eval()

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        complex_id = row["complex"]
        protein_id = row["id"]
        chain = row["chain"]
        sequence = row["sequence"]

        complex_dir = Path(BASE_PATH) / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)

        pdb_path = complex_dir / f"{protein_id}.pdb"

        if pdb_path.exists():
            continue

        try:
            protein = ESMProtein(sequence=sequence)

            with torch.no_grad():
                protein = struct_model.generate(
                    protein,
                    GenerationConfig(track="structure", num_steps=STRUCTURE_STEPS),
                )

            protein.to_pdb(str(pdb_path))
            write_pdb_with_chain_id(pdb_path, chain)

        except Exception as e:
            print(f"Failed to generate structure for {protein_id}: {e}")


def hotspot_sites_from_binary_array(binary_array):
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)


def probabilities_to_csv_string(prob_array, decimals=6):
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)


def run_inference(df_input, model, tokenizer, device, seed):
    print("STEP 2 - INFERENCE")

    set_seed(seed)

    max_residues = MAX_LENGTH - 2
    results_list = []
    model.eval()

    with torch.no_grad():
        for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
            protein_id = row["id"]
            complex_id = row["complex"]
            sequence = row["sequence"]
            seq_len = len(sequence)

            try:
                inputs = tokenizer(
                    sequence,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)

                effective_len = min(seq_len, max_residues)

                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.sigmoid(logits)
                probs_np = probs.squeeze(0).cpu().numpy()

                start_idx = 1
                end_idx = start_idx + effective_len
                valid_probs = probs_np[start_idx:end_idx]

                threshold = 0.7 if seq_len < 100 else 0.6
                valid_preds = (valid_probs > threshold).astype(np.int32)

                hotspot_pred = hotspot_sites_from_binary_array(valid_preds)
                probability_vector = probabilities_to_csv_string(valid_probs[valid_preds == 1])

                results_list.append(
                    {
                        "complex": complex_id,
                        "id": protein_id,
                        "hotspot_pred": hotspot_pred,
                        "prob": probability_vector,
                        "length": seq_len,
                    }
                )

            except Exception as e:
                print(f"Failed inference for {protein_id}: {e}")
                results_list.append(
                    {
                        "complex": complex_id,
                        "id": protein_id,
                        "hotspot_pred": "",
                        "prob": "",
                        "length": seq_len,
                    }
                )

    return pd.DataFrame(results_list, columns=["complex", "id", "hotspot_pred", "prob", "length"])


def extract_patches(residues, probs):
    if not residues:
        return

    curr_res = [residues[0]]
    curr_probs = [probs[0]]

    for r, p in zip(residues[1:], probs[1:]):
        if r - curr_res[-1] <= 4:
            curr_res.append(r)
            curr_probs.append(p)
        else:
            yield curr_res, curr_probs
            curr_res, curr_probs = [r], [p]

    yield curr_res, curr_probs


def make_patches(df):
    print("STEP 3 - PATCHES")

    patch_data = []

    for _, row in df.iterrows():
        residues = [int(float(x)) for x in str(row["hotspot_pred"]).split(",") if x.strip()]
        probs = [float(x) for x in str(row["prob"]).split(",") if x.strip()]
        seq_len = row["length"]
        min_patch_length = 5 if seq_len < 400 else 4

        if len(residues) > 15:
            groups = [
                (res_group, prob_group)
                for res_group, prob_group in extract_patches(residues, probs)
                if len(res_group) > min_patch_length
            ]
        else:
            groups = [(residues, probs)]

        for res_group, prob_group in groups:
            patch_data.append(
                {
                    "complex": row["complex"],
                    "id": row["id"],
                    "patch": ",".join(map(str, res_group)),
                    "avg_probability": np.mean(prob_group),
                }
            )

    return pd.DataFrame(patch_data)


def filter_patches_for_pairing(df_all):
    df_top = df_all.sort_values(by=["id", "avg_probability"], ascending=[True, False])
    group_sizes = df_top.groupby("id")["id"].transform("size")
    ranks = df_top.groupby("id").cumcount()
    conditional_mask = (ranks < 2) | ((group_sizes == 3) & (ranks == 2))
    df_filtered = df_top[conditional_mask].reset_index(drop=True)
    df_filtered["patch_id"] = "patch_" + df_filtered.groupby("id").cumcount().astype(str)
    df_filtered = df_filtered[["complex", "id", "patch_id", "patch", "avg_probability"]].sort_values(by=["complex"])
    return df_filtered


def compile_eval_config(id_path, tbl_file, config_file):
    id_path = Path(id_path)
    tbl_file = Path(tbl_file)
    config_file = Path(config_file)

    run_dir = id_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    patch = tbl_file.stem

    prot_w_list = list(id_path.glob("*_W.pdb"))
    prot_z_list = list(id_path.glob("*_Z.pdb"))

    if not prot_w_list or not prot_z_list:
        print(f"Error: Missing '_W.pdb' or '_Z.pdb' in {id_path}")
        return

    prot_w = prot_w_list[0].name
    prot_z = prot_z_list[0].name

    config_content = (
        f'run_dir = "{run_dir / patch}"\n'
        f'molecules = ["{id_path / prot_w}","{id_path / prot_z}"]\n'
        f'ncores = 64\n\n'
        f'[topoaa]\n'
        f'iniseed = 42\n\n'
        f'[rigidbody]\n'
        f'ambig_fname = "{tbl_file}"\n'
        f'sampling = 100\n'
        f'iniseed = 42\n'
        f'npart = 5\n\n'
        f'[seletop]\n'
        f'select = 1\n\n'
        f'[emref]\n'
        f'ambig_fname = "{tbl_file}"\n'
        f'iniseed = 42\n\n'
    )

    config_file.write_text(config_content)
    print(f"Created configuration file at: {config_file}")


def run_pairing(df_top):
    print("STEP 4 - PAIRING AND CONFIGS")

    df_w = df_top[df_top["id"].str.endswith("_W")].copy()
    df_z = df_top[df_top["id"].str.endswith("_Z")].copy()
    df_pairs = pd.merge(df_w, df_z, on="complex", suffixes=("_W", "_Z"))

    for _, row in df_pairs.iterrows():
        complex_id = row["complex"]

        id_path = Path(BASE_PATH) / complex_id
        tbl_dir = id_path / "tbls"
        config_dir = id_path / "configs"
        tbl_dir.mkdir(exist_ok=True, parents=True)
        config_dir.mkdir(exist_ok=True, parents=True)

        i = str(row["patch_id_W"]).split("_")[-1]
        j = str(row["patch_id_Z"]).split("_")[-1]

        p_w = [int(x) for x in str(row["patch_W"]).split(",")]
        p_z = [int(x) for x in str(row["patch_Z"]).split(",")]

        is_w_longer = len(p_w) >= len(p_z)
        long, short = (p_w, p_z) if is_w_longer else (p_z, p_w)
        c_long, c_short = ("W", "Z") if is_w_longer else ("Z", "W")

        s = (len(long) - len(short)) // 2

        for k, sequence in enumerate([short, short[::-1]]):
            file_name = f"patches_{i}_{j}_{k}"
            tbl_file = tbl_dir / f"{file_name}.tbl"
            config_file = config_dir / f"{file_name}.cfg"

            with open(tbl_file, "w") as f:
                for idx_s, r_short in enumerate(sequence):
                    idx_l_center = s + idx_s
                    target_window = [idx_l_center - 1, idx_l_center, idx_l_center + 1]

                    passives = []
                    for idx_l in target_window:
                        if 0 <= idx_l < len(long):
                            res_val = long[idx_l]
                            passives.append(f"(resid {res_val} and segid {c_long} and name CA)")

                    if passives:
                        active_selection = f"(resid {r_short} and segid {c_short} and name CA)"
                        f.write("assign " + active_selection + "\n")
                        f.write("        (" + " or ".join(passives) + ") 10.0 6.0 4.0\n")

            compile_eval_config(id_path, tbl_file, config_file)


def main():
    args = parse_args()

    set_seed(SEED)

    print("Authenticating with Hugging Face...")
    login(token=args.token)

    print("Loading model and tokenizer from Hugging Face...", flush=True)
    repo_id = "area-science-park/ESM3-PPISites"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model = ModelForResidueClassification.from_pretrained(repo_id, token=args.token).to(device)
    model.eval()

    tokenizer = EsmSequenceTokenizer()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BASE_PATH, exist_ok=True)

    df_input = load_input_pairs(args.input)
    df_input.to_csv(Path(OUT_DIR) / "paired_input.csv", index=False)

    print(f"Device: {device}")
    print(f"Model: {repo_id}")
    print(f"Node: {node_name}")

    generate_structures(df_input, device)

    df_results = run_inference(
        df_input=df_input,
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed=SEED,
    )

    results_path = Path(OUT_DIR) / "results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"Saved inference results to: {results_path}")

    df_all = make_patches(df_results)

    if not df_all.empty:
        patches_csv = os.path.join(OUT_DIR, "patches.csv")
        pairing_csv = os.path.join(OUT_DIR, "for_pairing.csv")

        df_all.to_csv(patches_csv, index=False)
        print(f"Saved patches to: {patches_csv}")

        df_filtered = filter_patches_for_pairing(df_all)
        df_filtered.to_csv(pairing_csv, index=False)
        print(f"Saved pairing input to: {pairing_csv}")

        run_pairing(df_filtered)
    else:
        print("No patches generated.")

    print("Done.")


if __name__ == "__main__":
    main()

