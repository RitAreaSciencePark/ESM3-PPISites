#!/usr/bin/env python3

import argparse
import os
import random
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import login

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig, LogitsConfig


SEED = 42

default_model_name = "small_model"
default_dataset_name = "BioLiP-3693"
default_models_dir = "../baseline/models_baseline"
default_emb_pt_path = "esm3_reps/test_embeddings.pt"
default_out_dir = "inference_results"
default_base_path = "data/haddock_units"

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="test_input.csv")
    parser.add_argument("--token", required=True)

    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--dataset-name", default=default_dataset_name)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=default_models_dir)

    parser.add_argument("--emb-pt-path", default=default_emb_pt_path)
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--base-path", default=default_base_path)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--structure-steps", type=int, default=8)
    parser.add_argument("--force-structures", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")

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


def generate_structures(df_input, args, device):
    print("STEP 1 - STRUCTURE GENERATION")

    struct_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    struct_model.eval()

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        complex_id = row["complex"]
        protein_id = row["id"]
        chain = row["chain"]
        sequence = row["sequence"]

        complex_dir = Path(args.base_path) / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)

        pdb_path = complex_dir / f"{protein_id}.pdb"

        if pdb_path.exists() and not args.force_structures:
            continue

        try:
            protein = ESMProtein(sequence=sequence)

            with torch.no_grad():
                protein = struct_model.generate(
                    protein,
                    GenerationConfig(track="structure", num_steps=args.structure_steps),
                )

            protein.to_pdb(str(pdb_path))
            write_pdb_with_chain_id(pdb_path, chain)

        except Exception as e:
            print(f"Failed to generate structure for {protein_id}: {e}")


def generate_embeddings(df_input, args, device):
    print("STEP 2 - EMBEDDINGS")

    emb_path = Path(args.emb_pt_path)

    if emb_path.exists():
        return str(emb_path)

    if args.skip_embeddings:
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    emb_path.parent.mkdir(parents=True, exist_ok=True)

    esm3_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    esm3_model.eval()

    emb_dict = {}

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        protein_id = row["id"]
        complex_id = row["complex"]
        sequence = row["sequence"]

        try:
            protein = ESMProtein(sequence=sequence)

            with torch.no_grad():
                protein_tensor = esm3_model.encode(protein)
                logits_output = esm3_model.logits(
                    protein_tensor,
                    LogitsConfig(return_embeddings=True),
                )

            emb = logits_output.embeddings

            if emb.dim() == 3:
                emb = emb.squeeze(0)

            if emb.shape[0] == len(sequence) + 2:
                emb = emb[1:-1]
            elif emb.shape[0] > len(sequence):
                emb = emb[: len(sequence)]

            emb_dict[protein_id] = {
                "tensor": emb.detach().cpu().float(),
                "complex": complex_id,
            }

        except Exception as e:
            print(f"Failed to generate embedding for {protein_id}: {e}")

    torch.save(emb_dict, emb_path)
    print(f"Saved embeddings to: {emb_path}")

    return str(emb_path)


def resolve_checkpoint_path(model_name, dataset_name, models_dir, model_path=None):
    checkpoint_path = os.path.join(models_dir, f"{model_name}_{dataset_name}.pt")

    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if model_path is None:
        raise FileNotFoundError(
            "Checkpoint was not found and no --model-path was provided.\n"
            f"Tried: {checkpoint_path}"
        )

    if os.path.isfile(model_path):
        return model_path

    if os.path.isdir(model_path):
        fallback_checkpoint_path = os.path.join(model_path, f"{model_name}_{dataset_name}.pt")

        if os.path.isfile(fallback_checkpoint_path):
            return fallback_checkpoint_path

        raise FileNotFoundError(f"Checkpoint not found inside directory: {fallback_checkpoint_path}")

    raise FileNotFoundError(f"Checkpoint path does not exist: {model_path}")


def hotspot_sites_from_binary_array(binary_array):
    return ",".join(str(idx + 1) for idx, value in enumerate(binary_array) if value == 1)


def probabilities_to_csv_string(prob_array, decimals=6):
    return ",".join(f"{float(p):.{decimals}f}" for p in prob_array)


def normalize_data(pt_path, df_input):
    emb_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    original_to_paired = dict(zip(df_input["original_id"], df_input["id"]))
    paired_to_complex = dict(zip(df_input["id"], df_input["complex"]))

    processed_data = []

    for u_id in sorted(emb_dict.keys()):
        data = emb_dict[u_id]
        protein_id = original_to_paired.get(u_id, u_id)

        if protein_id not in paired_to_complex:
            continue

        raw_tensor = data["tensor"] if isinstance(data, dict) else data

        if isinstance(data, dict):
            complex_id = data.get("complex", paired_to_complex[protein_id])
        else:
            complex_id = paired_to_complex[protein_id]

        normalized_tensor = F.normalize(raw_tensor, p=2, dim=1).float()

        processed_data.append(
            {
                "id": protein_id,
                "complex": complex_id,
                "tensor": normalized_tensor,
            }
        )

    expected = set(df_input["id"])
    found = {item["id"] for item in processed_data}
    missing = sorted(expected - found)

    if missing:
        raise ValueError(
            "Missing embeddings for these ids:\n"
            + "\n".join(missing)
            + "\nEmbedding keys must match either original input ids or paired ids with _W/_Z suffix."
        )

    return sorted(processed_data, key=lambda x: x["id"])


def run_inference(checkpoint_path, emb_pt_path, df_input, seed):
    print("STEP 3 - INFERENCE")

    set_seed(seed)

    processed_data = normalize_data(emb_pt_path, df_input)
    hidden_dim = processed_data[0]["tensor"].shape[1]

    model = nn.Linear(hidden_dim, 1)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results_list = []

    with torch.no_grad():
        for item in processed_data:
            x_single = item["tensor"]
            protein_id = item["id"]
            complex_id = item["complex"]
            seq_len = x_single.shape[0]

            logits = model(x_single).squeeze(-1)
            y_prob = torch.sigmoid(logits).cpu().numpy()

            threshold = 0.7 if seq_len < 100 else 0.6
            y_pred = (y_prob > threshold).astype(np.int32)

            hotspot_pred = hotspot_sites_from_binary_array(y_pred)
            probability_vector = probabilities_to_csv_string(y_prob[y_pred == 1])

            results_list.append(
                {
                    "complex": complex_id,
                    "id": protein_id,
                    "hotspot_pred": hotspot_pred,
                    "prob": probability_vector,
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
    print("STEP 4 - PATCHES")

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

    config_content = f"""run_dir = "{run_dir / patch}"
molecules = ["{id_path / prot_w}","{id_path / prot_z}"]
ncores = 64

[topoaa]
iniseed = 42

[rigidbody]
ambig_fname = "{tbl_file}"
sampling = 100
iniseed = 42
npart = 5

[seletop]
select = 1

[emref]
ambig_fname = "{tbl_file}"
iniseed = 42

"""

    config_file.write_text(config_content)
    print(f"Created configuration file at: {config_file}")


def run_pairing(base_path, df_top):
    print("STEP 5 - PAIRING AND CONFIGS")

    df_w = df_top[df_top["id"].str.endswith("_W")].copy()
    df_z = df_top[df_top["id"].str.endswith("_Z")].copy()
    df_pairs = pd.merge(df_w, df_z, on="complex", suffixes=("_W", "_Z"))

    for _, row in df_pairs.iterrows():
        complex_id = row["complex"]

        id_path = Path(base_path) / complex_id
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
                        f.write(f"assign {active_selection}\n")
                        f.write("        (" + " or ".join(passives) + ") 10.0 6.0 4.0\n")

            compile_eval_config(id_path, tbl_file, config_file)


def main():
    args = parse_args()

    set_seed(args.seed)
    login(token=args.token)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.base_path, exist_ok=True)

    df_input = load_input_pairs(args.input)
    df_input.to_csv(Path(args.out_dir) / "paired_input.csv", index=False)

    checkpoint_path = resolve_checkpoint_path(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        models_dir=args.models_dir,
        model_path=args.model_path,
    )

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Node: {node_name}")
    print(f"Checkpoint: {checkpoint_path}")

    generate_structures(df_input, args, device)

    current_emb_pt_path = generate_embeddings(df_input, args, device)

    out_csv = os.path.join(args.out_dir, f"{args.model_name}_results.csv")
    patches_csv = os.path.join(args.out_dir, f"{args.model_name}_patches.csv")
    pairing_csv = os.path.join(args.out_dir, f"{args.model_name}_for_pairing.csv")

    df_results = run_inference(
        checkpoint_path=checkpoint_path,
        emb_pt_path=current_emb_pt_path,
        df_input=df_input,
        seed=args.seed,
    )

    df_results.to_csv(out_csv, index=False)
    print(f"Saved inference results to: {out_csv}")

    df_all = make_patches(df_results)

    if not df_all.empty:
        df_all.to_csv(patches_csv, index=False)
        print(f"Saved patches to: {patches_csv}")

        df_filtered = filter_patches_for_pairing(df_all)
        df_filtered.to_csv(pairing_csv, index=False)
        print(f"Saved pairing input to: {pairing_csv}")

        run_pairing(args.base_path, df_filtered)
    else:
        print("No patches generated.")

    print("Done.")


if __name__ == "__main__":
    main()


