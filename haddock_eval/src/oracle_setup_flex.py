#!/usr/bin/env python3

from pathlib import Path
import csv
import os
import glob

from config_defaults import *

input_basedir = Path(HADDOCK_BASEDIR)
target_dirs = [p for p in input_basedir.iterdir() if p.is_dir()]

for target in target_dirs:
    input_dir = target / "contacts"
    tbl_output_file = target / "tbls/ti.tbl"
    tbl_output_file.parent.mkdir(parents=True, exist_ok=True)

    pattern = os.path.join(input_dir, "contacts_*.tsv")
    matches = glob.glob(pattern)
    if not matches:
        print(f"Error: No contacts_*.tsv file found in {input_dir}")
        continue

    input_file = matches[0]
    skipped_count = 0
    valid_rows = []

    with open(input_file, "r", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row_num, row in enumerate(reader, 1):
            try:
                chainA = row["chain_j"]
                chainB = row["chain_i"]
                pos_resA = row["monomer_resseq_j"]
                pos_resB = row["monomer_resseq_i"]

                if not pos_resA.strip() or not pos_resB.strip():
                    skipped_count += 1
                    continue

                valid_rows.append((chainA, chainB, pos_resA, pos_resB))

            except KeyError as e:
                print(f"Error: Missing column {e} at row {row_num}")
                break

    total_written = len(valid_rows)

    if skipped_count > 0:
        print(f"[WARNING] Skipped {skipped_count} unmapped contacts ({total_written} total) in {target.stem}")

    with open(tbl_output_file, "w") as outfile:
        for chainA, chainB, pos_resA, pos_resB in valid_rows:
            tbl_line = (
                f"assign (resid {pos_resA:>4} and segid {chainA:>2} and name CA) "
                f"(resid {pos_resB:>4} and segid {chainB:>2} and name CA) "
                f"{DIST:.1f} {D_MINUS:.1f} {D_PLUS:.1f}\n"
            )
            outfile.write(tbl_line)

    complex_files = list(target.glob("*_WZ.pdb"))
    mol1_files = list(target.glob("*_Z.pdb"))
    mol2_files = list(target.glob("*_W.pdb"))

    if len(complex_files) != 1 or len(mol1_files) != 1 or len(mol2_files) != 1:
        print(f"ERROR: Missing or ambiguous PDB files in {target.name}")
        continue

    pdb_files = {
        "complex": complex_files[0].name,
        "mol1": mol1_files[0].name,
        "mol2": mol2_files[0].name,
    }

    configs_dir = target / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    run_name = "oracle_100_10_flex"
    run_dir = target / "runs" / run_name

    config_content = f"""
run_dir = "{run_dir}"
molecules = ["{target / pdb_files['mol1']}","{target / pdb_files['mol2']}"]
ncores = {DEF_NCORES}

[topoaa]
iniseed = 42

[rigidbody]
unambig_fname = "{tbl_output_file}"
sampling = 100
iniseed = 42

[seletop]
select = 10

[flexref]
unambig_fname = "{tbl_output_file}"
iniseed = 42

[emref]
unambig_fname = "{tbl_output_file}"
iniseed = 42

[caprieval]
reference_fname = "{target / pdb_files['complex']}"
fnat_cutoff = {FNAT_CUTOFF}
"""

    config_path = configs_dir / "config_oracle_flex.cfg"
    config_path.write_text(config_content)

print("\n[COMPLETE]!")

