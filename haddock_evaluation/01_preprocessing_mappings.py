#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
import pandas as pd
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI ---
TSV_FILE = Path("data/pdb_lookup.tsv")
BASE_INPUT = Path("raw-data/haddock_units")
BASE_OUTPUT = Path("data/haddock_units")  # La cartella principale dei risultati

### PERCORSI DEGLI SCRIPT ESTERNI
SCRIPT_DOWNLOAD = Path("scripts_preprocessing/download_PDB_data.py")
SCRIPT_PREPROCESS = Path("scripts_preprocessing/rename_single_chain.py")
SCRIPT_TO_RESIDUES = Path("scripts_preprocessing/PDB_to_residues.py")
SCRIPT_CONTACT_MAP = Path("scripts_preprocessing/contact_map_dimer.py")
SCRIPT_ALIGN_SEQS = Path("scripts_preprocessing/align_PDB_seqs.py")
SCRIPT_REMAP_CONTACTS = Path("scripts_preprocessing/remap_contacts.py")
SCRIPT_SUMMARY = Path("scripts_preprocessing/extract_true_mapped.py")
#SUMMARY_OUTPUT = Path("contacts_summary/remapped_contacts_summary.tsv")

# PARAMETRO DISTANZA CONTACTS
CUTOFF = 0.8


# In[2]:


tsv_tbl = pd.read_csv(TSV_FILE, sep = "\t")
tsv_tbl.head()


# In[3]:


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, output_dir: Path) -> None:
    """Scarica il file PDB se non è già presente."""
    pdb_file = output_dir / f"{pdb_id}.pdb"
    if pdb_file.exists():
        print(f"  Skipping download: {pdb_id}.pdb (already exists)")
        return
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DOWNLOAD), pdb_id, str(output_dir), "--pdb"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Error downloading {pdb_id}: {result.stderr}", file=sys.stderr)
    else:
        print(f"  Downloaded {pdb_id}.pdb")

def run_rename(src_pdb, chain, dst_pdb, rename_to):
    """Chiama lo script per l'estrazione e rinomina di una singola catena."""
    cmd = [
        sys.executable, str(SCRIPT_PREPROCESS),
        '--complex', str(src_pdb), '--chain', chain,
        '--rename-to', rename_to, '--output', str(dst_pdb)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def run_rename_two_chains(src_pdb, chain1, chain2, dst_pdb):
    """Estrae e rinomina due catene dallo stesso complesso e le unisce."""
    from Bio import PDB
    tmp_w = dst_pdb.parent / f"_tmp_W_{dst_pdb.name}"
    tmp_z = dst_pdb.parent / f"_tmp_Z_{dst_pdb.name}"
    try:
        run_rename(src_pdb, chain1, tmp_w, rename_to='W')
        run_rename(src_pdb, chain2, tmp_z, rename_to='Z')

        p = PDB.PDBParser(QUIET=True)
        struct_w = p.get_structure('W', str(tmp_w))
        struct_z = p.get_structure('Z', str(tmp_z))
        for model_w, model_z in zip(struct_w, struct_z):
            for chain in model_z:
                model_w.add(chain)
        io = PDB.PDBIO()
        io.set_structure(struct_w)
        io.save(str(dst_pdb))
    finally:
        tmp_w.unlink(missing_ok=True)
        tmp_z.unlink(missing_ok=True)


# In[4]:


# ---------------------------------------------------------------------------
# Core Pipeline Flow per singolo complesso
# ---------------------------------------------------------------------------

def process_unit(row):
    cpx_id, cpx_chains = row['complex_PDB'], row['complex_chainID']
    ch1_id, ch1_chain  = row['chain1_PDB'], row['chain1_chainID']
    ch2_id, ch2_chain  = row['chain2_PDB'], row['chain2_chainID']

    # Definizione Directory per questo specifico complesso
    download_dir = BASE_INPUT / cpx_id
    unit_out_dir = BASE_OUTPUT / cpx_id

    download_dir.mkdir(parents=True, exist_ok=True)
    unit_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=========================================")
    print(f" PROCESSING COMPLEX: {cpx_id}")
    print(f"=========================================")

    # ---- STEP 0: Download PDBs ----
    print("\n--- Step 0: Downloading PDB files ---")
    for pdb_id in [cpx_id, ch1_id, ch2_id]:
        download_pdb(pdb_id, download_dir)

    # ---- STEP 1 & 2: Rename & Extract Chains ----
    print("\n--- Step 1 & 2: Extracting and Renaming Chains ---")
    ch1_src = download_dir / f"{ch1_id}.pdb"
    ch1_dst = unit_out_dir / f"{ch1_id}_W.pdb"
    run_rename(ch1_src, ch1_chain, ch1_dst, rename_to='W')

    ch2_src = download_dir / f"{ch2_id}.pdb"
    ch2_dst = unit_out_dir / f"{ch2_id}_Z.pdb"
    run_rename(ch2_src, ch2_chain, ch2_dst, rename_to='Z')

    cpx_src = download_dir / f"{cpx_id}.pdb"
    cpx_dst = unit_out_dir / f"{cpx_id}_WZ.pdb"
    cpx_c1, cpx_c2 = cpx_chains.split(':')
    run_rename_two_chains(cpx_src, cpx_c1, cpx_c2, cpx_dst)
    print(f"  Generated: {ch1_dst.name}, {ch2_dst.name}, {cpx_dst.name}")

    # ---- STEP 3: PDB to Residues TSV (Ex 03_toresidues.py) ----
    print("\n--- Step 3: Converting PDBs to Residue TSVs ---")
    residues_dir = unit_out_dir / "residues"
    residues_dir.mkdir(exist_ok=True)

    # Processa tutti i file .pdb generati nello step precedente per questa unità
    for pdb_path in unit_out_dir.glob("*.pdb"):
        output_tsv = residues_dir / f"{pdb_path.stem}_residues.tsv"
        print(f"  Converting {pdb_path.name} -> residues/{output_tsv.name}")
        subprocess.run([
            "python3", str(SCRIPT_TO_RESIDUES),
            str(pdb_path), "--output", str(output_tsv)
        ], check=True)

    # ---- STEP 4: Compute Contacts (Ex 04_compute_contacts.py) ----
    print("\n--- Step 4: Computing Contact Maps ---")
    contacts_dir = unit_out_dir / "contacts"
    contacts_dir.mkdir(exist_ok=True)

    contact_output = contacts_dir / f"contacts_{cpx_id}_WZ.tsv"
    print(f"  Computing contacts for {cpx_dst.name} -> contacts/{contact_output.name}")
    subprocess.run([
        "python3", str(SCRIPT_CONTACT_MAP),
        str(cpx_dst), "--cutoff", str(CUTOFF), "--output", str(contact_output)
    ], check=True)

    # ---- STEP 5: Align Chains (Ex 05_align_chains.py) ----
    print("\n--- Step 5: Aligning Sequences ---")
    aln_dir = unit_out_dir / "aln"
    aln_dir.mkdir(exist_ok=True)

    complex_residues_file = residues_dir / f"{cpx_id}_WZ_residues.tsv"

    for chain_file in residues_dir.glob("*_residues.tsv"):
        if chain_file.name == f"{cpx_id}_WZ_residues.tsv":
            continue  # Evita il self-alignment del complesso

        chain_base = chain_file.name.replace("_residues.tsv", "")
        output_aln_file = aln_dir / f"{cpx_id}_WZ_vs_{chain_base}.tsv"

        print(f"  Aligning: {chain_base} -> aln/{output_aln_file.name}")
        subprocess.run([
            "python3", str(SCRIPT_ALIGN_SEQS),
            "--complex", str(complex_residues_file),
            "--chain", str(chain_file),
            "--output", str(output_aln_file)
        ], check=True)

    # ---- STEP 6: Make Target Pairs (Ex 06_make_targetpairs.py) ----
    print("\n--- Step 6: Remapping Contacts (Target Pairs) ---")
    print(f"  Remapping contacts for directory: {unit_out_dir}")
    subprocess.run([
        "python3", str(SCRIPT_REMAP_CONTACTS),
        str(contacts_dir), str(aln_dir)
    ], check=True)


# In[5]:


def main():
    if not TSV_FILE.exists():
        print(f"Error: Reference TSV file not found at {TSV_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(TSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            complex_id = row["complex_PDB"]
            try:
                # Esegui l'intera pipeline per il complesso corrente
                process_unit(row)
                print(f"\n=> Complex {complex_id} completed successfully!")
            except Exception as e:
                print(f"\n[CRITICAL ERROR] Failed to pipeline {complex_id}: {e}", file=sys.stderr)
                print("Moving to next complex...\n", file=sys.stderr)

    print("\n=========================================")
    print(" PIPELINE EXECUTION COMPLETE FOR ALL UNITS")
    print("=========================================")

    # ---- STEP 7: Run Summary Script ----
    print("\n--- Step 7: Running Collection & Summary Script ---")
    if SCRIPT_SUMMARY.exists():
        try:
            subprocess.run(["python3", str(SCRIPT_SUMMARY)], check=True)
            print("Summary successfully generated via subprocess.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Summary script failed with exit code {e.returncode}", file=sys.stderr)
    else:
        print(f"[ERROR] Summary script not found at {SCRIPT_SUMMARY}", file=sys.stderr)

if __name__ == "__main__":
    main()


# In[ ]:




