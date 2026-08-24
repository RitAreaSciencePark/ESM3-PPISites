#!/usr/bin/env python3

### <deps>
import os
import re
import numpy as np
import pandas as pd
import urllib
from pathlib import Path
import warnings
from collections import Counter
from typing import Union, List
from tqdm import tqdm

import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser
from Bio.PDB import PDBParser, Select, Selection 
from Bio.PDB import Structure, Model, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
import mdtraj as md

from config_defaults import * ### CONSTANTS
from PDButils import * 		  ### scientific data manipulation
## <!deps>

########## <PRIVATE FUNCTIONS>

def _which_chains(structure):
    chains = [chain.id for chain in Selection.unfold_entities(structure, "C")]
    return chains

def _unfold_job(m1_fp:str, m2_fp:str, cc_fp:str, c1_id:str, c2_id:str, cc1_id:str, cc2_id:str):
    m1 = pars.get_structure(Path(m1_fp).stem, m1_fp)
    m2 = pars.get_structure(Path(m2_fp).stem, m2_fp)
    cc = pars.get_structure(Path(cc_fp).stem, cc_fp)
    c1_chain = select_chains(m1, c1_id)
    c2_chain = select_chains(m2, c2_id)
    cc1_chain = select_chains(cc, cc1_id)
    cc2_chain = select_chains(cc, cc2_id)
    return {'c1':c1_chain, 'c2':c2_chain, 'cc1':cc1_chain, 'cc2':cc2_chain}

def _check_identity(df):
    checked_rows = []
    for index, job in df.iterrows():
        check_1 = job['up_c1'] == job['up_cc1']
        check_2 = job['up_c2'] == job['up_cc2']
        verify = check_1 and check_2
        checked_rows.append(verify)
    return pd.Series(checked_rows)

def _export_seqdata(dict_sequence_chain, base_path):
    required = {'seq', 'resdata'}
    if not required.issubset(dict_sequence_chain.keys()):
        raise ValueError('required dictionary slots not found')   
    resdata_fp = Path(f"{base_path}_map.tsv")
    seq_fp = Path(f"{base_path}_pdbseq.fasta")
    # Export resdata if it doesn't exist
    #if resdata_fp.exists():
        #print(f"[SKIP] {resdata_fp} already exists")
    if not resdata_fp.exists():
        dict_sequence_chain['resdata'].to_csv(resdata_fp, sep="\t", index=False)
        #print(f"[CREATE] {resdata_fp}")
    # Export sequence if it doesn't exist
    #if seq_fp.exists():
        #print(f"[SKIP] {seq_fp} already exists")
    if not seq_fp.exists():
        header = f">{Path(base_path).stem} extraction=protein_letters_3to1_extended"
        seq_fp.write_text(f"{header}\n{dict_sequence_chain['seq']}")
        #print(f"[CREATE] {seq_fp}")
    #print(f"[SUCCESS] Dict export completed for: {base_path}")

############################## <!PRIVATE FUNCTIONS>

########################################################### < BODY >
#####################################################################

########## Section: <input validation>
# Input: file path to PDB lookup table, output directory for Haddock jobs
# Output: valid pdb_lookup object, output directory exists existance
input_fp = Path(REF_FP)
if not input_fp.is_file():
    raise FileNotFoundError(f"File not found: {input_fp}")

pdb_lookup = pd.read_csv(input_fp, sep="\t")
if not set(REQUIRED_KEYS).issubset(pdb_lookup.columns):
    raise ValueError(
        f"Missing required columns: {set(REQUIRED_KEYS) - set(pdb_lookup.columns)}"
    )
if (~pdb_lookup["ids_cc"].str.contains(":", na=False)).any():
    raise ValueError(
        f"Inconsistent usage of column ids_cc in {input_fp.stem}"
    )
if pdb_lookup["jobname"].duplicated().any():
    raise ValueError(f"Jobnames are not unique in {input_fp.stem}")
cc_chains = pdb_lookup['ids_cc'].str.split(':')
pdb_lookup['id_cc1'] = pd.Series(c[0] for c in cc_chains)
pdb_lookup['id_cc2'] = pd.Series(c[1] for c in cc_chains)

output_basedir = Path(HADDOCK_BASEDIR)
if output_basedir.exists() and any(output_basedir.iterdir()) and BEHAVIOUR == "stop":
    raise FileExistsError(f"Directory not empty: {output_basedir}")
output_basedir.mkdir(parents=True, exist_ok=True)
############################## <!input validation>

########## Section: <Download>
# Input: unique PDB ids from keys PDB_c1, PDB_c2, PDB_cc, directory for .cif files
# Output: dowloaded .cif files, skipped downloads if file exist
cifs_dir = Path(DEST_CIFS)
cifs_dir.mkdir(parents=True, exist_ok=True)

pdbs_list = pdb_lookup[PDB_KEYS].values.ravel()
pdbs_list = pd.Series(pdbs_list).dropna().unique().tolist()

existing_cifs = {f.stem for f in cifs_dir.glob("*.cif")}
pdbs_to_process = [p for p in pdbs_list if p not in existing_cifs]
skipped_pdbs = [p for p in pdbs_list if p in existing_cifs]
cifs_fps = [cifs_dir / f"{p}.cif" for p in pdbs_to_process]

if skipped_pdbs:
    print(f"\n[SKIP] {len(skipped_pdbs)} existing CIFs: {skipped_pdbs}\n")

if cifs_fps:
    for p in tqdm(cifs_fps, 'Downloading .cif files'):
        pdb_id = p.stem
        try:
            download_rcsb(pdb_id, p)
        except Exception as e:
            raise ValueError(f"[FAILED] while downloading PDB ID: {pdb_id}") from e
else:
    print("All required CIF files already exist. Skipping download phase.")

############################## <!Download>

########## Section: <Preprocessing>
# Input: .cif files, reference lookup
# Output: .pdb files with filtered and renamed chains
ref = pdb_lookup.copy()
ref_short = ref.head(2)
print(ref_short)

pars = MMCIFParser(QUIET=True)

haddock_jobs = {}
for row in tqdm(ref.itertuples(index=False), desc='Chain Selection for Docking', total=len(ref)):
    cc_fp = cifs_dir / f"{row.PDB_cc}.cif"
    c1_fp = cifs_dir / f"{row.PDB_c1}.cif"
    c2_fp = cifs_dir / f"{row.PDB_c2}.cif"    
    for fp in (cc_fp, c1_fp, c2_fp):
        if not fp.exists():
            raise FileNotFoundError(fp)
        if fp.stat().st_size == 0:
            raise ValueError(fp)
    job_chains = _unfold_job(m1_fp=c1_fp, m2_fp=c2_fp, cc_fp=cc_fp,
        c1_id=row.id_c1, c2_id=row.id_c2, cc1_id=row.id_cc1, cc2_id=row.id_cc2)
    #print(job_chains)
    jobs_free = {key: get_polypeptide_chain(job_chains[key]) for key in ['c1', 'c2', 'cc1', 'cc2']}
    jobs_free['c1'].id = STANDARD_CHAINS[0]
    jobs_free['c2'].id = STANDARD_CHAINS[1]
    jobs_free['cc1'].id = STANDARD_CHAINS[0]
    jobs_free['cc2'].id = STANDARD_CHAINS[1]
    monomer_1 = assemble_structure(jobs_free['c1'], structure_id = row.PDB_c1)
    monomer_2 = assemble_structure(jobs_free['c2'], structure_id = row.PDB_c2)
    heterodim = assemble_structure(jobs_free['cc1'], jobs_free['cc2'], structure_id = row.PDB_cc)
    stru_dict = {'m1': monomer_1, 'm2': monomer_2, 'cc': heterodim}
    haddock_jobs[row.jobname] = stru_dict

io = PDBIO()

skipped_write = []
for jobname, stru_dict in tqdm(haddock_jobs.items(), desc='Saving Structures', leave=False):
    job_dir = output_basedir / jobname
    job_dir.mkdir(parents=True, exist_ok=True)
    for name, structure in stru_dict.items():
        struct_id = structure.id
        chains = ''.join(_which_chains(structure))
        out_fp = job_dir / f"{struct_id}_{chains}.pdb"
        if out_fp.exists():
            skipped_write.append(out_fp.stem)
            print(f'[SKIP] {out_fp.stem}.pdb found in {jobname}')
        if not out_fp.exists():
            io.set_structure(structure)
            io.save(str(out_fp))
            #print(f'[SUCCESS] saved structure: {out_fp.stem}.pdb')
print(f"[SKIP] Skipped {len(skipped_write)} .pdb files: {skipped_write}")
############################## <!Preprocessing>





########## Section: <cif-Metadata>
# Input: cif file paths in {cifs_dir}, pdb_lookup as chains_lookup
# Output: pdb lookup table updated with UniProtKB crossreferences
# Default file path 'data/UP_seqs/pdb_lookup_crossref.tsv' 
print(f"{'-'*20}\nSTEP 2: CIF Metadata Extraction")
meta_lookup_fp = Path(METADATA_FP)
if meta_lookup_fp.exists():
    meta_lookup = pd.read_csv(meta_lookup_fp, sep="\t")
    print(f'[SKIP] metadata exists, loaded from {str(meta_lookup_fp)}\n')
if not meta_lookup_fp.exists():
    chains_lookup = pdb_lookup.copy()

    cifs_list = [str(fp) for fp in list(cifs_dir.rglob('*.cif'))]
    cif_ids = [Path(c).stem for c in cifs_list]
    meta_df = extract_uniprot_from_cifs(cifs_list)
    meta_df.head(5)

    meta_lookup = chains_lookup.copy()
    lookups = {
        "up_c1":  ("PDB_c1", "id_c1"),
        "up_c2":  ("PDB_c2", "id_c2"),
        "up_cc1": ("PDB_cc", "id_cc1"),
        "up_cc2": ("PDB_cc", "id_cc2"),
    }
    for up_col, (pdb_col, chain_col) in lookups.items():
        tmp = meta_df.rename(columns={
            "pdb_id": pdb_col,
            "chain": chain_col,
            "uniprot_id": up_col
        })[[pdb_col, chain_col, up_col]]
        meta_lookup = meta_lookup.merge(tmp, on=[pdb_col, chain_col], how="left")

    meta_lookup['up_pass'] = _check_identity(meta_lookup)

    ########## ATTENTION: ATTENTION: ATTENTION!!! <Manual-curation>
    print(f'[WARNING] Manual curation implemented for current job!!!')
    #meta_lookup[~meta_lookup['up_pass']]
    meta_lookup.loc[meta_lookup['PDB_cc'] == '2AJF', 'up_c1'] = 'Q9BYF1' # inputed
    meta_lookup.loc[meta_lookup['PDB_cc'] == '2C0L', 'up_c2'] = 'P22307' # rabbit ortologue no.
    #################### <!Manual-curation>

    meta_lookup['up_pass'] = _check_identity(meta_lookup)

    if not meta_lookup['up_pass'].all():
        raise ValueError('crossreferencing failed')
    meta_lookup = meta_lookup.drop(columns=['up_pass'])
    meta_lookup['c1_rename'] = STANDARD_CHAINS[0]
    meta_lookup['c2_rename'] = STANDARD_CHAINS[1]

    meta_lookup_fp = Path(METADATA_FP)
    meta_lookup_fp.parent.mkdir(parents=True, exist_ok=True)
    meta_lookup.to_csv(meta_lookup_fp, sep="\t")
    print(f"{'-'*10}\n[SUCCESS] written metadata with UniProtKB ids {meta_lookup_fp}\n{'-'*10}\n")
############################## <!cif-Metadata>

########## Section: <UniProtKB>
# Input: UniProtKB IDs from PDB ID Crossreferences
# Output: .fasta file for each chain
# !!! ASSUMES and asserts that same reference for complex and monomers
skipped_upids = []
for _, row in tqdm(meta_lookup.iterrows(), 'Downloading UniProtKB crossreferences'):
    job_dir = output_basedir / row.PDB_cc
    outdir = job_dir / 'UP_seqs'
    outdir.mkdir(parents=True, exist_ok=True)
    upid_1 = row.up_c1
    upid_2 = row.up_c2
    assert upid_1 == row.up_cc1, f"up_c1 != up_cc1 at {row.PDB_cc}"
    assert upid_2 == row.up_cc2, f"up_c2 != up_cc2 at {row.PDB_cc}"
    up1_fp = outdir / f'{upid_1}_{STANDARD_CHAINS[0]}_upseq.fasta'
    up2_fp = outdir / f'{upid_2}_{STANDARD_CHAINS[1]}_upseq.fasta'
    for up_fp, upid in [(up1_fp, upid_1), (up2_fp, upid_2)]:
        if up_fp.exists():
            skipped_upids.append(str(up_fp.stem))
            #print(f"[SKIP] {up_fp.stem}.fasta already there")
        if not up_fp.exists():
            download_uniprot_fasta(upid, up_fp)
            #print(f'[SUCCESS] {upid} to {up_fp}')
if skipped_upids:
	print(f"[SKIP] {len(skipped_upids)} UniProtKB fasta files: {skipped_upids}\n")
print("[SUCCESS]\n\n")
############################## <!UniProtKB>

########## Section: <PDB-Seqs>
input_basedir = Path(HADDOCK_BASEDIR)
target_dirs = [p for p in input_basedir.iterdir() if p.is_dir()]

pars = PDBParser(QUIET=True)

for target in tqdm(target_dirs, 'PDB sequence extraction'):
    outdir = target / 'PDB_seqs'
    outdir.mkdir(parents=True, exist_ok=True)
    m1_fp = list(target.glob(f'*_{STANDARD_CHAINS[0]}.pdb'))
    m2_fp = list(target.glob(f'*_{STANDARD_CHAINS[1]}.pdb'))
    cc_fp = list(target.glob(f"*_{''.join(STANDARD_CHAINS)}.pdb"))
    if len(m1_fp) != 1 or len(m2_fp) != 1 or len(cc_fp) != 1:
        raise ValueError(f"Expected exactly 1 file for each pattern in {target}, got: m1={len(m1_fp)}, m2={len(m2_fp)}, cc={len(cc_fp)}")
    m1_fp = str(m1_fp[0])
    m2_fp = str(m2_fp[0])
    cc_fp = str(cc_fp[0])
    pdb_id_m1 = str(Path(m1_fp).stem).split('_')[0] 
    pdb_id_m2 = str(Path(m2_fp).stem).split('_')[0]
    pdb_id_cc = str(Path(cc_fp).stem).split('_')[0]

    chains = _unfold_job(m1_fp, m2_fp, cc_fp, 
        STANDARD_CHAINS[0], STANDARD_CHAINS[1],
        STANDARD_CHAINS[0], STANDARD_CHAINS[1])

    new_names = [f'c1_{pdb_id_m1}_{STANDARD_CHAINS[0]}',
        f'c2_{pdb_id_m2}_{STANDARD_CHAINS[1]}',
        f'cc1_{pdb_id_cc}_{STANDARD_CHAINS[0]}',
        f'cc2_{pdb_id_cc}_{STANDARD_CHAINS[1]}']

    chains = {new_key: chains[old_key] for new_key, old_key in zip(new_names, chains.keys())}

    for jobname, pdb_chain in chains.items():
        base_path = f'{str(outdir)}/{jobname}'
        pdb_seqdata = sequence_chain_bio(pdb_chain)
        _export_seqdata(pdb_seqdata, base_path=base_path)
############################## <!PDB-Seqs>

########################################################### <! BODY >

