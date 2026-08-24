#!/usr/bin/env python3

### <deps>
import os
import re
import io
import numpy as np
import pandas as pd
import urllib
from pathlib import Path, PosixPath
import warnings
from collections import Counter, defaultdict
from typing import Union, List
from tqdm import tqdm
from types import SimpleNamespace
import tempfile
import hashlib

import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser
from Bio.PDB import PDBParser, Select, Selection 
from Bio.PDB import Structure, Model, PDBIO
from Bio.PDB.Chain import Chain
from Bio.Align import PairwiseAligner
#from Bio.Align substitution_matrices
from Bio.PDB.Polypeptide import protein_letters_3to1_extended

import mdtraj as md
import pymol
from pymol import cmd

from config_defaults import * ### CONSTANTS
from PDButils import *        ### scientific data manipulation
## <!deps>

########## <PRIVATE FUNCTIONS>

def _pair_fasta(target_fp: Path | str) -> dict[str, dict[str, dict[str, Path]]]:
    fasta_fps = list(Path(target_fp).rglob("*.fasta"))
    paired = defaultdict(lambda: defaultdict(dict))

    for fasta_fp in fasta_fps:
        stem = fasta_fp.stem
        type_chain = stem.split('_')[0]
        if "_pdbseq" in stem:
            seq_type = "pdbseq"
        elif "_upseq" in stem:
            seq_type = "upseq"
        else:
            continue
        if f"_{STANDARD_CHAINS[0]}_" in stem:
            chain = STANDARD_CHAINS[0]
        elif f"_{STANDARD_CHAINS[1]}_" in stem:
            chain = STANDARD_CHAINS[1]
        else:
            continue
        paired[seq_type][chain][type_chain] = fasta_fp
    return {chain: dict(seqs) for chain, seqs in paired.items()}

def _unfold_job(m1_fp:str, m2_fp:str, cc_fp:str, c1_id:str, c2_id:str, cc1_id:str, cc2_id:str):
    m1 = pars.get_structure(Path(m1_fp).stem, m1_fp)
    m2 = pars.get_structure(Path(m2_fp).stem, m2_fp)
    cc = pars.get_structure(Path(cc_fp).stem, cc_fp)
    c1_chain = select_chains(m1, c1_id)
    c2_chain = select_chains(m2, c2_id)
    cc1_chain = select_chains(cc, cc1_id)
    cc2_chain = select_chains(cc, cc2_id)
    return {'c1':c1_chain, 'c2':c2_chain, 'cc1':cc1_chain, 'cc2':cc2_chain}


def _unnest_dict(data):
    if isinstance(data, dict):
        if len(data) == 1:
            # Unwrap single-key dicts: {'seq': 'MKT...'} → 'MKT...'
            return _unnest_dict(next(iter(data.values())))
        return SimpleNamespace(**{k: _unnest_dict(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_unnest_dict(v) for v in data]
    return data

def _fasta_seq(fp):
  if not fp.exists():
    raise ValueError(f"[FAIL] {fp} does not exist")
  with open(str(fp)) as fasta:
    content = fasta.read()
  lines = content.splitlines()
  output = "".join(lines[1:])
  return output

def _merge_positional_data(target: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aln_dir = target / 'aln'
    pdbseq_dir = target / 'PDB_seqs'

    aln_fps = sorted(aln_dir.glob("*pdbseq*pdbseq*.tsv"))
    map_fps = list(pdbseq_dir.glob("*_map.tsv"))

    maps = {fp.stem.replace('_map', ''): pd.read_csv(fp, sep='\t') for fp in map_fps}
    out: dict[str, pd.DataFrame] = {}

    for fp in aln_fps:
        aln = pd.read_csv(fp, sep='\t')
        ref_base = aln['ref_id'].iloc[0].replace('_pdbseq', '')
        qry_base = aln['query_id'].iloc[0].replace('_pdbseq', '')

        m_ref = maps[ref_base][['seq_index', 'sequence', 'resid']].copy()
        m_qry = maps[qry_base][['seq_index', 'sequence', 'resid']].copy()

        m_ref['ref_pos'] = m_ref['seq_index'] + 1
        m_qry['query_pos'] = m_qry['seq_index'] + 1

        merged = aln.merge(
            m_ref,
            left_on=['ref_pos', 'ref_aa'],
            right_on=['ref_pos', 'sequence'],
            how='left'
        ).rename(columns={
            'seq_index': 'seq_index_complex',
            'resid': 'resid_complex',
            'sequence': 'sequence_complex'
        })

        merged = merged.merge(
            m_qry,
            left_on=['query_pos', 'query_aa'],
            right_on=['query_pos', 'sequence'],
            how='left'
        ).rename(columns={
            'seq_index': 'seq_index_monomer',
            'resid': 'resid_monomer',
            'sequence': 'sequence_monomer'
        })

        merged = merged[[
            'ref_id', 'query_id',
            'ref_pos', 'query_pos',
            'seq_index_complex', 'seq_index_monomer',
            'resid_complex', 'resid_monomer',
            'ref_aa', 'query_aa',
            'sequence_complex', 'sequence_monomer',
            'match_aln'
        ]]

        tag = 'c1' if qry_base.startswith('c1_') and not qry_base.startswith('cc1_') else 'c2'
        out[tag] = merged

    return out['c1'], out['c2']
############################## <PRIVATE FUNCTIONS>

########## Section: <loading>
input_basedir = Path(HADDOCK_BASEDIR)
subdirs = [p for p in input_basedir.iterdir() if p.is_dir()]
monomers = []
for jobdir in tqdm(subdirs, 'Loading'):
    for any_chain in STANDARD_CHAINS:
        p_chain = list(jobdir.glob(f'*_{any_chain}.pdb'))
        if p_chain is None:
            raise ValueError(f"Monomer with chain {any_chain} not found")
        if len(p_chain) > 1:
            raise ValueError(f"Too many monomers with chain {any_chain}:\n{p_chain}")
        monomers.extend(p_chain)
all_chains = ''.join(STANDARD_CHAINS)
heterodimers = []
for jobdir in subdirs:
    d_chain = list(jobdir.glob(f'*_{all_chains}.pdb'))
    if d_chain is None:
        raise ValueError(f"Dimer with chains {all_chains} not found")
    if len(d_chain) > 1:
        raise ValueError(f"Too many dimers with chain {all_chains}:\n{d_chain}")
    heterodimers.extend(d_chain)
all_strus = list(set(monomers) | set(heterodimers))
############################## <!loading>

########## Section: <contacts>
# skipped_ct = []
# for ht in tqdm(heterodimers, 'Computing distance and contacts'):
#     mother = ht.parent
#     basename = ht.stem
#     cont_dir = mother / 'contacts'
#     ofile = cont_dir / f'contacts_{basename}.tsv'
#     if ofile.exists():
#         #print(f'[SKIP] {basename} contacts df exists')
#         skipped_ct.append(str(ofile.stem))
#     if not ofile.exists():
#         contacts_df = ca_contacts(ht, cutoff=DIST_CA)
#         cont_dir.mkdir(parents=True, exist_ok=True)
#         contacts_df.to_csv(ofile, sep="\t")

# if skipped_ct:
#     print(f"[SKIP] Skipped {len(skipped_ct)} existing contact maps: {skipped_ct}")
#         #print(f'[SUCCESS] contacts {ofile}')
############################## <!contacts>

########## Section: <Seq-aln>
target_dirs = [p for p in input_basedir.iterdir() if p.is_dir()]

fasta_pairs = [_pair_fasta(target) for target in target_dirs]
meta_aln_stats = []
Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)
for fp_dict, target in tqdm(zip(fasta_pairs, target_dirs), "Alignments PDB to PDB"):
    u1_fp = _unnest_dict(fp_dict['upseq']['W'])
    u2_fp = _unnest_dict(fp_dict['upseq']['Z'])
    c1_fp = _unnest_dict(fp_dict['pdbseq']['W']['c1'])
    cc1_fp = _unnest_dict(fp_dict['pdbseq']['W']['cc1'])
    c2_fp = _unnest_dict(fp_dict['pdbseq']['Z']['c2'])
    cc2_fp = _unnest_dict(fp_dict['pdbseq']['Z']['cc2'])
    u1_id = u1_fp.stem
    u2_id = u2_fp.stem
    c1_id = c1_fp.stem
    cc1_id = cc1_fp.stem
    c2_id = c2_fp.stem
    cc2_id = cc2_fp.stem
    u1_seq = _fasta_seq(u1_fp)
    u2_seq = _fasta_seq(u2_fp)
    c1_seq = _fasta_seq(c1_fp)
    cc1_seq = _fasta_seq(cc1_fp)
    c2_seq = _fasta_seq(c2_fp)
    cc2_seq = _fasta_seq(cc2_fp)
    pairs = [
        (u1_seq, c1_seq, u1_id, c1_id),
        (u1_seq, cc1_seq, u1_id, cc1_id),
        (u2_seq, c2_seq, u2_id, c2_id),
        (u2_seq, cc2_seq, u2_id, cc2_id),
        (cc1_seq, c1_seq, cc1_id, c1_id),
        (cc2_seq, c2_seq, cc2_id, c2_id),
    ]
    aln_stats_list = []
    aln_dir = target / 'aln'
    aln_dir.mkdir(parents=True, exist_ok=True)
    for ref_seq, query_seq, ref_id, query_id in pairs:
        aln = get_alignment(ref_seq=ref_seq, query_seq=query_seq)
        aln_stats = aln_to_stats(aln, ref_id=ref_id, query_id=query_id)
        aln_stats['target'] = target.name
        aln_stats_list.append(aln_stats)
        meta_aln_stats.append(aln_stats)
        aln_df = aln_to_dataframe(aln, ref_id=ref_id, query_id=query_id)
        aln_df_dp = aln_dir / f'{target.name}_{ref_id}_vs_{query_id}_aln.tsv'
        aln_df.to_csv(aln_df_dp, sep='\t', index=False)
        stat = round(len(aln_df[aln_df["match_aln"]=="1"])*100/len(query_seq))
        if stat < 80:
            print(f"[WARNING] {stat}% coverage of query (monomer) over reference (dimer's chain)")
    aln_stats_df = pd.concat(aln_stats_list, ignore_index=True)
    aln_stats_dp = aln_dir / f'{target.name}_aln_stats.tsv'
    aln_stats_df.to_csv(aln_stats_dp, sep='\t', index=False)
meta_df = pd.concat(meta_aln_stats, ignore_index=True)
meta_df.to_csv(Path(METADATA_DIR) / 'jobs_aln_stats.tsv', sep='\t', index=False)
############################## <!Seq-aln>

########## Section: <Str-rmsd>
# OUTPUT from 'align_pdb_structures'
#[0] alignment data monomer vs dimer's chain
# >>> rmsd_cc1_c1[0].keys()
# ['mol_1', 'mol_2', 'rmsd', 'rmsd_before_refinement', 'n_atoms_aligned',
#  'n_cycles', 'n_atoms_pre_refinement', 'score', 'n_residues_aligned']
#[1] aligned structure (dictionary 'mol_1', 'mol_2'):
# >>> rmsd_cc1_c1[1]['mol_1'].split('\n')[0]
# 'ATOM      1  N   GLU W 536      -8.864  -6.648 -21.167  1.00 50.97           N  '
pbd_io = PDBIO()
pars = PDBParser(QUIET=True)
rmsd_df_list = []

for target in tqdm(target_dirs, "Computing pymol RMSD caprieval"):
    m1_fp = list(target.glob(f'*_{STANDARD_CHAINS[0]}.pdb'))
    m2_fp = list(target.glob(f'*_{STANDARD_CHAINS[1]}.pdb'))
    cc_fp = list(target.glob(f"*_{''.join(STANDARD_CHAINS)}.pdb"))
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

    pymol_dir = target / 'pymol'
    chains_dir = pymol_dir / 'chains'
    pymol_dir.mkdir(parents=True, exist_ok=True)
    chains_dir.mkdir(parents=True, exist_ok=True)
    chain_keys = list(chains.keys())
    chain_paths: dict[str, PosixPath] = {}
    for k in chain_keys:
        cstru_tmp = assemble_structure(chains[k], structure_id=k)
        chain_path = chains_dir / f'{k}.pdb'
        pbd_io.set_structure(cstru_tmp)
        pbd_io.save(str(chain_path))
        chain_paths[k] = chain_path

    c1_keys = [k for k in chain_keys if k.startswith(('c1_', 'cc1_'))]
    c2_keys = [k for k in chain_keys if k.startswith(('c2_', 'cc2_'))]

    c1_aln_fp = pymol_dir / f"{'_'.join([str(chain_paths[k].stem) for k in c1_keys])}_pymol.pdb"
    c2_aln_fp = pymol_dir / f"{'_'.join([str(chain_paths[k].stem) for k in c2_keys])}_pymol.pdb"
    cc1_key = next(k for k in c1_keys if k.startswith('cc1_'))
    c1_key  = next(k for k in c1_keys if k.startswith('c1_'))
    cc2_key = next(k for k in c2_keys if k.startswith('cc2_'))
    c2_key  = next(k for k in c2_keys if k.startswith('c2_'))
    rmsd_cc1_c1 = align_pdb_structures(
        mol1_path=str(chain_paths[cc1_key]), 
        mol2_path=str(chain_paths[c1_key]), 
        return_structure=True
    )
    rmsd_cc2_c2 = align_pdb_structures(
        mol1_path=str(chain_paths[cc2_key]), 
        mol2_path=str(chain_paths[c2_key]), 
        return_structure=True
    )
    rmsd_df_list.append(pd.concat([rmsd_cc1_c1[0], rmsd_cc2_c2[0]]))

    c1_str = rmsd_cc1_c1[1]
    structure_cc = pars.get_structure(str(c1_aln_fp.stem), io.StringIO(c1_str['mol_1']))
    structure_c = pars.get_structure(str(c1_aln_fp.stem), io.StringIO(c1_str['mol_2']))
    str_cc = Selection.unfold_entities(structure_cc, 'C')[0]
    str_c = Selection.unfold_entities(structure_c, 'C')[0]
    str_c.id = str_c.id.lower()
    test = assemble_structure(str_cc, str_c, structure_id=str(c1_aln_fp.stem))
    pbd_io.set_structure(test)
    pbd_io.save(str(c1_aln_fp))

    c2_str = rmsd_cc2_c2[1]
    structure_cc2 = pars.get_structure(str(c2_aln_fp.stem), io.StringIO(c2_str['mol_1']))
    structure_c2 = pars.get_structure(str(c2_aln_fp.stem), io.StringIO(c2_str['mol_2']))
    str_cc2 = Selection.unfold_entities(structure_cc2, 'C')[0]
    str_c2 = Selection.unfold_entities(structure_c2, 'C')[0]
    str_c2.id = str_c2.id.lower()
    test2 = assemble_structure(str_cc2, str_c2, structure_id=str(c2_aln_fp.stem))
    pbd_io.set_structure(test2)
    pbd_io.save(str(c2_aln_fp))

rmsd_df = pd.concat(rmsd_df_list)
rmsd_df.to_csv(Path(METADATA_DIR) / 'dimer_monomer_pymol_RMSD.tsv', sep="\t", index=False)
############################## Section: <Str-rmsd>

########## Section: <contacts>
skipped_ct = []
for ht in tqdm(heterodimers, 'Computing distance and contacts'):
    mother = ht.parent
    basename = ht.stem
    cont_dir = mother / 'contacts'
    ofile = cont_dir / f'contacts_{basename}.tsv'
    if ofile.exists():
        skipped_ct.append(str(ofile.stem))
    if not ofile.exists():
        test = _merge_positional_data(mother)
        contacts_df = ca_contacts(ht, cutoff=DIST_CA)
        contacts_df = contacts_df.rename(columns={
            'resseq_i': 'dimer_resseq_i',
            'resseq_j': 'dimer_resseq_j'
        })
        for mapping_df in test:
            if mapping_df.empty:
                continue
            parts = mapping_df['query_id'].iloc[0].replace('_pdbseq', '').split('_')
            complex_chain = parts[-1]
            pdb_id = parts[1]
            lookup = (
                mapping_df
                .dropna(subset=['resid_complex', 'resid_monomer'])
                .drop_duplicates(subset=['resid_complex'])
                .astype({'resid_complex': int, 'resid_monomer': int})
                .set_index('resid_complex')['resid_monomer']
            )
            mask_i = contacts_df['chain_i'] == complex_chain
            contacts_df.loc[mask_i, 'monomer_resseq_i'] = contacts_df.loc[mask_i, 'dimer_resseq_i'].map(lookup)
            contacts_df.loc[mask_i, 'PDB_id_i'] = pdb_id
            mask_j = contacts_df['chain_j'] == complex_chain
            contacts_df.loc[mask_j, 'monomer_resseq_j'] = contacts_df.loc[mask_j, 'dimer_resseq_j'].map(lookup)
            contacts_df.loc[mask_j, 'PDB_id_j'] = pdb_id
        for col in ['dimer_resseq_i', 'dimer_resseq_j', 'monomer_resseq_i', 'monomer_resseq_j']:
            contacts_df[col] = contacts_df[col].astype('Int64')
        contacts_df = contacts_df[[
            'pdb_id', 'dist_nm', 'dist_A', 'PDB_id_i', 'PDB_id_j',
            'dimer_resseq_i', 'monomer_resseq_i', 'dimer_resseq_j', 'monomer_resseq_j',
            'chain_i',  'resname_i', 'res_i', 'aminoacid_i',
            'chain_j', 'resname_j', 'res_j', 'aminoacid_j'
        ]]
        cont_dir.mkdir(parents=True, exist_ok=True)
        contacts_df.to_csv(ofile, sep="\t", index=False)

if skipped_ct:
    print(f"[SKIP] Skipped {len(skipped_ct)} existing contact maps: {skipped_ct}")
############################## <!contacts>



