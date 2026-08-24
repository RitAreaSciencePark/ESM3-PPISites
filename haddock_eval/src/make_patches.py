#!/usr/bin/env python3

### <deps>
#import os
import re
import io
import sys
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
from tqdm import tqdm

from config_defaults import * ### CONSTANTS
#from PDButils import *        ### scientific data manipulation
## <!deps>

########## <PRIVATE FUNCTIONS>
def extract_patches(df, res_col, max_dist):
    if df.empty:
        df['patch_id'] = []
        return df
    df = df.sort_values(by=res_col).reset_index(drop=True)
    patches = []
    current_patch = 1
    residues = df[res_col].tolist()
    if residues:
        patches.append(str(current_patch)) 
        for i in range(1, len(residues)):
            if residues[i] - residues[i - 1] > max_dist:
                current_patch += 1
            patches.append(str(current_patch))
    df['patch_id'] = patches
    return df

def _wrap_patches(df, min_res_clustering=0, min_patch_length=0):
    expected_columns = [
        'jobname',
        'id',
        'monomer',
        'patch_id',
        'avg_probability',
        'patch_size',
        'residues',
        'seq_indexes'
    ]
    #return_cols = ['jobname', 'id','monomer', 'patch_rank', 'patch_id', 'avg_probability', 'patch_size', 'residues']
    if df['jobname'].nunique() > 1:
        raise ValueError("Expected a single unique jobname")
    jobname = df['jobname'].unique()[0]
    if df['monomer'].nunique() > 1:
        raise ValueError("Expected a single unique monomer")
    monomer_id = df['monomer'].unique()[0]
    seq_id = df['id'].unique()[0]

    num_passing = len(df)
    if num_passing == 0:
        print(f"[ERROR] in {monomer_id} no residues passed cutoff {PROB_CUTOFF}")

    too_few_residues = num_passing < min_res_clustering
    
    # Initialize variables for safe fallback if too_few_residues is True
    biggest_patch = 0
    patches_too_short = True
    concatenated_patch_sizes = ""

    if not too_few_residues:
        patches_dryrun = (
            df.groupby(['jobname', 'id', 'monomer', 'patch_id'])
            .agg(
                avg_probability=('probability', 'mean'),
                patch_size=('probability', 'count'),
                residues=('resid', lambda x: ','.join(map(str, x))),
                seq_indexes=('position', lambda x: ','.join(map(str, x))),
            )
            .reset_index()
        )
        patches_dryrun = patches_dryrun.sort_values(by='avg_probability', ascending=False).reset_index(drop=True)
        #patches_dryrun['patch_rank'] = range(1, len(patches_dryrun) + 1)        
        biggest_patch = max(patches_dryrun['patch_size'])
        patches_too_short = biggest_patch < min_patch_length
        
        # --- NEW: Create comma-separated string of patch sizes ---
        pass_patches = patches_dryrun[patches_dryrun['patch_size']>=min_patch_length]
        pass_patches['patch_rank'] = range(1, len(pass_patches) + 1) 
        pass_patches_sizes = pass_patches['patch_size']
        concatenated_patch_sizes = ','.join(pass_patches_sizes.astype(str))
        top_patches = pass_patches.nlargest(N_TOP_PATCHES, 'avg_probability')
        top_patches_sizes = top_patches['patch_size']
        concatenated_top_sizes = ','.join(top_patches_sizes.astype(str))
        min_avg_prob = top_patches['avg_probability'].min()

    validation = pd.DataFrame([{
        "id":seq_id,
        "threshold": PROB_CUTOFF, 
        "max_res_distance": MAX_RES_DISTANCE,
        "min_to_cluster": min_res_clustering, 
        "min_patch_length": min_patch_length, 
        "jobname": jobname, 
        "monomer": monomer_id, 
        "pass_threshold": num_passing, 
        "too_few_passing_filter": too_few_residues, 
        "biggest_patch": biggest_patch, 
        "patches_too_short": patches_too_short,
        "patch_sizes": concatenated_patch_sizes,  # --- Added here ---
        "top2_sizes": concatenated_top_sizes,  # --- Added here ---
        "top2_min_prob": min_avg_prob, 
    }])
    #print(validation)
    return pass_patches, top_patches, validation


############################## <!PRIVATE FUNCTIONS>

########## <patches>
res_wide_fp = Path(INFERENCE_OUTPUT)
df_seqs = pd.read_csv(res_wide_fp)
df_seqs['seq_len'] = df_seqs['sequence'].astype(str).str.len()

res_wide = Path(INFERENCE_OUTPUT)
res_long_fp = Path(MATCHED_INFERENCE_FP)
patches_fp = Path(PATCHES_FP)
patches_fp.parent.mkdir(parents=True, exist_ok=True)
df_res = pd.read_csv(res_long_fp, sep="\t")

patches_df_list = []
top_patches_list = []
validation_df_list = []
indexed_long_res = []

for seq_id in set(df_res['monomer']):
    #seq_id = str(df_res['monomer'].iloc[0])
    prob_array = df_res[df_res['monomer'] == seq_id]
    curr_job = str(prob_array['jobname'].unique()[0])                       #F# dangerous but works
    path_pdbseqs = Path(Path(HADDOCK_BASEDIR) / str(curr_job) / 'PDB_seqs')
    map_fp = list(path_pdbseqs.glob(f"*{seq_id}_map.tsv"))
    if len(map_fp) != 1:
        raise ValueError('Sequence to PDB mapping .tsv must be unique')
    
    index_map = pd.read_csv(map_fp[0], sep="\t")
    index_map['position'] = index_map['seq_index'] + 1
    index_map = index_map.drop(columns=['seq_index'])
    
    #print(index_map.head())
    
    prob_array = prob_array.merge(index_map, on='position', how='left')
    indexed_long_res.append(prob_array)
    sele_array = prob_array[prob_array['probability'] >= PROB_CUTOFF]

    prob_peak = max(sele_array['probability'])
    if prob_peak < 0.8:
        sele_array = prob_array[prob_array['probability'] >= LOWER_PROB_CUTOFF]

    patches_long = extract_patches(sele_array, res_col='position',
                        max_dist = MAX_RES_DISTANCE)
    #print(patches_long.head(8))
    #print(patches_long)
    wrapped_patches = _wrap_patches(patches_long, 
                          min_res_clustering=MIN_RES_CLUSTERING,
                          min_patch_length=MIN_PATCH_LENGTH)
    print(f"{seq_id}: {len(wrapped_patches[1])}")
    if len(wrapped_patches[1]) < MIN_NUM_PATCHES:
        wrapped_patches = _wrap_patches(patches_long, 
            min_res_clustering=ALT_RES_CLUSTERING,
            min_patch_length=ALT_PATCH_LENGTH)
    print(f"{seq_id}: {len(wrapped_patches[1])}")
    curr_patches = wrapped_patches[0]
    #print(curr_patches.head(8))
    curr_patches = curr_patches.merge(df_seqs[['id', 'seq_len']], on='id', how='left')
    curr_top_patches = wrapped_patches[1]
    curr_top_patches = curr_top_patches.merge(df_seqs[['id', 'seq_len']], on='id', how='left')
    curr_patchdata = wrapped_patches[2]
    curr_patchdata = curr_patchdata.merge(df_seqs[['id', 'seq_len']], on='id', how='left')
    #print(wrapped_patches)
    patches_df_list.append(curr_patches)
    top_patches_list.append(curr_top_patches)
    validation_df_list.append(curr_patchdata)

patches_df = pd.concat(patches_df_list)
top_patches_df = pd.concat(top_patches_list)
validation_df = pd.concat(validation_df_list)
indexed_df = pd.concat(indexed_long_res)

patches_fp = Path(PATCHES_FP)
top_patches_fp = Path(FOR_PAIRING_FP)
patchdata_fp = Path(PATCHDATA_FP)
indexed_fp = Path(MAPPED_INFERENCE_FP)

patches_df.to_csv(patches_fp, sep="\t", index=False)
top_patches_df.to_csv(top_patches_fp, sep="\t", index=False)
validation_df.to_csv(patchdata_fp, sep="\t", index=False)
indexed_df.to_csv(indexed_fp, sep="\t", index=False)


