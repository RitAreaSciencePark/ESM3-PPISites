#!/usr/bin/env python3

### <deps>
#import os
import re
import io
import sys
import numpy as np
import pandas as pd
#import urllib
from pathlib import Path, PosixPath
#import warnings
#from collections import Counter, defaultdict
#from typing import Union, List
from tqdm import tqdm
#from types import SimpleNamespace
#import tempfile
#import hashlib

from config_defaults import * ### CONSTANTS
#from PDButils import *        ### scientific data manipulation
## <!deps>

########## <PRIVATE FUNCTIONS>
def _parse_inference(fp: str, keep_seq: bool = False):
    df = pd.read_csv(fp)
    if not df["id"].is_unique:
        raise ValueError("duplicate ids found")
    exploded = (
        df.assign(probability=df["probabilities"].str.split(" "))
        .explode("probability", ignore_index=True)
    )
    exploded["probability"] = exploded["probability"].astype(float)
    exploded["position"] = exploded.groupby("id").cumcount() + 1
    result = exploded[["id", "position", "probability"]]
    if keep_seq:
        return result, df.set_index("id")["sequence"]
    return result
########## <PRIVATE FUNCTIONS>

########## <parsing>
res_fp = Path(INFERENCE_OUTPUT)
lookup_fp = Path(METADATA_FP)
df_res = _parse_inference(res_fp)
res_fp = Path(MATCHED_INFERENCE_FP)
res_fp.parent.mkdir(parents=True, exist_ok=True)
df_res['monomer'] = df_res['id'].str.replace('_pdbseq', '')

df_lookup = pd.read_csv(lookup_fp, sep="\t")
long_lookup = pd.melt(
    df_lookup,
    id_vars=['jobname', 'c1_rename', 'c2_rename'],
    value_vars=['PDB_c1', 'PDB_c2'],
    var_name='chain_type',
    value_name='PDB'
)

long_lookup['c_name'] = long_lookup.apply(
    lambda row: row['c1_rename'] if row['chain_type'] == 'PDB_c1' else row['c2_rename'],
    axis=1
)

long_lookup = long_lookup[['jobname', 'PDB', 'c_name']].dropna(subset=['PDB'])

long_lookup['monomer'] = long_lookup['PDB'] + "_" + long_lookup['c_name']

df_res = df_res.merge(
    long_lookup[['monomer', 'jobname']],
    on='monomer',
    how='left'
)
jobname_counts = df_res.groupby('jobname')['monomer'].nunique()
invalid_jobs = jobname_counts[jobname_counts != 2]
if not invalid_jobs.empty:
    print('[FAIL] exactly two monomers expected for this workflow')
    exit(1)

df_res.to_csv(MATCHED_INFERENCE_FP, sep="\t", index=False)
print(f"Parsed inference results have been saved to: {MATCHED_INFERENCE_FP}")
############################## <!parsing>

