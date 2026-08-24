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
############################## <!PRIVATE FUNCTIONS>


########## <pairing>

chain_1 = STANDARD_CHAINS[0]
chain_2 = STANDARD_CHAINS[1]
for_pairing_fp = Path(FOR_PAIRING_FP)
filtered_df = pd.read_csv(for_pairing_fp, sep="\t")

all_pairs = []
for jobname in set(filtered_df['jobname']):
    df_top = filtered_df[filtered_df['jobname']==jobname]
    #print(df_top.head())
    df_c1 = df_top[df_top['monomer'].str.endswith(chain_1)]
    df_c2 = df_top[df_top['monomer'].str.endswith(chain_2)]
#    print(df_c1.head())
    df_pairs = pd.merge(df_c1, df_c2, on='jobname',
        suffixes=('_c1', '_c2'))
    all_pairs.append(df_pairs)
    for _, row in df_pairs.iterrows():
        jobname = row['jobname']
        # Setup paths
        id_path = Path(HADDOCK_BASEDIR) / jobname
        tbl_dir = id_path / "tbls"
        config_dir = id_path / "configs"
        tbl_dir.mkdir(exist_ok=True, parents=True)
        config_dir.mkdir(exist_ok=True, parents=True)

        # i and j represents patches id for each chain'
        i = str(row['patch_rank_c1']) #.split('_')[-1]
        j = str(row['patch_rank_c2']) #.split('_')[-1]

        # Convert string representations of lists to actual lists of ints
        p_w = [int(x) for x in str(row['residues_c1']).split(',')]
        p_z = [int(x) for x in str(row['residues_c2']).split(',')]

        # Logic for alignment
        is_w_longer = len(p_w) >= len(p_z)
        long, short = (p_w, p_z) if is_w_longer else (p_z, p_w)
        c_long, c_short = (chain_1, chain_2) if is_w_longer else (chain_2, chain_1)

        s = (len(long) - len(short)) // 2

        # 4. Orientation loop (k=0 is forward, k=1 is reverse)
        for k, sequence in enumerate([short, short[::-1]]):
            # Naming convention: {patchW}_{patchZ}_{orientation}
            file_name = f"p_{i}_{j}_{k}"
            tbl_file = tbl_dir / f"{file_name}.tbl"
            config_file = config_dir / f"{file_name}_noflex.cfg"

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
                run_dir = id_path / "runs"
                run_dir.mkdir(parents=True, exist_ok=True)
                patch = tbl_file.stem
                id_path = Path(id_path)
                tbl_file = Path(tbl_file)

                run_dir = id_path / "runs"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Equivalent to $(basename $tbl_file | sed 's/.tbl//g')
                patch = tbl_file.stem

                # Find the _W and _Z PDB files in the id_path directory
                prot_w_list = list(id_path.glob("*_W.pdb"))
                prot_z_list = list(id_path.glob("*_Z.pdb"))

                if not prot_w_list or not prot_z_list:
                    print(f"Error: Missing '_W.pdb' or '_Z.pdb' in {id_path}")
                    sys.exit(1)

                prot_w = prot_w_list[0].name
                prot_z = prot_z_list[0].name

                run_name = f"{patch}_noflex"
                flexref_block = ""

                # Construct the config contents
                config_content = f"""run_dir = "{run_dir / run_name}"
        molecules = ["{id_path / prot_w}","{id_path / prot_z}"]
        ncores = {DEF_NCORES}

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

        {flexref_block}[caprieval]
        reference_fname = "{id_path / f'{id_path.name}_WZ.pdb'}"
        fnat_cutoff = {FNAT_CUTOFF}
        """

                config_file.write_text(config_content)
                print(f"Created configuration file at: {config_file}")


            
########## <!pairing>

