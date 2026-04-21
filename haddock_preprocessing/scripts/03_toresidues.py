#!/usr/bin/env python3
import os
import subprocess

HADDOCK_UNITS = "data/haddock_units"
SCRIPT = "scripts/PDB_to_residues.py"

for root, dirs, files in os.walk(HADDOCK_UNITS):
    for file in files:
        if file.endswith(".pdb"):
            pdb_file = os.path.join(root, file)
            
            # 1. Directory path
            dir_name = os.path.dirname(pdb_file)
            
            # 2. Filename without .pdb
            base_name = file[:-4]  # Remove .pdb
            
            # 3. Output path
            output_dir = os.path.join(dir_name, "residues")
            output_tsv = os.path.join(output_dir, f"{base_name}_residues.tsv")
            
            # Create the directory
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing: {base_name} -> {output_tsv}")
            
            # 4. Execute
            subprocess.run([
                "python3", SCRIPT, 
                pdb_file, 
                "--output", output_tsv
            ], check=True)
