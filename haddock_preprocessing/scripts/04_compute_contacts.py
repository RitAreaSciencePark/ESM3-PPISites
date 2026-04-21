#!/usr/bin/env python3
import os
import glob
import subprocess

HADDOCK_UNITS = "data/haddock_units"
SCRIPT = "scripts/contact_map_dimer.py"
CUTOFF = 0.8

# Find all *_WZ.pdb files
pattern = os.path.join(HADDOCK_UNITS, "*", "*_WZ.pdb")
for pdb_file in glob.glob(pattern):
    # Get directory and base name
    dir_path = os.path.dirname(pdb_file)
    base_name = os.path.basename(pdb_file)[:-4]  # Remove .pdb
    
    # 1. Define output directory and path
    output_dir = os.path.join(dir_path, "contacts")
    output_file = os.path.join(output_dir, f"contacts_{base_name}.tsv")
    
    # 2. Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {pdb_file} -> {output_file}")
    
    # 3. Execute
    subprocess.run([
        "python3", SCRIPT,
        pdb_file,
        "--cutoff", str(CUTOFF),
        "--output", output_file
    ], check=True)

print("All processing complete!")

