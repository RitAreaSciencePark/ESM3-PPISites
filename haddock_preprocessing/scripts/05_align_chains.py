#!/usr/bin/env python3
import os
import glob
import subprocess

HADDOCK_UNITS = "data/haddock_units"
SCRIPT = "scripts/align_PDB_seqs.py"

# Iterate over all directories
for dir_path in glob.glob(os.path.join(HADDOCK_UNITS, "*/")):
    unit = os.path.basename(os.path.dirname(dir_path))
    
    # Input path (residing in the 'residues' subdirectory)
    residues_dir = os.path.join(dir_path, "residues")
    complex_file = os.path.join(residues_dir, f"{unit}_WZ_residues.tsv")
    
    # Define output directory and ensure it exists
    output_dir = os.path.join(dir_path, "aln")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find chain files within the 'residues' subdirectory
    search_pattern = os.path.join(residues_dir, "*_residues.tsv")
    for chain_file in glob.glob(search_pattern):
        # Skip the complex file itself to avoid self-alignment
        if os.path.basename(chain_file) == f"{unit}_WZ_residues.tsv":
            continue
            
        # Get base name
        chain_filename = os.path.basename(chain_file)
        chain_base = chain_filename.replace("_residues.tsv", "")
        
        # Output path inside 'aln' directory
        output_file = os.path.join(output_dir, f"{unit}_WZ_vs_{chain_base}.tsv")
        
        print(f"Aligning: {chain_base} -> {output_file}")
        
        # Execute
        subprocess.run([
            "python3", SCRIPT,
            "--complex", complex_file,
            "--chain", chain_file,
            "--output", output_file
        ], check=True)

print("All processing complete!")

