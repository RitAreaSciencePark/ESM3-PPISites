#!/usr/bin/env python3
import os
import glob
import subprocess

HADDOCK_UNITS = "data/haddock_units"
SCRIPT = "scripts/remap_contacts.py"

for dir_path in glob.glob(os.path.join(HADDOCK_UNITS, "*/")):
    dir_contacts = os.path.join(dir_path, "contacts")
    dir_aln = os.path.join(dir_path, "aln")
    
    if os.path.exists(dir_contacts) and os.path.exists(dir_aln):
        print(f"Remapping: {dir_path}")
        subprocess.run([
            "python3", SCRIPT, 
            dir_contacts, 
            dir_aln
        ], check=True)
    else:
        print(f"Skipping: Missing directories in {dir_path}")

print("All processing complete!")

