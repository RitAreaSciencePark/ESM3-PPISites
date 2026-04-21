#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

TSV_FILE = Path("raw-data/zlab_dbdock_dimers_only.tsv")
BASE_OUTPUT = Path("raw-data/haddock_units")

def download_pdb(pdb_id: str, output_dir: Path) -> None:
    """Call the external download script if file doesn't exist."""
    pdb_file = output_dir / f"{pdb_id}.pdb"
    
    if pdb_file.exists():
        print(f"  Skipping {pdb_id}.pdb (already exists)")
        return
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_PDB_data.py",
            pdb_id,
            str(output_dir),
            "--pdb"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  Error downloading {pdb_id}: {result.stderr}", file=sys.stderr)
    else:
        print(f"  Downloaded {pdb_id}.pdb")


def process_tsv(tsv_path: Path) -> None:
    """Parse TSV and orchestrate downloads for each complex."""
    with open(tsv_path, "r") as f:
        next(f)  # Skip header
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split("\t")
            if len(parts) < 7:
                continue  # or raise ValueError(f"Malformed line: {line}")
                
            _type, complex_pdb, chain1_pdb, chain2_pdb, complex_id, chain1, chain2 = parts[:7]
            
            output_dir = BASE_OUTPUT / complex_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing: {complex_id} (folder: {complex_id})")
            
            # Download all three PDB files
            for pdb_id in [complex_id, chain1, chain2]:
                download_pdb(pdb_id, output_dir)
    
    print("Download complete!")


if __name__ == "__main__":
    process_tsv(TSV_FILE)


