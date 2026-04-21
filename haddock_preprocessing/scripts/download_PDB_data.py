#!/usr/bin/env python3
"""
pdb_download.py

Downloads mmCIF (default), FASTA, and PDB files for a PDB entry from RCSB.

Usage:
    python pdb_download.py 1ABC output_dir/ --fasta --pdb
"""

import argparse
import urllib.request
import sys
from pathlib import Path

# RCSB URL Templates
RCSB_MMCIF = "https://files.rcsb.org/download/{id}.cif"
RCSB_FASTA = "https://www.rcsb.org/fasta/entry/{id}"
RCSB_PDB   = "https://files.rcsb.org/download/{id}.pdb"

def download(url: str, dest: Path):
    """Downloads a file from a URL to a destination path."""
    try:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to:    {dest}")
    except Exception as e:
        print(f"  [ERROR] Could not download {url}: {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Download structural files from RCSB PDB."
    )
    # Positional arguments
    ap.add_argument("pdb_id", help="4-character PDB ID (e.g. 1ABC)")
    ap.add_argument("output_directory", help="Directory to write files into")
    
    # Optional flags
    ap.add_argument("--fasta", action="store_true", help="Download the FASTA file")
    ap.add_argument("--pdb", action="store_true", help="Download the PDB format file (if available)")
    
    args = ap.parse_args()

    # Clean up PDB ID
    pdb_id = args.pdb_id.strip().upper()
    if len(pdb_id) != 4:
        print(f"Error: '{pdb_id}' does not look like a standard 4-character PDB ID.")
        sys.exit(1)

    outdir = Path(args.output_directory).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing {pdb_id} ---")

    # 1. mmCIF (Always downloaded)
    download(RCSB_MMCIF.format(id=pdb_id), outdir / f"{pdb_id}.cif")

    # 2. FASTA (Optional)
    if args.fasta:
        download(RCSB_FASTA.format(id=pdb_id), outdir / f"{pdb_id}.fasta")

    # 3. PDB (Optional)
    if args.pdb:
        download(RCSB_PDB.format(id=pdb_id), outdir / f"{pdb_id}.pdb")

    print("\nDone.")

if __name__ == "__main__":
    main()


