#!/usr/bin/env python3
"""
Process HADDOCK unit directory to create filtered PDB file with single chain.
Specify chain to keep and the target chain name to rename to from command line.
"""
import argparse
import sys
from Bio import PDB


def _renumber_chain(chain):
    """Renumber residues 1 → N (two-pass to avoid ID collisions)."""
    residues = list(chain.get_residues())
    # Pass 1: shift to a safe high range unlikely to exist
    for i, residue in enumerate(residues):
        residue.id = (' ', 100000 + i, ' ')
    # Pass 2: assign final sequential IDs
    for i, residue in enumerate(residues):
        residue.id = (' ', i + 1, ' ')

def process_pdb_single_chain(complex_pdb_path, chain_to_keep, output_path, rename_to):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', complex_pdb_path)

    io = PDB.PDBIO()

    for model in structure:
        chains_to_remove = [c.id for c in model if c.id != chain_to_keep]
        for cid in chains_to_remove:
            model.detach_child(cid)
        for chain in model:
            chain.id = rename_to
            het_ids = [r.id for r in chain if r.id[0] != ' ']
            for rid in het_ids:
                chain.detach_child(rid)
            _renumber_chain(chain)

    io.set_structure(structure)
    io.save(str(output_path))

    return 1


def main():
    parser = argparse.ArgumentParser(
        description='Filter complex PDB to keep only a single specified chain, renamed to a target ID'
    )
    parser.add_argument('--complex', required=True,
                        help='Path to complex PDB file')
    parser.add_argument('--chain', required=True,
                        help='Chain ID in complex to keep (e.g., A)')
    parser.add_argument('--rename-to', default='W',
                        help='Chain ID to rename the kept chain to (default: W)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output PDB file path')

    args = parser.parse_args()

    print(f"Processing: {args.complex}")
    print(f"  Keeping chain '{args.chain}' and renaming to '{args.rename_to}'")

    try:
        process_pdb_single_chain(args.complex, args.chain, args.output, rename_to=args.rename_to)
        print(f"  Saved: {args.output}")
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()