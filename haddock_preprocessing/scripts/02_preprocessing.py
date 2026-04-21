#!/usr/bin/env python3
"""
Process all HADDOCK unit directories by reading the TSV reference file.

For each directory in haddock_units/:
  Step 1 — Extract single chains from unbound PDBs:
    chain1_PDBchains → keep its chain, rename to W → <pdbID>_W.pdb
    chain2_PDBchains → keep its chain, rename to Z → <pdbID>_Z.pdb

  Step 2 — Extract both chains from the complex PDB:
    complex_PDBchains → keep chain1→W and chain2→Z → <pdbID>_WZ.pdb

Output goes to data/haddock_units/<complex_PDB>/.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

RENAME_SCRIPT = Path(__file__).parent / 'rename_single_chain.py'


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------

def run_rename(src_pdb, chain, dst_pdb, rename_to):
    """Call rename_single_chain.py for a single chain extraction."""
    cmd = [
        sys.executable, str(RENAME_SCRIPT),
        '--complex', str(src_pdb),
        '--chain',   chain,
        '--rename-to', rename_to,
        '--output',  str(dst_pdb),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rename_single_chain.py failed for {src_pdb}:\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )


def run_rename_two_chains(src_pdb, chain1, chain2, dst_pdb):
    """
    Call rename_single_chain.py twice (once per chain) then merge the
    two single-chain PDBs into one WZ file using BioPython.
    """
    from Bio import PDB

    tmp_w = dst_pdb.parent / f"_tmp_W_{dst_pdb.name}"
    tmp_z = dst_pdb.parent / f"_tmp_Z_{dst_pdb.name}"

    try:
        run_rename(src_pdb, chain1, tmp_w, rename_to='W')
        run_rename(src_pdb, chain2, tmp_z, rename_to='Z')

        # Merge the two single-chain structures into one file
        p = PDB.PDBParser(QUIET=True)
        struct_w = p.get_structure('W', str(tmp_w))
        struct_z = p.get_structure('Z', str(tmp_z))

        # Add chain Z into struct_w's model
        for model_w, model_z in zip(struct_w, struct_z):
            for chain in model_z:
                model_w.add(chain)

        io = PDB.PDBIO()
        io.set_structure(struct_w)
        io.save(str(dst_pdb))
    finally:
        tmp_w.unlink(missing_ok=True)
        tmp_z.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TSV parsing helpers
# ---------------------------------------------------------------------------

def parse_pdb_chain(field, default_chain='A'):
    if '_' in field:
        pdb_id, chain_part = field.split('_', 1)
        chain_id = chain_part.strip() if chain_part.strip() else default_chain
    else:
        pdb_id = field.strip()
        chain_id = default_chain
    return pdb_id, chain_id


def parse_complex_chains(field):
    pdb_part, chains_part = field.split('_', 1)
    chain1, chain2 = chains_part.split(':')
    return pdb_part, chain1.strip(), chain2.strip()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_unit(row, haddock_dir, output_base, dry_run=False):
    complex_pdb   = row['complex_PDB']
    chain1_field  = row['chain1_PDBchains']
    chain2_field  = row['chain2_PDBchains']
    complex_field = row['complex_PDBchains']

    unit_in_dir  = haddock_dir / complex_pdb
    unit_out_dir = output_base / complex_pdb
    if not dry_run:
        unit_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1a: chain1 → W -----------------------------------------------
    chain1_pdb_id, chain1_id = parse_pdb_chain(chain1_field)
    chain1_src = unit_in_dir / f"{chain1_pdb_id}.pdb"
    chain1_dst = unit_out_dir / f"{chain1_pdb_id}_W.pdb"

    print(f"  [Step 1a] {chain1_src.name}: chain '{chain1_id}' → W  →  {chain1_dst.name}")
    if not dry_run:
        run_rename(chain1_src, chain1_id, chain1_dst, rename_to='W')

    # ---- Step 1b: chain2 → Z -----------------------------------------------
    chain2_pdb_id, chain2_id = parse_pdb_chain(chain2_field)
    chain2_src = unit_in_dir / f"{chain2_pdb_id}.pdb"
    chain2_dst = unit_out_dir / f"{chain2_pdb_id}_Z.pdb"

    print(f"  [Step 1b] {chain2_src.name}: chain '{chain2_id}' → Z  →  {chain2_dst.name}")
    if not dry_run:
        run_rename(chain2_src, chain2_id, chain2_dst, rename_to='Z')

    # ---- Step 2: complex → WZ ----------------------------------------------
    cpx_pdb_id, cpx_chain1, cpx_chain2 = parse_complex_chains(complex_field)
    cpx_src = unit_in_dir / f"{cpx_pdb_id}.pdb"
    cpx_dst = unit_out_dir / f"{cpx_pdb_id}_WZ.pdb"

    print(f"  [Step 2 ] {cpx_src.name}: chains '{cpx_chain1}'→W, '{cpx_chain2}'→Z  →  {cpx_dst.name}")
    if not dry_run:
        run_rename_two_chains(cpx_src, cpx_chain1, cpx_chain2, cpx_dst)


def main():
    parser = argparse.ArgumentParser(
        description='Process HADDOCK unit directories using the TSV reference.'
    )
    parser.add_argument('--tsv', default='raw-data/zlab_dbdock_dimers_only.tsv')
    parser.add_argument('--haddock-dir', default='raw-data/haddock_units')
    parser.add_argument('--output-dir', default='data/haddock_units')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--units', nargs='*', metavar='COMPLEX_PDB')
    args = parser.parse_args()

    tsv_path    = Path(args.tsv)
    haddock_dir = Path(args.haddock_dir)
    output_base = Path(args.output_dir)

    with tsv_path.open() as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

    tsv_lookup = {row['complex_PDB']: row for row in rows}

    if args.units:
        unit_dirs = [haddock_dir / u for u in args.units]
    else:
        unit_dirs = sorted(p for p in haddock_dir.iterdir() if p.is_dir())

    errors = []
    for unit_dir in unit_dirs:
        complex_pdb = unit_dir.name
        if complex_pdb not in tsv_lookup:
            print(f"[SKIP] {complex_pdb}: no matching row in TSV")
            continue

        print(f"\n[PROCESSING] {complex_pdb}")
        try:
            process_unit(tsv_lookup[complex_pdb], haddock_dir, output_base, dry_run=args.dry_run)
            print(f"  → done")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append((complex_pdb, exc))

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Finished. "
          f"{len(unit_dirs) - len(errors)} OK, {len(errors)} errors.")
    if errors:
        for name, exc in errors:
            print(f"  FAILED: {name} — {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
