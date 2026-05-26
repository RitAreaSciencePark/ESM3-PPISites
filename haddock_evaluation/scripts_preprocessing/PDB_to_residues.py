"""
Extract sequence positions for each protein chain from a PDB file.
Standard and modified amino acids from ATOM/HETATM records are kept.
Modified residues are mapped to their standard equivalent; the original
name is recorded in the is_modified column. Waters, ligands, and truly
unknown residues are skipped.
Only the first MODEL is processed. Insertion codes are preserved.

Usage: python PDB_to_residues.py <file.pdb> [--output <file_posmatch.tsv>]
"""

import sys
import os
import csv
import argparse

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# Modified residue → standard residue mapping
MODIFIED_TO_STD = {
    # Methionine derivatives
    'MSE': 'MET', 'FME': 'MET', 'CXM': 'MET', 'OMT': 'MET',
    # Selenocysteine / Cysteine derivatives
    'SEC': 'CYS', 'CSO': 'CYS', 'CSS': 'CYS', 'CSX': 'CYS',
    'CME': 'CYS', 'CSD': 'CYS', 'CSP': 'CYS', 'CEA': 'CYS',
    'CAF': 'CYS', 'SCH': 'CYS', 'SCS': 'CYS', 'SCY': 'CYS',
    'OCS': 'CYS', 'CMT': 'CYS',
    # Serine derivatives
    'SEP': 'SER', 'SVA': 'SER', 'SAC': 'SER', 'MIS': 'SER',
    # Threonine derivatives
    'TPO': 'THR', 'BMT': 'THR',
    # Tyrosine derivatives
    'PTR': 'TYR', 'TYS': 'TYR', 'NIY': 'TYR', 'PAQ': 'TYR',
    'DTY': 'TYR', 'IYR': 'TYR',
    # Lysine derivatives
    'MLY': 'LYS', 'MLZ': 'LYS', 'LYZ': 'LYS', 'ALY': 'LYS',
    'M3L': 'LYS', 'KCX': 'LYS', 'LLP': 'LYS', 'LYM': 'LYS',
    'TRG': 'LYS', 'SHR': 'LYS',
    # Arginine derivatives
    'ARM': 'ARG', 'ARO': 'ARG',
    # Aspartate / Asparagine derivatives
    'ASX': 'ASN', 'PHD': 'ASP', 'BHD': 'ASP',
    # Glutamate / Glutamine derivatives
    'GLX': 'GLN', 'PCA': 'GLU', 'CGU': 'GLU',
    # Histidine derivatives
    'NEP': 'HIS', 'HIC': 'HIS', 'HID': 'HIS', 'HIE': 'HIS',
    'HIP': 'HIS', 'MHS': 'HIS',
    # Proline derivatives
    'HYP': 'PRO', 'DPR': 'PRO',
    # Tryptophan derivatives
    'HTR': 'TRP', 'TOX': 'TRP',
    # Phenylalanine derivatives
    'PHI': 'PHE', 'PHL': 'PHE',
    # Leucine / Isoleucine derivatives
    'MVA': 'VAL', 'DIV': 'VAL',
    'NLE': 'LEU', 'MLE': 'LEU',
    # Alanine derivatives
    'DAL': 'ALA', 'MAA': 'ALA',
    # Glycine derivatives
    'GL3': 'GLY', 'SAR': 'GLY',
}


def resolve_residue(resname):
    """Return (standard_resname, is_modified_flag).
    Returns (None, None) if the residue is not a known AA or modified AA."""
    if resname in AA3TO1:
        return resname, ''
    if resname in MODIFIED_TO_STD:
        return MODIFIED_TO_STD[resname], resname
    return None, None


def parse_pdb(path):
    pdb_id = os.path.basename(path).replace('.pdb', '')
    seen = set()
    rows = []
    in_model = False

    with open(path) as f:
        for line in f:
            record = line[:6].strip()

            if record == 'MODEL':
                if in_model:
                    break       # second MODEL: stop
                in_model = True
                continue

            if record == 'ENDMDL':
                break

            if record not in ('ATOM', 'HETATM'):
                continue

            resname  = line[17:20].strip()
            std_res, is_modified = resolve_residue(resname)
            if std_res is None:
                continue        # ligand, water, or unknown

            chain    = line[21]
            resnum   = line[22:26].strip()
            icode    = line[26].strip()
            position = resnum + icode

            key = (chain, position)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                'pdb':         pdb_id,
                'chain':       chain,
                'aminoacid':   std_res,
                'aa':          AA3TO1[std_res],
                'position':    position,
                'is_modified': is_modified,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', help='Input PDB file')
    parser.add_argument('--output', default=None,
                        help='Output TSV (default: <basename>_posmatch.tsv)')
    args = parser.parse_args()

    out_path = args.output or args.pdb.replace('.pdb', '_posmatch.tsv')

    rows = parse_pdb(args.pdb)
    if not rows:
        print("WARNING: no amino acid ATOM/HETATM records found.", file=sys.stderr)
        sys.exit(1)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['pdb', 'chain', 'aminoacid', 'aa', 'position', 'is_modified'],
            delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(rows)

    n_mod = sum(1 for r in rows if r['is_modified'])
    chains = sorted(set(r['chain'] for r in rows))
    print(f"Written {len(rows)} residues ({n_mod} modified) across chains {chains} → {out_path}")


if __name__ == '__main__':
    main()

