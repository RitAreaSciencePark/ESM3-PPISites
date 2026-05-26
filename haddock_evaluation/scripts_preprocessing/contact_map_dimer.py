#!/usr/bin/env python3
"""
Fast contact map calculator using MDTraj.
Uses CA-CA distances, computes only inter-chain contacts.
"""
import argparse
import mdtraj as md
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description='Calculate inter-chain CA contact map from PDB')
    parser.add_argument('pdb_file', help='Input PDB file')
    parser.add_argument('-o', '--output', default='contacts.tsv', help='Output TSV file')
    parser.add_argument('--cutoff', type=float, default=None, help='Contact cutoff in nm')

    args = parser.parse_args()

    # Load structure
    traj = md.load(args.pdb_file)
    top = traj.topology

    # Get CA atoms, grouped by residue
    ca_atoms = [a for a in top.atoms if a.name == 'CA']
    print(f"Found {len(ca_atoms)} CA atoms")

    if len(ca_atoms) < 2:
        print("Error: Fewer than 2 CA atoms found.")
        return

    # Generate pairs only between different chains
    pairs = []
    for i, ai in enumerate(ca_atoms):
        for j, aj in enumerate(ca_atoms[i+1:], start=i+1):
            if ai.residue.chain.index != aj.residue.chain.index:
                pairs.append((ai.index, aj.index))

    pairs = np.array(pairs)
    print(f"Computing {len(pairs)} inter-chain CA-CA distances...")

    if len(pairs) == 0:
        print("No inter-chain pairs found.")
        return

    # Compute distances
    dists = md.compute_distances(traj, pairs)[0]  # shape (n_pairs,)

    # Filter by cutoff if specified
    if args.cutoff is not None:
        mask = dists <= args.cutoff
        pairs = pairs[mask]
        dists = dists[mask]
        print(f"Kept {len(pairs)} pairs within {args.cutoff} nm cutoff")

    # Write main TSV
    with open(args.output, 'w') as f:
        f.write("res_i\tres_j\tchain_i\tchain_j\tresname_i\tresname_j\t"
                "resseq_i\tresseq_j\tdist_nm\tdist_A\n")

        for (ai_idx, aj_idx), d in zip(pairs, dists):
            ri = top.atom(ai_idx).residue
            rj = top.atom(aj_idx).residue

            f.write(f"{ri.index}\t{rj.index}\t"
                    f"{ri.chain.chain_id}\t{rj.chain.chain_id}\t"
                    f"{ri.name}\t{rj.name}\t"
                    f"{ri.resSeq}\t{rj.resSeq}\t"
                    f"{d:.6f}\t{d*10:.6f}\n")

    print(f"Wrote {len(pairs)} inter-chain contacts to {args.output}")

    # If cutoff specified, write per-chain contact files
    if args.cutoff is not None:
        # Get output directory from the main output file
        out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
        base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
        
        # Collect contacting residues per chain
        chain_contacts = {}  # chain_id -> set of resSeq values
        
        for (ai_idx, aj_idx), d in zip(pairs, dists):
            ri = top.atom(ai_idx).residue
            rj = top.atom(aj_idx).residue
            
            # Add both residues to their respective chains
            if ri.chain.chain_id not in chain_contacts:
                chain_contacts[ri.chain.chain_id] = set()
            if rj.chain.chain_id not in chain_contacts:
                chain_contacts[rj.chain.chain_id] = set()
                
            chain_contacts[ri.chain.chain_id].add(ri.resSeq)
            chain_contacts[rj.chain.chain_id].add(rj.resSeq)
        
        # Write one file per chain
        for chain_id, residues in chain_contacts.items():
            filename = os.path.join(out_dir, f"{base_name}_{chain_id}_truecontacts.txt")
            sorted_residues = sorted(residues)
            with open(filename, 'w') as f:
                f.write(' '.join(map(str, sorted_residues)) + '\n')
            print(f"Wrote {len(sorted_residues)} residues to {filename}")

if __name__ == '__main__':
    main()


