import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import argparse

def main():
    parser = argparse.ArgumentParser(description='Align residues from chain and complex TSV files using PairwiseAligner')
    parser.add_argument('--complex', required=True, help='Path to the complex TSV file (multi-chain)')
    parser.add_argument('--chain', required=True, help='Path to the single-chain TSV file')
    parser.add_argument('--output', default='aligned.tsv', help='Path to the output TSV file')
    args = parser.parse_args()

    # Read the TSV files
    df_complex = pd.read_csv(args.complex, sep='\t')
    df_chain = pd.read_csv(args.chain, sep='\t')

    # Determine the chain ID from the --chain file (assume single chain)
    if df_chain['chain'].nunique() != 1:
        raise ValueError("The --chain file should contain exactly one unique chain ID")
    chain_id = df_chain['chain'].iloc[0]

    print(f"Using chain ID: {chain_id}")

    # Extract the matching chain from both files
    df_chain_sel = df_chain[df_chain['chain'] == chain_id].reset_index(drop=True)
    df_complex_sel = df_complex[df_complex['chain'] == chain_id].reset_index(drop=True)

    if df_chain_sel.empty:
        raise ValueError(f"No residues found for chain {chain_id} in --chain file")
    if df_complex_sel.empty:
        raise ValueError(f"No residues found for chain {chain_id} in --complex file")

    # Get sequences (one-letter codes)
    seq_chain = ''.join(df_chain_sel['aa'])
    seq_complex = ''.join(df_complex_sel['aa'])

    if not seq_chain or not seq_complex:
        raise ValueError("One of the sequences is empty after filtering — check input files")

    print(f"Chain sequence length: {len(seq_chain)}")
    print(f"Complex sequence length: {len(seq_complex)}")

    # Prepare aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    # Perform alignment
    alignments = aligner.align(seq_chain, seq_complex)

    # Take the best-scoring alignment
    alignment = alignments[0]

    # Convert to strings
    aligned_chain = str(alignment[0])
    aligned_complex = str(alignment[1])

    # Walk through alignment
    results = []
    i_chain = 0
    i_complex = 0

    for a1, a2 in zip(aligned_chain, aligned_complex):
        row = {
            'aa_chain': a1,
            'aa_complex': a2,
            'identical_aa': False
        }

        # Metadata from chain file
        if a1 != '-':
            if i_chain < len(df_chain_sel):
                row['pdb_chain'] = df_chain_sel['pdb'].iloc[i_chain]
                row['chain_chain'] = df_chain_sel['chain'].iloc[i_chain]
                row['aminoacid_chain'] = df_chain_sel['aminoacid'].iloc[i_chain]
                row['position_chain'] = df_chain_sel['position'].iloc[i_chain]
                row['is_modified_chain'] = df_chain_sel['is_modified'].iloc[i_chain]
            i_chain += 1
        else:
            row['pdb_chain'] = ''
            row['chain_chain'] = ''
            row['aminoacid_chain'] = ''
            row['position_chain'] = None
            row['is_modified_chain'] = ''

        # Metadata from complex file
        if a2 != '-':
            if i_complex < len(df_complex_sel):
                row['pdb_complex'] = df_complex_sel['pdb'].iloc[i_complex]
                row['chain_complex'] = df_complex_sel['chain'].iloc[i_complex]
                row['aminoacid_complex'] = df_complex_sel['aminoacid'].iloc[i_complex]
                row['position_complex'] = df_complex_sel['position'].iloc[i_complex]
                row['is_modified_complex'] = df_complex_sel['is_modified'].iloc[i_complex]
            i_complex += 1
        else:
            row['pdb_complex'] = ''
            row['chain_complex'] = ''
            row['aminoacid_complex'] = ''
            row['position_complex'] = None
            row['is_modified_complex'] = ''

        # identical_aa flag
        if a1 != '-' and a2 != '-':
            row['identical_aa'] = (a1 == a2)

        results.append(row)

    # Create output DataFrame
    df_out = pd.DataFrame(results)

    # Save with float_format to remove .0 from whole numbers, preserve strings/insertion codes
    df_out.to_csv(
        args.output,
        sep='\t',
        index=False,
        float_format=lambda x: f"{int(x):d}" if isinstance(x, (int, float)) and x == int(x) else str(x)
    )
    print(f"Alignment score: {alignment.score:.1f}")
    print(f"Output saved to {args.output}  ({len(df_out)} rows)")

if __name__ == '__main__':
    main()


