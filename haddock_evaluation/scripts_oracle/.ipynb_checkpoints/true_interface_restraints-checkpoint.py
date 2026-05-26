import csv
import argparse
import os
import glob
import sys

def create_haddock_tbl_from_tsv(input_filename, output_filename, dist=10.0, d_minus=6.0, d_plus=2.0):
    """
    Converts a TSV file with specific headers into a HADDOCK .tbl file.
    Target columns: chain_i, chain_j, mapped_to_chain_i, mapped_to_chain_j
    """
    try:
        with open(input_filename, 'r', newline='') as infile, open(output_filename, 'w') as outfile:
            reader = csv.DictReader(infile, delimiter='\t')
            for row_num, row in enumerate(reader, 1):
                try:
                    chainA    = row['chain_i']
                    chainB    = row['chain_j']
                    pos_resA  = row['mapped_to_chain_i']
                    pos_resB  = row['mapped_to_chain_j']
                    tbl_line = (f"assign (resid {pos_resA:>4} and segid {chainA:>2} and name CA) "
                                f"(resid {pos_resB:>4} and segid {chainB:>2} and name CA) "
                                f"{dist:.1f} {d_minus:.1f} {d_plus:.1f}\n")
                    outfile.write(tbl_line)
                except KeyError as e:
                    print(f"Error: Missing column {e} at row {row_num}. Check your TSV headers.")
                    continue
        print(f"Success! HADDOCK restraints saved to '{output_filename}'.")
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def find_target_pairs_tsv(directory):
    """Find a file ending with '_targetPairs.tsv' in the contacts/ subdirectory."""
    contacts_dir = os.path.join(directory, 'contacts')
    pattern = os.path.join(contacts_dir, '*_targetPairs.tsv')
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: Multiple '_targetPairs.tsv' files found; using '{matches[0]}'.")
    return matches[0]

def process_directory(target_dir, output_file=None, dist=10.0, d_minus=6.0, d_plus=2.0):
    """Process a single directory."""
    target_dir = os.path.abspath(target_dir)
    
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory.")
        return False
    
    input_file = find_target_pairs_tsv(target_dir)
    if input_file is None:
        print(f"Warning: No '*_targetPairs.tsv' file found in '{target_dir}'.")
        return False
    
    print(f"\nProcessing: {target_dir}")
    print(f"Input file : {input_file}")
    
    # Create tbls/ directory if it doesn't exist
    tbls_dir = os.path.join(target_dir, "tbls")
    if not os.path.exists(tbls_dir):
        os.makedirs(tbls_dir)
        print(f"Created directory: {tbls_dir}")
    
    # Resolve output file
    if output_file is None:
        output_file = os.path.join(tbls_dir, "ti.tbl")
    else:
        # If custom output provided, still place it in tbls/ subdirectory
        output_file = os.path.join(tbls_dir, os.path.basename(output_file))
    
    print(f"Output file: {output_file}")
    print(f"Parameters : dist={dist}, d_minus={d_minus}, d_plus={d_plus}")
    
    create_haddock_tbl_from_tsv(input_file, output_file, dist, d_minus, d_plus)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert *_targetPairs.tsv files into HADDOCK .tbl restraints files. "
                    "Iterates over all subdirectories in the specified root directory."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing subdirectories with *_targetPairs.tsv files "
             "(default searches 'data/haddock_units').",
        nargs='?',
        default='data/haddock_units'
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .tbl file path template. If not specified, defaults to '<subdir>/ti.tbl' for each subdirectory."
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=10.0,
        help="Distance restraint value in Å (default: 10.0)."
    )
    parser.add_argument(
        "--d_minus",
        type=float,
        default=6.0,
        help="Lower bound tolerance in Å (default: 6.0)."
    )
    parser.add_argument(
        "--d_plus",
        type=float,
        default=2.0,
        help="Upper bound tolerance in Å (default: 2.0)."
    )
    args = parser.parse_args()
    
    # Resolve root directory
    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        sys.exit(1)
    
    print(f"Root directory: {root_dir}")
    print(f"Searching for subdirectories...\n")
    
    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))])
    
    if not subdirs:
        print(f"Error: No subdirectories found in '{root_dir}'.")
        sys.exit(1)
    
    print(f"Found {len(subdirs)} subdirectories.\n")
    print("=" * 80)
    
    # Process each subdirectory
    success_count = 0
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if process_directory(subdir_path, args.output, args.dist, args.d_minus, args.d_plus):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"\nSummary: Successfully processed {success_count}/{len(subdirs)} directories.")

if __name__ == "__main__":
    main()


