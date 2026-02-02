import pandas as pd
import os
import shutil
import argparse
import sys

def extract_non_similar_proteins(input_folder, m8_file_path, output_folder):
    """
    Identifies proteins in the input_folder that do NOT meet the similarity 
    criteria found in the m8_file, and copies them to the output_folder.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")
        except OSError as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)

    # 1. Load and Filter the .m8 file to find the "Similar" ones
    # Using the column names and logic from your provided snippet
    column_names = [
        "query", "target", "alntmscore", 
        "qtmscore", "ttmscore", "evalue"
    ]

    print(f"Reading {m8_file_path}...")
    try:
        # Load the dataframe
        df = pd.read_csv(m8_file_path, sep='\t', names=column_names)
        
        # Apply the logic from your provided function to find "Matches"
        # We look for rows that ARE similar, so we can exclude them later.
        
        # Logic 1: Ignore self-matches (we don't exclude a protein just because it matches itself)
        # We only care if it matches *something else*
        df_filtered = df[df['query'] != df['target']]
        
        # Logic 2: E-value < 1e-3
        df_filtered = df_filtered[df_filtered['evalue'] < 1e-3]
        
        # Logic 3: qtmscore > 0.5
        df_filtered = df_filtered[df_filtered['qtmscore'] > 0.5]

        # Create a set of proteins that HAVE a match
        similar_proteins = set(df_filtered['query'].unique())
        
        print(f"Found {len(similar_proteins)} proteins that matched similarity criteria (to be excluded).")

    except Exception as e:
        print(f"Error processing m8 file: {e}")
        sys.exit(1)

    # 2. Iterate through Input Folder and Copy "Non-Similar"
    copied_count = 0
    skipped_count = 0
    
    print(f"Scanning {input_folder} for PDBs...")

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):
            # Extract the query name (remove .pdb extension)
            query_name = os.path.splitext(filename)[0]
            
            source_path = os.path.join(input_folder, filename)
            dest_path = os.path.join(output_folder, filename)

            # 3. The Logic Reversal
            # If the protein is NOT in the similar_proteins set, we copy it.
            if query_name not in similar_proteins:
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    # Optional: Print every file copied (can be verbose)
                    # print(f"Copied non-similar: {filename}")
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
            else:
                skipped_count += 1

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total PDBs scanned: {copied_count + skipped_count}")
    print(f"Similar proteins excluded: {skipped_count}")
    print(f"Non-similar proteins copied: {copied_count}")
    print(f"Output folder: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy proteins that do NOT meet similarity criteria in a Foldseek m8 file."
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the folder containing original PDB files."
    )
    parser.add_argument(
        "--m8", "-m", 
        required=True, 
        help="Path to the Foldseek .m8 output file."
    )
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="Target folder where non-similar PDBs will be copied."
    )

    args = parser.parse_args()

    extract_non_similar_proteins(args.input, args.m8, args.output)