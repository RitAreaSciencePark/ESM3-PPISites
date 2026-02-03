import csv
import os
import argparse
import sys

def filter_csv_by_pdb(input_csv, pdb_dir, output_csv):
    """
    Filters rows in a CSV based on whether a corresponding .pdb file exists
    in the specified directory.
    """
    
    # 1. Validation
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found: {input_csv}")
        sys.exit(1)
        
    if not os.path.isdir(pdb_dir):
        print(f"Error: PDB directory not found: {pdb_dir}")
        sys.exit(1)

    found_count = 0
    total_count = 0
    
    try:
        with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            # Verify columns exist
            required_columns = ['uniprot_id'] # sequence and interface are optional for the check, but must be in file
            if not reader.fieldnames or 'uniprot_id' not in reader.fieldnames:
                print(f"Error: Input CSV must contain a 'uniprot_id' column. Found: {reader.fieldnames}")
                sys.exit(1)
            
            # Prepare output
            with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                
                print(f"Processing...")
                
                for row in reader:
                    total_count += 1
                    uniprot_id = row['uniprot_id'].strip()
                    
                    # Construct the expected PDB filename
                    pdb_filename = f"{uniprot_id}_esm3.pdb"
                    pdb_path = os.path.join(pdb_dir, pdb_filename)
                    
                    # Check if file exists
                    if os.path.exists(pdb_path):
                        writer.writerow(row)
                        found_count += 1
                    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print("-" * 30)
    print(f"Filtering complete.")
    print(f"Total rows processed: {total_count}")
    print(f"Rows kept (PDB found): {found_count}")
    print(f"Output saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a CSV file to keep only rows with corresponding PDB files in a directory."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input CSV file containing 'uniprot_id', 'sequence', 'interface'"
    )
    parser.add_argument(
        "-d", "--pdb_dir", 
        required=True, 
        help="Path to the directory containing .pdb files (named uniprot_id.pdb)"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path where the final filtered CSV will be saved"
    )

    args = parser.parse_args()
    
    filter_csv_by_pdb(args.input, args.pdb_dir, args.output)