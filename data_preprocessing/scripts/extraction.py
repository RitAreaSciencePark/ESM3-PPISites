"""
extractions.py contains the functions necessary to extract PDBS from sequences
"""
import os
import csv
import argparse  # Added for command line argument parsing
from tqdm import tqdm
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

def generate_all_structures_with_esm3_csv(csv_path, target_folder):
    """
    Iterates through a CSV file (UniprotID, Sequence) and generates 3D structures 
    using the ESM3 model. 
    
    Expects the sequence in the second column to be comma-separated 
    (e.g., "M,A,K,E"), which will be cleaned to "MAKE".
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(target_folder):
        print(f"Creating output directory: {target_folder}")
        os.makedirs(target_folder)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading ESM3 model (esm3-open)...")
    try:
        model = ESM3.from_pretrained("esm3-open").to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Reading data from {csv_path}...")
    
    # Read CSV data first to get total count for tqdm
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Optional: Uncomment the next line if your CSV has a header row
            # next(reader, None) 
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Starting structure generation for {len(rows)} sequences...")

    # Iterate through CSV rows
    for row in tqdm(rows):
        if len(row) < 2:
            continue
            
        uid = row[0].strip()
        raw_sequence = row[1]
        
        # CLEANING STEP: Remove commas and whitespace from the sequence
        sequence = raw_sequence.replace(",", "").strip()

        # Skip entries with empty sequences
        if not sequence:
            continue

        output_filename = f"{uid}_esm3.pdb" 
        output_path = os.path.join(target_folder, output_filename)
        
        # Skip if the file already exists
        if os.path.exists(output_path):
            continue
    
        try:

            protein = ESMProtein(sequence=sequence)
            protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
            protein.to_pdb(output_path)
            
        except Exception as e:
            print(f"Failed to generate structure for {uid}: {e}")

    print(f"ESM3 processing complete. Files saved in '{target_folder}'")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Generate PDB structures from sequences in a CSV file using ESM3."
    )
    
    # Add arguments
    parser.add_argument(
        "input_csv", 
        help="Path to the input CSV file containing UniprotID and Sequence."
    )
    parser.add_argument(
        "output_folder", 
        help="Path to the output folder where PDB files will be created."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the function with command line arguments
    generate_all_structures_with_esm3_csv(args.input_csv, args.output_folder)