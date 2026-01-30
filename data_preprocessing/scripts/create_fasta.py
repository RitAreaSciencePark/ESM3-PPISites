import sys
import os
import pandas as pd
import argparse

def load_data(file_path):
    """
    Loads a CSV file and normalizes columns to 'id' and 'sequence'.
    """
    try:
        df = pd.read_csv(file_path)
        
        mapping = {
            'uniprot_id': 'id', 
            'pdb_id': 'id', 
            'sequence': 'sequence'
        }
        df = df.rename(columns=mapping)

        if 'id' not in df.columns:
            df = pd.read_csv(file_path, header=None)
            if 'id' not in df.columns:
                df = pd.read_csv(file_path, header=None)
                df.columns = ['id', 'sequence', 'labels']
            
        return df

    except Exception as e:
        raise RuntimeError(f"Error processing CSV {file_path}: {e}")


def save_fasta(df, output_path):
    """
    Writes a pandas DataFrame to a FASTA file.
    Expects 'id' and 'sequence' columns.
    """
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            # Clean the sequence string
            clean_seq = str(row['sequence']).replace(',', '').strip()
            # Write FASTA format
            f.write(f">{row['id']}\n{clean_seq}\n")

def convert_file(input_path, output_dir="fastas"):
    """
    1. Reads input CSV
    2. Creates output directory
    3. Writes FASTA file
    
    Returns the path to the created file.
    """
    #Prepare output path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(input_path).rsplit('.', 1)[0]
    output_file = os.path.join(output_dir, f"{base_name}.fasta")

    #load and save data
    df = load_data(input_path)
    save_fasta(df, output_file)
    
    return output_file

def main():
    """
    Main function to handle CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Convert CSV protein data to FASTA format.")
    parser.add_argument("input_file", help="Path to the input .csv file")
    parser.add_argument("--output-dir", "-o", default="fastas", help="Directory to save output files (default: fastas)")
    
    args = parser.parse_args()

    try:
        result_path = convert_file(args.input_file, args.output_dir)
        print(f"Done! File saved to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()