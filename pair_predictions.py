import argparse
import csv
import sys
import os
import torch
from tqdm import tqdm
from huggingface_hub import login

# Prediction specific imports
from finetuning.src.models import ModelForResidueClassification
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# ESM3 structure generation imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

def run_inference(model, tokenizer, sequence, device, max_length=1024):
    max_residues = max_length - 2  # exclude BOS and EOS slots

    # Pad to max_length to match training/notebook behaviour
    inputs = tokenizer(
        sequence,
        padding="max_length",  
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Compute effective length from original sequence, not from padded tensor
    effective_len = min(len(sequence), max_residues)   

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probs = torch.sigmoid(logits)

        probs_np = probs.squeeze(0).cpu().numpy()      
        preds_np = (probs_np > 0.7).astype(int)

        # Skip BOS (index 0), extract exactly effective_len residues
        start_idx = 1
        end_idx = start_idx + effective_len

        valid_probs = probs_np[start_idx:end_idx]
        valid_preds = preds_np[start_idx:end_idx]

        prob_str = " ".join([f"{p:.2f}" for p in valid_probs])
        pred_str = "".join(map(str, valid_preds))

        was_truncated = len(sequence) > max_residues   

    return pred_str, prob_str, was_truncated

def get_hotspot_indices(pred_str):
    """
    Extracts the 0-based indices where the prediction string has '1's.
    Returns a comma-separated string of indices, or empty string if none.
    """
    return ",".join([str(i) for i, char in enumerate(pred_str) if char == '1'])

def main():
    parser = argparse.ArgumentParser(description="Run ESM3 Residue Classification Inference & PDB Generation")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    
    args = parser.parse_args()

    # 1. Authenticate with Hugging Face
    print("Authenticating with Hugging Face...")
    login(token=args.token)

    print("Loading models and tokenizer...", flush=True)
    repo_id = "area-science-park/ESM3-PPISites"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load Residue Classification Model
    print("Loading Residue Classification Model...")
    cls_model = ModelForResidueClassification.from_pretrained(repo_id, token=args.token).to(device)
    cls_model.eval()  
    tokenizer = EsmSequenceTokenizer()

    # Load ESM3 Open Model for Structure Generation
    print("Loading ESM3 Structure Generation Model (esm3-open)...")
    try:
        struct_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        struct_model.eval()
    except Exception as e:
        print(f"Error loading structure model: {e}")
        sys.exit(1)

    print(f"Reading input from {args.input}...", flush=True)
    with open(args.input, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames or 'id' not in fieldnames or 'sequence' not in fieldnames:
            print("Error: Input CSV must contain at least 'id' and 'sequence' columns.")
            sys.exit(1)

        rows = list(reader)

    # Prepare output fieldnames, adding hotspot_pred
    out_fieldnames = fieldnames + ['probabilities', 'prediction', 'hotspot_pred']

    print(f"Processing {len(rows)} sequences...", flush=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, row in enumerate(tqdm(rows, desc="Inference Progress")):
            uid = row['id'].strip()
            raw_seq = row['sequence'].strip()
            
            # Clean sequence
            seq = raw_seq.replace(",", "").strip()

            # Skip empty sequences
            if not seq:
                row['probabilities'] = ""
                row['prediction'] = ""
                row['hotspot_pred'] = ""
                writer.writerow(row)
                continue

            # Run classification inference
            pred_str, prob_str, truncated = run_inference(
                model=cls_model, 
                tokenizer=tokenizer, 
                sequence=seq, 
                device=device
            )

            if truncated:
                print(f"\n⚠️  Sequence {uid} (len={len(seq)}) was truncated during classification.")

            # Calculate hotspots
            hotspot_indices = get_hotspot_indices(pred_str)

            # Update row with new data
            row['probabilities'] = prob_str
            row['prediction'] = pred_str
            row['hotspot_pred'] = hotspot_indices
            
            writer.writerow(row)

            output_filename = f"{uid}.pdb"
            output_path = output_filename
            
            # Skip if the file already exists
            if os.path.exists(output_path):
                continue
        
            try:
                # Wrap sequence in ESMProtein format
                protein = ESMProtein(sequence=seq)
                
                # Generate structure tracking 8 steps as requested
                with torch.no_grad():
                    protein = struct_model.generate(protein, GenerationConfig(track="structure", num_steps=8))
                
                # Save out to PDB
                protein.to_pdb(output_path)
                
            except Exception as e:
                print(f"\nFailed to generate structure for {uid}: {e}")

    print(f"Inference complete. Results saved to {args.output}")
    print("PDB files saved to the current working directory.")

if __name__ == "__main__":
    main()