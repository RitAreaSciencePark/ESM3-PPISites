import argparse
import csv
import sys
import torch
from tqdm import tqdm
from huggingface_hub import login

from finetuning.src.models import ModelForResidueClassification
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

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

        was_truncated = len(sequence) > max_residues   # ← also cleaner to compare directly

    return pred_str, prob_str, was_truncated
def main():
    parser = argparse.ArgumentParser(description="Run ESM3 Residue Classification Inference")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    
    args = parser.parse_args()

    # 1. Authenticate with Hugging Face
    print("Authenticating with Hugging Face...")
    login(token=args.token)

    # 2. Load the model and tokenizer
    print("Loading model and tokenizer...", flush=True)
    repo_id = "area-science-park/ESM3-PPISites"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Pass the token directly to from_pretrained as an extra precaution
    model = ModelForResidueClassification.from_pretrained(repo_id, token=args.token).to(device)
    model.eval()  # Set model to evaluation mode
    
    tokenizer = EsmSequenceTokenizer()

    # 3. Read Input CSV
    print(f"Reading input from {args.input}...", flush=True)
    with open(args.input, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames or 'id' not in fieldnames or 'sequence' not in fieldnames:
            print("Error: Input CSV must contain at least 'id' and 'sequence' columns.")
            sys.exit(1)

        rows = list(reader)

    # Prepare output fieldnames, preserving all original columns
    out_fieldnames = fieldnames + ['probabilities', 'prediction']

    # 4. Process Sequences and Write Output CSV
    print(f"Processing {len(rows)} sequences...", flush=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fieldnames)
        writer.writeheader()

        for i, row in enumerate(tqdm(rows, desc="Inference Progress")):
            seq = row['sequence'].strip()
            
            # Skip empty sequences
            if not seq:
                row['probabilities'] = ""
                row['prediction'] = ""
                writer.writerow(row)
                continue

            # Run inference
            pred_str, prob_str, truncated = run_inference(
                model=model, 
                tokenizer=tokenizer, 
                sequence=seq, 
                device=device
            )

            if truncated:
                print(f"\n⚠️  Sequence {row['id']} (len={len(seq)}) was truncated.")

            # Update row with new data
            row['probabilities'] = prob_str
            row['prediction'] = pred_str
            
            writer.writerow(row)

    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()