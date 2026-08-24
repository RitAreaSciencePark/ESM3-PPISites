#!/usr/bin/env python3
### <deps>
import pandas as pd
from pathlib import Path
import subprocess
import getpass

from config_defaults import *
### <!deps>

###########
def _fasta_seq(fp):
  if not fp.exists():
    raise ValueError(f"[FAIL] {fp} does not exist")
  with open(str(fp)) as fasta:
    content = fasta.read()
  lines = content.splitlines()
  output = "".join(lines[1:])
  return output
###########

PDB_seqs = 'PDB_seqs'

input_basedir = Path(HADDOCK_BASEDIR)
target_dirs = [p for p in input_basedir.iterdir() if p.is_dir()]

pdb_fasta_files = []
for target in target_dirs:
    location = target / PDB_seqs
    if location.is_dir():
        files = [
            f for f in location.iterdir() 
            if f.is_file() and f.suffix == '.fasta' and (f.name.startswith('c1_') or f.name.startswith('c2_'))
        ]
        pdb_fasta_files.extend(files)

print(pdb_fasta_files)

data_inference = []
for fp in pdb_fasta_files:
    file_id = fp.stem 
    csv_id = file_id.replace('c1_', '').replace('c2_', '').replace('_pdbseqs', '')
    sequence = _fasta_seq(fp)
    data_inference.append({'id': csv_id, 'sequence': sequence})

df_inference = pd.DataFrame(data_inference)

Path(INFERENCE_INPUT).parent.mkdir(parents=True, exist_ok=True)
df_inference.to_csv(INFERENCE_INPUT, index=False)

api_token = getpass.getpass("Enter your token for inference: ")

# Execute the subprocess command
cmd = [
    'python3',
    RUN_INFERENCE_SCRIPT,
    '--input', INFERENCE_INPUT,
    '--output', INFERENCE_OUTPUT,
    '--token', api_token
]
result = subprocess.run(cmd, text=True)
if result.returncode != 0:
    raise RuntimeError(f"[FAIL] Inference script failed with exit code {result.returncode}")
else:
    print(f"[SUCCESS] Inference completed. Results saved to {INFERENCE_OUTPUT}")

