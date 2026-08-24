#!/usr/bin/env python3
"""Generate ab-initio HADDOCK3 configurations."""

from pathlib import Path
from config_defaults import *

# CONFIG
RUN_NAME = "abinitio_10k"
SAMPLING = 10000
SELECT = 200
INI_SEED = 42

target_dir = Path(HADDOCK_BASEDIR)

if not target_dir.is_dir():
    print(f"Error: Target directory '{target_dir}' does not exist.")
    raise SystemExit(1)

for unit_dir in sorted(target_dir.iterdir()):
    if not unit_dir.is_dir():
        continue

    unit_name = unit_dir.name
    print(f"Processing unit: {unit_name}")

    complex_files = list(unit_dir.glob("*_WZ.pdb"))
    mol1_files = list(unit_dir.glob("*_Z.pdb"))
    mol2_files = list(unit_dir.glob("*_W.pdb"))

    if len(complex_files) != 1 or len(mol1_files) != 1 or len(mol2_files) != 1:
        print(f"  ERROR: Missing or ambiguous PDB files in {unit_name}")
        continue

    pdb_files = {
        "complex": complex_files[0].name,
        "mol1": mol1_files[0].name,
        "mol2": mol2_files[0].name,
    }

    print(f"  Found: {pdb_files['complex']}, {pdb_files['mol1']}, {pdb_files['mol2']}")

    configs_dir = unit_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = unit_dir / "runs" / RUN_NAME
    mol1_fp = unit_dir / pdb_files["mol1"]
    mol2_fp = unit_dir / pdb_files["mol2"]
    complex_fp = unit_dir / pdb_files["complex"]

    config_content = f"""run_dir = "{run_dir}"
molecules = ["{mol1_fp}","{mol2_fp}"]
ncores = {DEF_NCORES}

[topoaa]
iniseed = {INI_SEED}

[rigidbody]
ranair = true
sampling = {SAMPLING}
iniseed = {INI_SEED}

[seletop]
select = {SELECT}

[flexref]
contactairs = true
iniseed = {INI_SEED}

[emref]
iniseed = {INI_SEED}

[clustfcc]

[seletopclusts]

[caprieval]
reference_fname = "{complex_fp}"
fnat_cutoff = {FNAT_CUTOFF}
"""

    config_path = configs_dir / "config_abinitio.cfg"
    config_path.write_text(config_content)
    print(f"  Created configs/config_abinitio.cfg")

print("\nDone!")

