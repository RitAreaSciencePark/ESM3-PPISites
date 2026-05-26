#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 4:
        print("Usage: python compile_eval_configs.py <id_path> <tbl_file> <config_file>")
        sys.exit(1)

    id_path = Path(sys.argv[1])
    tbl_file = Path(sys.argv[2])
    config_file = Path(sys.argv[3])

    run_dir = id_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Equivalent to $(basename $tbl_file | sed 's/.tbl//g')
    patch = tbl_file.stem

    # Find the _W and _Z PDB files in the id_path directory
    prot_w_list = list(id_path.glob("*_W.pdb"))
    prot_z_list = list(id_path.glob("*_Z.pdb"))

    if not prot_w_list or not prot_z_list:
        print(f"Error: Missing '_W.pdb' or '_Z.pdb' in {id_path}")
        sys.exit(1)

    prot_w = prot_w_list[0].name
    prot_z = prot_z_list[0].name

    # Construct the config contents
    config_content = f"""run_dir = "{run_dir / patch}"
molecules = ["{id_path / prot_w}","{id_path / prot_z}"]
ncores = 64

[topoaa]
iniseed = 42

[rigidbody]
ambig_fname = "{tbl_file}"
sampling = 100
iniseed = 42
npart = 5

[seletop]
select = 1

[emref]
ambig_fname = "{tbl_file}"
iniseed = 42

[caprieval]
reference_fname = "{id_path / f'{id_path.name}_WZ.pdb'}"
"""

    config_file.write_text(config_content)
    print(f"Created configuration file at: {config_file}")

if __name__ == "__main__":
    main()