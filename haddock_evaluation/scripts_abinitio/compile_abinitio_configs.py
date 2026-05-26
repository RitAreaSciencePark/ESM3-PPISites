#!/usr/bin/env python3
"""Generate ab-initio HADDOCK3 configurations."""

from pathlib import Path


def validate_pdb_files(unit_dir):
    """Validate required PDB files exist in unit directory."""
    complex_files = list(unit_dir.glob("*_WZ.pdb"))
    mol1_files = list(unit_dir.glob("*_Z.pdb"))
    mol2_files = list(unit_dir.glob("*_W.pdb"))

    if len(complex_files) != 1 or len(mol1_files) != 1 or len(mol2_files) != 1:
        return None

    return {
        "complex": complex_files[0].name,
        "mol1": mol1_files[0].name,
        "mol2": mol2_files[0].name,
    }


def generate_config(mol1, mol2, complex_pdb):
    """Generate ab-initio config content."""
    return f"""run_dir = "haddock_unrestr_10k"
molecules = ["{mol1}","{mol2}"]
ncores = 64

[topoaa]

[rigidbody]
ranair = true
sampling = 10000

[seletop]
select = 200

[flexref]
contactairs = true

[emref]

[clustfcc]

[seletopclusts]

[caprieval]
reference_fname = "{complex_pdb}"
"""


def main():
    target_dir = Path("data/haddock_units")

    if not target_dir.is_dir():
        print(f"Error: Target directory '{target_dir}' does not exist.")
        return

    for unit_dir in sorted(target_dir.iterdir()):
        if not unit_dir.is_dir():
            continue

        unit_name = unit_dir.name
        print(f"Processing unit: {unit_name}")

        pdb_files = validate_pdb_files(unit_dir)
        if pdb_files is None:
            print(f"  ERROR: Missing or ambiguous PDB files in {unit_name}")
            continue

        print(
            f"  Found: {pdb_files['complex']}, {pdb_files['mol1']}, {pdb_files['mol2']}"
        )

        configs_dir = unit_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        config_content = generate_config(
            pdb_files["mol1"], pdb_files["mol2"], pdb_files["complex"]
        )
        config_path = configs_dir / "config_abinitio.cfg"
        config_path.write_text(config_content)
        print(f"  Created configs/config_abinitio.cfg")

    print("\nDone!")


if __name__ == "__main__":
    main()