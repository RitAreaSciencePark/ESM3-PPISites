import pandas as pd
from pathlib import Path
import subprocess

base_path = "data/haddock_units"
csv_path = "inference_results/db5_for_pairing.csv"
df_top = pd.read_csv(csv_path)

def main(base_path, df_top):
    # 1. Isolate W and Z partners
    df_w = df_top[df_top['id'].str.endswith('_W')].copy()
    df_z = df_top[df_top['id'].str.endswith('_Z')].copy()

    # 2. Merge on 'complex' to get all combinations of patches between W and Z
    df_pairs = pd.merge(df_w, df_z, on='complex', suffixes=('_W', '_Z'))

    for _, row in df_pairs.iterrows():
        complex_id = row['complex']
        
        # Setup paths
        id_path = Path(base_path) / complex_id
        tbl_dir = id_path / "tbls"
        config_dir = id_path / "configs"
        tbl_dir.mkdir(exist_ok=True, parents=True)
        config_dir.mkdir(exist_ok=True, parents=True)

        # i and j represents patches id for each chain'
        i = str(row['patch_id_W']).split('_')[-1]
        j = str(row['patch_id_Z']).split('_')[-1]

        # Convert string representations of lists to actual lists of ints
        p_w = [int(x) for x in str(row['patch_W']).split(',')]
        p_z = [int(x) for x in str(row['patch_Z']).split(',')]

        # Logic for alignment
        is_w_longer = len(p_w) >= len(p_z)
        long, short = (p_w, p_z) if is_w_longer else (p_z, p_w)
        c_long, c_short = ("W", "Z") if is_w_longer else ("Z", "W")

        s = (len(long) - len(short)) // 2

        # 4. Orientation loop (k=0 is forward, k=1 is reverse)
        for k, sequence in enumerate([short, short[::-1]]):
            # Naming convention: {patchW}_{patchZ}_{orientation}
            file_name = f"patches_{i}_{j}_{k}"
            tbl_file = tbl_dir / f"{file_name}.tbl"
            config_file = config_dir / f"{file_name}.cfg"

            with open(tbl_file, "w") as f:
                for idx_s, r_short in enumerate(sequence):
                    idx_l_center = s + idx_s
                #    target_window = [idx_l_center - 1, idx_l_center, idx_l_center + 1]
                    target_window = [idx_l_center - 1, idx_l_center, idx_l_center + 1]

                    passives = []
                    for idx_l in target_window:
                        if 0 <= idx_l < len(long):
                            res_val = long[idx_l]
                            passives.append(f"(resid {res_val} and segid {c_long} and name CA)")

                    if passives:
                        active_selection = f"(resid {r_short} and segid {c_short} and name CA)"
                        f.write(f"assign {active_selection}\n")
                        f.write("        (" + " or ".join(passives) + ") 10.0 6.0 4.0\n")

            # Create configuration
            subprocess.run(["python", "scripts_eval/compile_eval_configs.py", str(id_path), str(tbl_file), str(config_file)], check=True)

if __name__ == "__main__":
    main(base_path, df_top)
