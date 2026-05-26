import glob
import os
import pandas as pd

base_dir = "data/haddock_units"

# Loop over each subdirectory inside base_dir
for sub in os.listdir(base_dir):
    sub_path = os.path.join(base_dir, sub)

    if not os.path.isdir(sub_path):
        continue
    
    files = glob.glob(os.path.join(sub_path, "runs", "*", "*_caprieval", "capri_ss.tsv"))
    all_best_models = []

    for file in files:
        try:
            df = pd.read_csv(file, sep="\t")

            # Sort by score
            df_sorted = df.sort_values("score", ascending=True)

            # Take best row (not used now since we select only one pose)
            best_row = df_sorted.iloc[[0]].copy()

            provenance = os.path.basename(os.path.dirname(os.path.dirname(file)))
            best_row["provenance"] = provenance

            all_best_models.append(best_row)

        except Exception as e:
            print(f"Skipping {file}: {e}")

    if all_best_models:
        result = pd.concat(all_best_models, ignore_index=True)
        final_table = result.sort_values(by='score', ascending=True)

        output_path = os.path.join(sub_path, f"{sub}_result.csv")
        final_table.to_csv(output_path, index=False)

        print(f"[{sub}] Saved {len(final_table)} results to {output_path}")
        print(final_table[['score', 'irmsd', 'air', 'vdw']].head(8))
    else:
        print(f"[{sub}] No valid capri_ss.tsv files found.")
