import pandas as pd
from pathlib import Path

from src.config_defaults import * ### CONSTANTS

def parse_capri(fp):
    capri_fp = Path(fp)
    capri_parts = capri_fp.parts
    runs_index = capri_parts.index('runs')
    jobname = capri_parts[runs_index-1]
    runname = capri_parts[runs_index+1]
    is_flex = runname.split('_')[-1]
    approach = runname.split('_')[0]
    return [jobname, runname, is_flex, approach]

input_basedir = Path(HADDOCK_BASEDIR)
if not input_basedir.exists():
    raise ValueError("Missing haddock runs directory... ?!")
subdirs = [p for p in input_basedir.iterdir() if p.is_dir()]

### by job directory, and by run type, 
capri_collection_list = []
for direc in subdirs:
    capri_fps = [p for p in direc.rglob("capri_ss.tsv") if "analysis" in p.parts]
    capri_collection_list.extend(capri_fps)

print(f"FOUND: {len(capri_collection_list)} files to read...")

runs_list = []
for fp in capri_collection_list:
    run_df = pd.read_csv(fp, sep="\t")
    run_df['raw_fp'] = str(input_basedir / fp.relative_to(input_basedir))
    run_data = parse_capri(fp)
    run_data[3] = 'patch' if run_data[3] == 'p' else run_data[3]
    run_df['complex'] = run_data[0]
    run_df['run_type'] = run_data[1]
    run_df['is_flex'] = run_data[2]
    run_df['docking_approach'] = run_data[3]
    if run_df.empty:
        print("[WARNING] capri_ss.tsv file... not found or empty")
    if not run_df.empty:
        runs_list.append(run_df)

all_runs_df = pd.concat(runs_list)
print(f"Haddock poses gathered: {len(all_runs_df)}")

# ADD SCORE RANKING
group_cols = ['complex', 'is_flex', 'docking_approach']

all_runs_df = all_runs_df.sort_values(group_cols + ['score'], ascending=[True, True, True, True])
all_runs_df['score_rank'] = (
    all_runs_df.groupby(group_cols)['score']
    .rank(method='min', ascending=True)
    .astype(int)
)

output_path = Path(CAPRIEVALS_FP)
output_path.parent.mkdir(parents=True, exist_ok=True)

all_runs_df.to_csv(output_path, sep="\t", index=False)
