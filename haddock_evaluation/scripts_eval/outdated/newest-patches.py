import pandas as pd
import numpy as np
import sys

ban = ['2GAF']
### remove from df rows where complex are in ban
df = pd.read_csv("inference_results/db5_results.csv")
df = df[~df['complex'].isin(ban)]
df = pd.read_csv("inference_results/db5_results.csv")
df = df[~df['complex'].isin(ban)]

print("Columns in CSV:", df.columns.tolist())  # <-- Add this to debug
print(df.head())     


def extract_patches(residues, probs):
    if not residues:
        return

    curr_res = [residues[0]]
    curr_probs = [probs[0]]

    for r, p in zip(residues[1:], probs[1:]):
        if r - curr_res[-1] <= 4:
            curr_res.append(r)
            curr_probs.append(p)
        else:
            yield curr_res, curr_probs
            curr_res, curr_probs = [r], [p]

    yield curr_res, curr_probs


def make_patches(df):
    
    patch_data = []
    for _, row in df.iterrows():
        residues = [int(float(x)) for x in str(row['hotspot_pred']).split(',') if x.strip()]
        probs = [float(x) for x in str(row['prob']).split(',') if x.strip()]
        seq_len = row['length']
        min_patch_length = 5 if seq_len < 400 else 4

        if len(residues) > 15:
            groups = [
                (res_group, prob_group)
                for res_group, prob_group in extract_patches(residues, probs)
                if len(res_group) > min_patch_length
            ]
        else:
            groups = [(residues, probs)]

        for res_group, prob_group in groups:
            patch_data.append({
                'complex': row['complex'],
                'id': row['id'],
                'patch': ",".join(map(str, res_group)),
                'avg_probability': np.mean(prob_group)
            })
    df_all_patches = pd.DataFrame(patch_data)
    return df_all_patches

df_all = make_patches(df)

if not df_all.empty:
    df_all.to_csv("inference_results/db5_patches.csv", index=False)
    df_top = df_all.sort_values(by=['id', 'avg_probability'], ascending=[True, False])
    group_sizes = df_top.groupby('id')['id'].transform('size')
    ranks = df_top.groupby('id').cumcount()  
    conditional_mask = (ranks < 2) | ((group_sizes == 3) & (ranks == 2))
    df_filtered = df_top[conditional_mask].reset_index(drop=True)
    df_filtered['patch_id'] = "patch_" + df_filtered.groupby('id').cumcount().astype(str)
    df_filtered = df_filtered[['complex', 'id', 'patch_id', 'patch', 'avg_probability']].sort_values(by=['complex'])
    df_filtered.to_csv("inference_results/db5_for_pairing.csv", index=False)

#    df_top2 = df_top.groupby('id').head(2).reset_index(drop=True)
#    df_top2['patch_id'] = "patch_" + df_top2.groupby('id').cumcount().astype(str)
#    df_top2 = df_top2[['complex', 'id', 'patch_id', 'patch', 'avg_probability']].sort_values(by=['complex'])
#    df_top2.to_csv("inference_results/db5_for_pairing.csv", index=False)
