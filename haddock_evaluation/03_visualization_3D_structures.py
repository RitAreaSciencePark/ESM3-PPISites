#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import py3Dmol


# In[2]:


# Load datasets
all_contacts = pd.read_csv("contacts_summary/remapped_contacts_summary.tsv", sep="\t")
all_preds = pd.read_csv("inference_results/db5_results_remapped.csv")

COMPLEX = "1KXP"
UNIT = Path("data", "haddock_units",COMPLEX)

pdb_Z = next(UNIT.glob("*_Z.pdb"), None)
pdb_W = next(UNIT.glob("*_W.pdb"), None)
pdb_WZ = next(UNIT.glob("*_WZ.pdb"), None)

# Quick safety check to ensure files actually exist
for name, path in [("_Z", pdb_Z), ("_W", pdb_W), ("_WZ", pdb_WZ)]:
    if path is None:
        print(f"Warning: File ending with {name}.pdb not found in {UNIT}")

# 2. Filter dataframes by the COMPLEX ID
contacts = all_contacts[all_contacts["complex"] == COMPLEX]
preds = all_preds[all_preds["complex"] == COMPLEX]


# In[3]:


contacts


# In[4]:


preds


# In[5]:


import py3Dmol

def visualize_pdb_highlight(pdb_fp, W_true=None, Z_true=None, W_pred=None, Z_pred=None):
    def _parse_positions(positions):
        """Flatten and parse positions that may be strings, lists, or comma-separated values."""
        result = set()
        if positions is None:
            return result

        # Ensure we're iterating over a list
        if isinstance(positions, (str, int)):
            positions = [positions]

        for pos in positions:
            if isinstance(pos, str):
                # Handle comma-separated values like "128,129,130"
                for p in pos.split(','):
                    p = p.strip()
                    if p:
                        result.add(int(p))
            elif isinstance(pos, (list, tuple, set)):
                # Handle nested lists
                for p in pos:
                    result.add(int(p))
            else:
                # Single integer
                result.add(int(pos))
        return result

    W_true = _parse_positions(W_true)
    Z_true = _parse_positions(Z_true)
    W_pred = _parse_positions(W_pred)
    Z_pred = _parse_positions(Z_pred)

    with open(pdb_fp) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_data, "pdb")

    chains = sorted(set(line[21] for line in pdb_data.splitlines() 
                        if line.startswith("ATOM  ") and line[21] in "WZ"))

    colors = {chains[0]: "#333333"} if chains else {}
    if len(chains) > 1:
        colors[chains[1]] = "#777777"

    for chain in chains:
        sel = {"chain": chain}
        color = colors.get(chain, "white")
        view.setStyle(sel, {"cartoon": {"color": color}})

        if chain == "W":
            true_set, pred_set = W_true, W_pred
        elif chain == "Z":
            true_set, pred_set = Z_true, Z_pred
        else:
            continue

        red_positions = true_set - pred_set
        green_positions = true_set & pred_set
        yellow_positions = pred_set - true_set

        for pos in red_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": "red"}})
        for pos in green_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": "green"}})
        for pos in yellow_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": "orange"}})

    view.zoomTo()
    return view.show()

import py3Dmol

import py3Dmol

def visualize_pdb_highlight2(pdb_fp, W_true=None, Z_true=None, W_pred=None, Z_pred=None, title=""):
    def _parse_positions(positions):
        """Flatten and parse positions that may be strings, lists, or comma-separated values."""
        result = set()
        if positions is None:
            return result

        # Ensure we're iterating over a list
        if isinstance(positions, (str, int)):
            positions = [positions]

        for pos in positions:
            if isinstance(pos, str):
                # Handle comma-separated values like "128,129,130"
                for p in pos.split(','):
                    p = p.strip()
                    if p:
                        result.add(int(p))
            elif isinstance(pos, (list, tuple, set)):
                # Handle nested lists
                for p in pos:
                    result.add(int(p))
            else:
                # Single integer
                result.add(int(pos))
        return result

    W_true = _parse_positions(W_true)
    Z_true = _parse_positions(Z_true)
    W_pred = _parse_positions(W_pred)
    Z_pred = _parse_positions(Z_pred)

    with open(pdb_fp) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_data, "pdb")

    chains = sorted(set(line[21] for line in pdb_data.splitlines() 
                        if line.startswith("ATOM  ") and line[21] in "WZ"))

    # Clean, high-contrast palette values
    palette = {
        "W": "#1E293B",   # Deep slate blue
        "Z": "#CBD5E1",   # Light slate gray
        "TP": "#2ECC71",  # True Positives: Green
        "FN": "#E74C3C",  # False Negatives: Red
        "FP": "gold"   # False Positives: Gold
    }

    colors = {chains[0]: palette["W"]} if chains else {}
    if len(chains) > 1:
        colors[chains[1]] = palette["Z"]

    for chain in chains:
        sel = {"chain": chain}
        color = colors.get(chain, "white")
        view.setStyle(sel, {"cartoon": {"color": color}})

        if chain == "W":
            true_set, pred_set = W_true, W_pred
        elif chain == "Z":
            true_set, pred_set = Z_true, Z_pred
        else:
            continue

        red_positions = true_set - pred_set
        green_positions = true_set & pred_set
        yellow_positions = pred_set - true_set

        for pos in red_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": palette["FN"]}})
        for pos in green_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": palette["TP"]}})
        for pos in yellow_positions:
            view.addStyle({"chain": chain, "resi": pos}, {"cartoon": {"color": palette["FP"]}})

    view.zoomTo()

    # Base label styling dictionary to avoid repetitive code
    label_style = {
        "useScreen": True,
        "fontSize": 12,
        "fontColor": "black",
        "backgroundColor": "white",
        "backgroundOpacity": 0.85
    }

    # Add Title in the top-left corner if provided
    if title:
        view.addLabel(title, {
            **label_style,
            "fontSize": 14,                # Made slightly larger so it stands out
            "fontColor": "#1E293B",        # Sleek dark navy/slate text
            "position": {"x": 299, "y": 10, "z": 0}  # Top-left anchor
        })

    # 3 INDEPENDENT STACKED LABELS IN THE BOTTOM LEFT CORNER    
    view.addLabel("Green: True Positives (Hit)", 
                  {**label_style, "position": {"x": 50, "y": 368, "z": 0}})

    view.addLabel("Red: False Negatives (Miss)", 
                  {**label_style, "position": {"x": 50, "y": 346, "z": 0}})

    view.addLabel("Gold: False Positives (Extra)", 
                  {**label_style, "position": {"x": 50, "y": 324, "z": 0}})

    return view.show()


# In[6]:


pos_W_true = contacts[contacts['id'].str.endswith("_W")]['complex_hotspot_true'].tolist()
pos_Z_true = contacts[contacts['id'].str.endswith("_Z")]['complex_hotspot_true'].tolist()
pos_W_pred = preds[preds['id'].str.endswith("_W")]['complex_hotspot_pred'].tolist()
pos_Z_pred = preds[preds['id'].str.endswith("_Z")]['complex_hotspot_pred'].tolist()


# In[7]:


pos_W_true
#pos_Z_true
#pos_W_pred


# In[8]:


visualize_pdb_highlight2(pdb_WZ, W_true=pos_W_true, Z_true=pos_Z_true, W_pred=pos_W_pred, Z_pred=pos_Z_pred, title=COMPLEX)


# In[9]:


from pathlib import Path

# Define the base directory containing the units
base_dir = Path("data", "haddock_units")

# Iterate through every directory inside haddock_units
for unit_dir in base_dir.iterdir():
    if not unit_dir.is_dir():
        continue  # Skip files, only process directories

    COMPLEX = unit_dir.name

    # 1. Locate the PDB files dynamically
    pdb_Z = next(unit_dir.glob("*_Z.pdb"), None)
    pdb_W = next(unit_dir.glob("*_W.pdb"), None)
    pdb_WZ = next(unit_dir.glob("*_WZ.pdb"), None)

    # Quick safety check to ensure files actually exist
    missing_file = False
    for name, path in [("_Z", pdb_Z), ("_W", pdb_W), ("_WZ", pdb_WZ)]:
        if path is None:
            print(f"Warning: File ending with {name}.pdb not found in {unit_dir}")
            missing_file = True

    if missing_file:
        print(f"Skipping complex {COMPLEX} due to missing PDB files.\n")
        continue

    # 2. Filter dataframes by the COMPLEX ID
    contacts = all_contacts[all_contacts["complex"] == COMPLEX]
    preds = all_preds[all_preds["complex"] == COMPLEX]

    # Extract lists
    pos_W_true = contacts[contacts['id'].str.endswith("_W")]['complex_hotspot_true'].tolist()
    pos_Z_true = contacts[contacts['id'].str.endswith("_Z")]['complex_hotspot_true'].tolist()
    pos_W_pred = preds[preds['id'].str.endswith("_W")]['complex_hotspot_pred'].tolist()
    pos_Z_pred = preds[preds['id'].str.endswith("_Z")]['complex_hotspot_pred'].tolist()

    # Build title: strip _W, _Z, _WZ suffixes, uppercase, space after colon
    wz_id = pdb_WZ.stem.removesuffix("_WZ").upper()
    w_id = pdb_W.stem.removesuffix("_W").upper()
    z_id = pdb_Z.stem.removesuffix("_Z").upper()
    title = f"WZ: {wz_id} - W: {w_id} - Z: {z_id}"

    # 3. Visualize
    print(f"==================================================")
    visualize_pdb_highlight2(
        pdb_WZ, 
        W_true=pos_W_true, 
        Z_true=pos_Z_true, 
        W_pred=pos_W_pred, 
        Z_pred=pos_Z_pred,
        title=title
    )


# In[ ]:





# In[ ]:





# In[ ]:




