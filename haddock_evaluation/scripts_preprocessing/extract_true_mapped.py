#!/usr/bin/env python3
import os
import re
from pathlib import Path
from collections import defaultdict

def extract_hotspots(base_dir="data/haddock_units"):
    """
    Extract hotspot residues from HADDOCK contact files.
    
    Args:
        base_dir: Path to the directory containing HADDOCK units
        
    Returns:
        List of tuples (complex, id, hotspot_true, original_hotspot_true)
    """
    results = []
    haddock_dir = Path(base_dir)
    
    # Iterate through each complex directory
    for complex_dir in sorted(haddock_dir.iterdir()):
        if not complex_dir.is_dir():
            continue
        
        complex_id = complex_dir.name
        contacts_dir = complex_dir / "contacts"
        
        if not contacts_dir.exists():
            continue
        
        # Find targetPairs.tsv files
        pattern = f"{complex_id}_WZ_*_targetPairs.tsv"
        target_files = list(contacts_dir.glob(pattern))
        
        if not target_files:
            continue
        
        target_file = target_files[0]
        
        # Parse filename to get chain identifiers
        # Expected format: {complex_id}_WZ_{chainW}_W_{chainZ}_Z_targetPairs.tsv
        match = re.search(r"_WZ_(.+)_W_(.+)_Z_targetPairs", target_file.name)
        if not match:
            continue
        
        chain_w = match.group(1)
        chain_z = match.group(2)
        
        # Sets to store mapped positions
        hotspots_w = set()
        hotspots_z = set()
        
        # Sets to store original positions (resseq)
        orig_hotspots_w = set()
        orig_hotspots_z = set()
        
        with open(target_file, 'r') as f:
            header_line = f.readline()
            if not header_line:
                continue
                
            # Parse header to dynamically map column indices to their names
            headers = header_line.strip().split('\t')
            try:
                idx_chain_i = headers.index('chain_i')
                idx_chain_j = headers.index('chain_j')
                idx_mapped_i = headers.index('mapped_to_chain_i')
                idx_mapped_j = headers.index('mapped_to_chain_j')
                idx_resseq_i = headers.index('resseq_i')
                idx_resseq_j = headers.index('resseq_j')
            except ValueError as e:
                print(f"Skipping {target_file.name}: missing required column header. Error: {e}")
                continue
            
            for line in f:
                fields = line.strip().split('\t')
                # Safety check to ensure line isn't truncated relative to calculated indices
                if len(fields) <= max(idx_chain_i, idx_chain_j, idx_mapped_i, idx_mapped_j, idx_resseq_i, idx_resseq_j):
                    continue
                
                chain_i = fields[idx_chain_i]
                chain_j = fields[idx_chain_j]
                mapped_to_i = fields[idx_mapped_i]
                mapped_to_j = fields[idx_mapped_j]
                resseq_i = fields[idx_resseq_i]
                resseq_j = fields[idx_resseq_j]
                
                # Process Chain W
                if chain_i == 'W' and mapped_to_i:
                    hotspots_w.add(mapped_to_i)
                    if resseq_i: orig_hotspots_w.add(resseq_i)
                if chain_j == 'W' and mapped_to_j:
                    hotspots_w.add(mapped_to_j)
                    if resseq_j: orig_hotspots_w.add(resseq_j)
                
                # Process Chain Z
                if chain_i == 'Z' and mapped_to_i:
                    hotspots_z.add(mapped_to_i)
                    if resseq_i: orig_hotspots_z.add(resseq_i)
                if chain_j == 'Z' and mapped_to_j:
                    hotspots_z.add(mapped_to_j)
                    if resseq_j: orig_hotspots_z.add(resseq_j)
        
        # Format hotspot strings (sorted numerically)
        hotspot_w_str = ','.join(map(str, sorted(hotspots_w, key=int))) if hotspots_w else ""
        hotspot_z_str = ','.join(map(str, sorted(hotspots_z, key=int))) if hotspots_z else ""
        
        orig_hotspot_w_str = ','.join(map(str, sorted(orig_hotspots_w, key=int))) if orig_hotspots_w else ""
        orig_hotspot_z_str = ','.join(map(str, sorted(orig_hotspots_z, key=int))) if orig_hotspots_z else ""
        
        # Add results for both chains
        if hotspot_w_str:
            results.append((complex_id, f"{chain_w}_W", hotspot_w_str, orig_hotspot_w_str))
        if hotspot_z_str:
            results.append((complex_id, f"{chain_z}_Z", hotspot_z_str, orig_hotspot_z_str))
            
    return results


def write_summary(results, output_file="contacts_summary/remapped_contacts_summary.tsv"):
    """Write results to TSV file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Adjusted header to account for the original position column
        f.write("complex\tid\tmonomer_hotspot_true\tcomplex_hotspot_true\n")
        
        # Write data rows
        for complex_id, chain_id, hotspots, orig_hotspots in results:
            f.write(f"{complex_id}\t{chain_id}\t{hotspots}\t{orig_hotspots}\n")
            
    print(f"Summary written to {output_file}")


if __name__ == "__main__":
    results = extract_hotspots()
    write_summary(results)