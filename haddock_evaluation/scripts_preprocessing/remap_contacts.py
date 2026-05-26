#!/usr/bin/env python3
import argparse
import os
import re
import sys

# [read_tsv, write_tsv, find_contacts_file, find_vs_files, build_lookup functions remain as they were]
# (Included here for completeness)

def read_tsv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.rstrip("\n")]
    if not lines: raise ValueError(f"File is empty: {path}")
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header): fields += [""] * (len(header) - len(fields))
        rows.append({col: fields[i].strip() for i, col in enumerate(header)})
    return header, rows

def write_tsv(path: str, header: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows: fh.write("\t".join(row.get(col, "") for col in header) + "\n")

def find_contacts_file(directory: str) -> str:
    hits = [f for f in os.listdir(directory) if re.fullmatch(r"contacts_.+\.tsv", f)]
    if not hits: sys.exit(f"[ERROR] No contacts file in {directory}")
    return hits[0]

def find_vs_files(directory: str, complex_name: str) -> list[str]:
    pattern = re.compile(rf"{re.escape(complex_name)}_vs_.+\.tsv$")
    hits = [f for f in os.listdir(directory) if pattern.fullmatch(f)]
    if len(hits) != 2: sys.exit(f"[ERROR] Expected 2 vs-files in {directory}, found {len(hits)}")
    return hits

def build_lookup(directory: str, vs_files: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    pos_map, chain_labels = {}, {}
    for fname in vs_files:
        path = os.path.join(directory, fname)
        header, rows = read_tsv(path)
        chain_letter = {r["chain_complex"] for r in rows if r["chain_complex"]}.pop()
        pos_map[chain_letter] = {r["position_complex"]: r["position_chain"] for r in rows if r["position_complex"]}
        chain_labels[chain_letter] = next(iter({r["pdb_chain"] for r in rows if r["pdb_chain"]}))
    return pos_map, chain_labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate contact map.")
    parser.add_argument("contacts_dir", help="Directory with contacts_*.tsv")
    parser.add_argument("aln_dir", help="Directory with *_vs_*.tsv")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    contacts_dir = os.path.abspath(args.contacts_dir)
    aln_dir = os.path.abspath(args.aln_dir)

    contacts_fname = find_contacts_file(contacts_dir)
    complex_name = re.sub(r"^contacts_", "", re.sub(r"\.tsv$", "", contacts_fname))
    
    vs_files = find_vs_files(aln_dir, complex_name)
    pos_map, chain_labels = build_lookup(aln_dir, vs_files)

    header, rows = read_tsv(os.path.join(contacts_dir, contacts_fname))
    new_header = header + ["mapped_to_chain_i", "mapped_to_chain_j"]

    for row in rows:
        row["mapped_to_chain_i"] = pos_map.get(row["chain_i"], {}).get(row["resseq_i"], "")
        row["mapped_to_chain_j"] = pos_map.get(row["chain_j"], {}).get(row["resseq_j"], "")

    rows = [r for r in rows if r["mapped_to_chain_i"] and r["mapped_to_chain_j"]]
    
    output_path = args.output or os.path.join(contacts_dir, f"{complex_name}_{chain_labels.get(rows[0]['chain_i'])}_{chain_labels.get(rows[0]['chain_j'])}_targetPairs.tsv")
    write_tsv(output_path, new_header, rows)
    print(f"[INFO] Output written : {output_path}")

if __name__ == "__main__":
    main()

