#!/usr/bin/env python3
"""
"""

### <deps>
import os
import re
import numpy as np
import pandas as pd
import urllib
from pathlib import Path, PosixPath
import warnings
from collections import Counter, defaultdict
from typing import Union, List
from tqdm import tqdm
from types import SimpleNamespace
import tempfile
import hashlib

import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser
from Bio.PDB import PDBParser, Select, Selection 
from Bio.PDB import Structure, Model, PDBIO
from Bio.PDB.Chain import Chain
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import protein_letters_3to1_extended

import mdtraj as md
#import pymol
#from pymol import cmd
### <!deps>


###
###
###

def download_rcsb(pdb_id: str, dest: Path, file_type: str = "cif"):
    """Arguments --- pdb id, file path destination, file format to download    #F# Allow buffering (get_?)
       Returns --- .cif an/or .pdb from RCSB API, single entry download
       Raises --- error from try """     #F# Raise, connection                          
    rcsb_url = {                                                               #F# Raise, pdb id valid
        "cif":   "https://files.rcsb.org/download/{id}.cif",
        "pdb":   "https://files.rcsb.org/download/{id}.pdb"#,                  #C# removed rcsb fasta
        # "fasta": "https://www.rcsb.org/fasta/entry/{id}"                     #C# can extract from file
    }
    url = rcsb_url[file_type.lower()].format(id=pdb_id.lower())
    try:
        #print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        #print(f"  Saved to:    {dest}")
    except Exception as e:
        print(f"  [ERROR] Could not download {url}: {e}")

def extract_uniprot_from_cifs(cif_paths: Union[str, List[str]]) -> pd.DataFrame:
    UNIPROT_RE = re.compile(
        r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$'
    )
    if isinstance(cif_paths, str):
        cif_paths = [cif_paths]
    def as_list(value):
        return value if isinstance(value, list) else ([value] if value is not None else [])
    def classify(accession: str):
        if not accession:
            return 'missing', None
        token = str(accession).strip().upper()
        if UNIPROT_RE.match(token):
            return 'valid', token
        return 'alt', token
    rows = []
    for cif_path in cif_paths:
        pdb_id = os.path.splitext(os.path.basename(cif_path))[0].replace('.cif', '')
        cif_dict = MMCIF2Dict.MMCIF2Dict(cif_path)
        strand_ids = as_list(cif_dict.get('_struct_ref_seq.pdbx_strand_id'))
        db_accs = as_list(cif_dict.get('_struct_ref_seq.pdbx_db_accession'))
        db_accs_alt = as_list(cif_dict.get('_struct_ref_seq.db_accession'))
        ref_ids = as_list(cif_dict.get('_struct_ref_seq.ref_id'))
        struct_ref_ids = as_list(cif_dict.get('_struct_ref.id'))
        struct_ref_accs = as_list(cif_dict.get('_struct_ref.pdbx_db_accession'))
        ref_id_to_acc = {
            str(ref_id).strip(): str(accession).strip()
            for ref_id, accession in zip(struct_ref_ids, struct_ref_accs)
            if ref_id is not None and accession is not None
        }
        num_entries = len(strand_ids)
        def pad_items(items):
            return items + [None] * (num_entries - len(items)) if len(items) < num_entries else items
        db_accs = pad_items(db_accs)
        db_accs_alt = pad_items(db_accs_alt)
        ref_ids = pad_items(ref_ids)
        chain_valid = {}
        chain_alt = {}
        for idx in range(num_entries):
            strand_raw = strand_ids[idx]
            if not strand_raw:
                continue
            candidates = []
            if db_accs[idx] is not None:
                candidates.append(str(db_accs[idx]).strip())
            if db_accs_alt[idx] is not None:
                candidates.append(str(db_accs_alt[idx]).strip())
            if ref_ids[idx] is not None:
                accession = ref_id_to_acc.get(str(ref_ids[idx]).strip())
                if accession:
                    candidates.append(accession)
            for chain_id in [segment.strip() for segment in str(strand_raw).split(',') if segment.strip()]:
                for candidate in candidates:
                    status, value = classify(candidate)
                    if status == 'valid':
                        chain_valid.setdefault(chain_id, []).append(value)
                    elif status == 'alt':
                        chain_alt.setdefault(chain_id, []).append(value)
        all_chains = set(chain_valid) | set(chain_alt)
        for chain_id in all_chains:
            uniprot_id = None
            alt_id = None
            if chain_id in chain_valid:
                uniprot_id = Counter(chain_valid[chain_id]).most_common(1)[0][0]
            if chain_id in chain_alt:
                alt_list = chain_alt[chain_id]
                counter = Counter(alt_list)
                best_alt, best_count = counter.most_common(1)[0]
                if best_count != len(alt_list):
                    warnings.warn(f"Alt ID conflict for {pdb_id} chain {chain_id}: {dict(counter)} -> using {best_alt}")
                alt_id = best_alt
            rows.append({
                'pdb_id': pdb_id,
                'chain': chain_id,
                'uniprot_id': uniprot_id,
                'alt_identifier': alt_id,
            })
    return pd.DataFrame(rows)

def download_uniprot_fasta(uniprot_id: str, output_path: str):
    """Accessory function that is useful when using Alignment modules
       Arguments --- uniprot entry and output file path 
       Returns --- .fasta file at path (single entry, no line wrapping)    #F# multifasta?
       Raises --- Error on url existance, connection, server response"""
    base_up_url = "https://rest.uniprot.org/uniprotkb/{}.fasta"
    url = base_up_url.format(uniprot_id)
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8').strip()
            if not data:
                raise Exception(f"No data returned for {uniprot_id}")
    except urllib.error.HTTPError as e:
        raise Exception(f"Failed to download {uniprot_id}: HTTP Error {e.code}")
    except urllib.error.URLError as e:
        raise Exception(f"Failed to reach server: {e.reason}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(data)

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id

def select_chains(structure, *chain_ids, keep_first_model_only=True):
    if not hasattr(structure, "id") or not hasattr(structure, "get_chains"):
        raise TypeError(
            f"Expected Bio.PDB.Structure.Structure, got {type(structure).__name__}"
        )
    if not chain_ids:
        raise ValueError("At least one chain ID must be provided")
    available = {c.id for c in Selection.unfold_entities(structure, "C")}
    if not available:
        raise ValueError(f"Structure '{structure.id}' contains no chains")
    if not any(cid in available for cid in chain_ids):
        raise ValueError(
            f"None of the requested chains {chain_ids} found in '{structure.id}'. "
            f"Available: {sorted(available)}"
        )
    if keep_first_model_only and len(structure) > 1:
        warnings.warn(f"{structure.id}: multiple models, using first")
        src = structure[0]
    else:
        src = structure
    target = set(chain_ids)
    chains = [c.copy() for c in src.get_chains() if c.id in target]
    if len(chain_ids) == 1:
        return chains[0]
    return chains

def get_polypeptide_chain(chain_bio, verbose=False):
    if not hasattr(chain_bio, 'id'):
        raise ValueError("not chain")
    cid = chain_bio.id
    res = Selection.unfold_entities(chain_bio, 'R')
    fail_chain = set()
    for r in res:
        rname = r.get_resname()
        rid = r.id
        aa = protein_letters_3to1_extended.get(rname)
        if aa is None:
            fail_chain.add(rname)
    if verbose == True:
        print(f'{cid}: removed {fail_chain}')
    new_chain = Chain(str(cid))
    for r in res:
        if protein_letters_3to1_extended.get(r.get_resname()) is not None:
            new_chain.add(r)
    return new_chain

def assemble_structure(*chains, structure_id):
    from Bio.PDB import Chain
    if not chains:
        raise ValueError("At least one chain must be provided")
    structure = Structure.Structure(structure_id)
    model = Model.Model(0)
    structure.add(model)
    
    for c in chains:
        if not isinstance(c, Chain.Chain):
            raise TypeError(f"Expected Bio.PDB.Chain.Chain, got {type(c).__name__}")
        model.add(c.copy())
        
    return structure

######################
def ca_contacts(fp, cutoff=0.8, mode="inter"):
    """Compute contact Carbon-alpha distance with filtering cutoff
       Relies on mdtraj and outputs a dataframe including index 
       and original residue id
       Arguments: file path to heterodimer, distance cutoff in nanometers
                  mode (inter|intra)
       Returns: dataframe with CA distances, chain, and residue data
    """
    fp = str(fp)
    traj = md.load(fp)
    top = traj.topology
    pdb_id = fp.split("/")[-1].split(".")[0]
    ca_atoms = [a for a in top.atoms if a.name == "CA"]
    if len(ca_atoms) < 2:
        return pd.DataFrame()
    pairs = []
    for i, ai in enumerate(ca_atoms):
        for j, aj in enumerate(ca_atoms[i + 1 :], start=i + 1):
            same_chain = ai.residue.chain.index == aj.residue.chain.index
            if mode == "intra" and not same_chain:
                continue
            if mode == "inter" and same_chain:
                continue
            pairs.append((ai.index, aj.index))
    if not pairs:
        return pd.DataFrame()
    pairs_arr = np.array(pairs)
    dists = md.compute_distances(traj, pairs_arr)[0]
    mask = dists <= cutoff
    pairs_arr = pairs_arr[mask]
    dists = dists[mask]
    contact_data = []
    for (ai_idx, aj_idx), d in zip(pairs_arr, dists):
        ri = top.atom(ai_idx).residue
        rj = top.atom(aj_idx).residue
        contact_data.append(
            {
                "pdb_id": pdb_id,
                "dist_nm": round(d, 6),
                "dist_A": round(d * 10, 3),
                "res_i": ri.index,
                "res_j": rj.index,
                "chain_i": ri.chain.chain_id,
                "chain_j": rj.chain.chain_id,
                "resname_i": ri.name,
                "resname_j": rj.name,
                "resseq_i": ri.resSeq,
                "resseq_j": rj.resSeq,
                "aminoacid_i": f"{ri.name}{ri.resSeq}",
                "aminoacid_j": f"{rj.name}{rj.resSeq}"
            }
        )
    return pd.DataFrame(contact_data)

def sequence_chain_bio(chain):
    """Extracts sequence as it is from residue names in Biopython chain entity
    Arguments --- chain entity parsed with Biopython
    Returns --- sequence extracted residude by residue as a string
    Raises --- error if passed object is not a Biopython chain, 
               passes on failed attempts"""
    if not hasattr(chain, 'id'):
        raise ValueError("Input must be a single Bio.PDB Chain object")
    sequence_parts = []
    resid_parts = []
    resnames = []
    for residue in chain:
        try:
            resname = residue.get_resname().strip().upper()
            if resname in protein_letters_3to1_extended:
                resid = residue.id[1]
                sequence_parts.append(protein_letters_3to1_extended[resname])
                resnames.append(resname)
                resid_parts.append(resid)
        except Exception:
            print(f'[WARNING] {resname} not recognized by `protein_letters_3to1_extended`')
            continue
    seq = ''.join(sequence_parts)
    resdf = pd.DataFrame({
        'resid': resid_parts,
        'resname' : resnames,
        'sequence': list(seq),
        'seq_index': range(len(seq))
    })
    return {'seq': seq, 'resdata': resdf}

### match_score=1.0, mismatch_score=-1.0, open_gap_score=-2.0, extend_gap_score=-0.5
def get_alignment(ref_seq, query_seq, match_score=1.0, mismatch_score=-1.0, open_gap_score=-2, extend_gap_score=-0.5):
    """Local pairwise alignment with tunable parameter configuration.
       Intended for mapping of PDB sequence (query) to full UniProtkB sequence (reference)              #F# should validate that len(query) < len(ref)
       Arguments --- reference and query sequence, optional alignment tunable parameters                
       Returns ---  highest scoring Bio.Align.Alignment object or None if no alignment found
       """ 
    mode = "local"
    aligner = PairwiseAligner(
        mode=mode,
        match_score=match_score,
        mismatch_score=mismatch_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )
    alignments = aligner.align(ref_seq, query_seq)
    if not alignments:
        return None
    return alignments[0] # best only

def aln_to_stats(aln, ref_id="ref", query_id="query"):
    if aln is None:
        return pd.DataFrame([{"ref_id": ref_id, "query_id": query_id, "score": None}])
    ref_len, query_len = len(aln.target), len(aln.query)
    c = aln.counts()
    ref_blocks, query_blocks = aln.aligned
    n_ref_aligned = sum(e - s for s, e in ref_blocks)
    n_query_aligned = sum(e - s for s, e in query_blocks)
    n_aligned = c.aligned
    return pd.DataFrame([{
        "ref_id": ref_id,
        "query_id": query_id,
        "score": aln.score,
        "ref_len": ref_len,
        "query_len": query_len,
        "aln_len": aln.length,
        "n_ident": c.identities,
        "n_mismatch": c.mismatches,
        "n_aligned": n_aligned,
        "n_gaps": c.gaps,
        "n_insertions": c.insertions,
        "n_deletions": c.deletions,
        "n_open_gaps": c.open_gaps,
        "ref_start": int(ref_blocks[0][0]) if len(ref_blocks) else None,
        "ref_end": int(ref_blocks[-1][1]) if len(ref_blocks) else None,
        "query_start": int(query_blocks[0][0]) if len(query_blocks) else None,
        "query_end": int(query_blocks[-1][1]) if len(query_blocks) else None,
        "n_ref_aligned": n_ref_aligned,
        "n_query_aligned": n_query_aligned,
        "query_coverage": round(n_query_aligned / query_len, 4) if query_len else 0.0,
        "ref_coverage": round(n_ref_aligned / ref_len, 4) if ref_len else 0.0,
        "pct_identity": round(c.identities / n_aligned * 100, 2) if n_aligned else 0.0,
        "pct_identity_vs_min": round(c.identities / min(ref_len, query_len) * 100, 2) if min(ref_len, query_len) else 0.0,
    }])

def aln_to_dataframe(aln, ref_id='ref', query_id='query'):
    def get_start_offsets(alignment):
        try:
            ref_blocks, query_blocks = alignment.aligned
            if len(ref_blocks) > 0:
                return int(ref_blocks[0][0]), int(query_blocks[0][0])
        except Exception:
            pass
        return 0, 0

    offsets = get_start_offsets(aln)
    ref_start_offset = offsets[0]
    query_start_offset = offsets[1]

    # FIXED: gapped sequences (not the original ungapped .target / .query)
    ref_aligned = aln[0]
    query_aligned = aln[1]

    rows = []
    ref_pos = ref_start_offset
    query_pos = query_start_offset
    alignment_pos = 0
    for x, y in zip(ref_aligned, query_aligned):
        alignment_pos += 1
        if x != '-':
            ref_pos += 1
        if y != '-':
            query_pos += 1
        if x == '-' and y == '-':
            continue  # should not occur with PairwiseAligner
        rows.append({
            'aln_pos': alignment_pos,
            'ref_id': ref_id,
            'ref_pos': ref_pos if x != '-' else -1,
            'ref_aa': x,
            'query_id': query_id,
            'query_pos': query_pos if y != '-' else -1,
            'query_aa': y,
            'match_aln': '1' if x == y and x != '-' else '0'
        })
    return pd.DataFrame(rows)

def assemble_structure(*chains, structure_id):
    from Bio.PDB import Chain
    if not chains:
        raise ValueError("At least one chain must be provided")
    structure = Structure.Structure(structure_id)
    model = Model.Model(0)
    structure.add(model)
    for c in chains:
        if not isinstance(c, Chain.Chain):
            raise TypeError(f"Expected Bio.PDB.Chain.Chain, got {type(c).__name__}")
        model.add(c.copy())
    return structure

def align_pdb_structures(mol1_path: str, mol2_path: str, return_structure: bool = False):
    with tempfile.TemporaryDirectory() as tmp_dir:

        def prepare(path):
            if path.endswith(".gz"):
                h = hashlib.md5(path.encode()).hexdigest()[:8]
                out = os.path.join(tmp_dir, f"{h}_{os.path.basename(path)[:-3]}")
                with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return out
            return path

        mol1_local, mol2_local = prepare(mol1_path), prepare(mol2_path)
        mol1_name = f"mol1_{hashlib.md5(mol1_path.encode()).hexdigest()[:8]}"
        mol2_name = f"mol2_{hashlib.md5(mol2_path.encode()).hexdigest()[:8]}"

        cmd.reinitialize()
        cmd.load(mol1_local, mol1_name)
        cmd.load(mol2_local, mol2_name)

        (rmsd_after, n_atoms_aligned, n_cycles, rmsd_before,
         n_atoms_pre, score, n_res_aligned) = cmd.align(mol1_name, mol2_name)

        df = pd.DataFrame([{
            "mol_1": mol1_path, "mol_2": mol2_path,
            "rmsd": rmsd_after, "rmsd_before_refinement": rmsd_before,
            "n_atoms_aligned": n_atoms_aligned, "n_cycles": n_cycles,
            "n_atoms_pre_refinement": n_atoms_pre, "score": score,
            "n_residues_aligned": n_res_aligned,
        }])

        if return_structure:
            aligned_structures = {
                "mol_1": cmd.get_pdbstr(mol1_name),
                "mol_2": cmd.get_pdbstr(mol2_name),
            }
            return df, aligned_structures

    return df

############# FINAL BOUNDARY 


