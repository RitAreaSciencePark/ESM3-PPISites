REF_FP = 'pdb_lookup.tsv'		                            #F# args.reference
DEST_CIFS = 'cifs'                                          #F# args.cif-dir
HADDOCK_BASEDIR = 'haddock_units'		                    #F# args.haddock-basedir
BEHAVIOUR = None                                            #F# args.behaviour
METADATA_DIR = 'metadata'#F# args.metadata-path
INFERENCE_DIR = 'inference'#F# args.inference-path
METADATA_FP = f'{METADATA_DIR}/RCSB_UniProtKB_crossref.tsv' #F# args.metadata-path
STANDARD_CHAINS = ['W', 'Z']                        	    #F# args.standard-chains
PDB_KEYS = ['PDB_cc', 'PDB_c1', 'PDB_c2']   
CHAIN_KEYS = ['ids_cc', 'id_c1', 'id_c2']   
REQUIRED_KEYS = ['jobname'] + PDB_KEYS + CHAIN_KEYS

DIST_CA = 0.8   ### nanometers			    #F# args.dist-ca

DIST = 10.0     ### Angstroms
D_MINUS = 6.0   ### Angstroms
D_PLUS = 2.0    ### Angstroms 
FNAT_CUTOFF = 8 ### Angstroms
DEF_NCORES = 64

RUN_INFERENCE_SCRIPT = '../run_prediction.py'
INFERENCE_INPUT = f'{INFERENCE_DIR}/ESM3PPISites_inference_input.csv' #..# args.input
INFERENCE_OUTPUT = f'{INFERENCE_DIR}/ESM3PPISites_inference_results.csv' #..# args.output
MATCHED_INFERENCE_FP = f"{INFERENCE_DIR}/ESM3PPISites_inference_long.tsv" # args.output
MAPPED_INFERENCE_FP = f"{INFERENCE_DIR}/ESM3PPISites_inference_mapped.tsv" # args.output
PATCHES_FP = f'{INFERENCE_DIR}/ESM3PPISites_linear_patches.tsv' #..# args.output
PATCHDATA_FP = f"{INFERENCE_DIR}/ESM3PPISites_patches_statistics.tsv" 
FOR_PAIRING_FP = f'{INFERENCE_DIR}/ESM3PPISites_patches_for_pairing.tsv' #..# args.output

PROB_CUTOFF = 0.6
LOWER_PROB_CUTOFF = 0.5
MIN_NUM_PATCHES = 3

MAX_RES_DISTANCE = 6
MIN_PATCH_LENGTH = 4 
ALT_PATCH_LENGTH = 3
MIN_RES_CLUSTERING = 4
ALT_RES_CLUSTERING = 3
N_TOP_PATCHES = 4
