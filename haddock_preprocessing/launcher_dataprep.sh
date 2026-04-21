#!/bin/bash
#SBATCH --job-name=pdbprep
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=2G         
#SBATCH --time=12:00:00          
#SBATCH --output=slurm_logs/pdbprep_%j.out
#SBATCH --error=slurm_logs/pdbprep_%j.err

echo "SECTION: Starting..."
source /orfeo/scratch/area/ssenci/venvs/ml_transformers/bin/activate
echo "loaded venv ml_transformers"

echo "SECTION: Download"
python3 scripts/01_download_PDBs.py
echo "-----\n"
echo "-----\n"
echo "-----\n"


echo "SECTION: W,Z,WZ chain filtering"
python3 scripts/02_preprocessing.py
echo "-----\n"
echo "-----\n"
echo "-----\n"


echo "SECTION: tmptsv files"
python3 scripts/03_toresidues.py
echo "-----\n"
echo "-----\n"
echo "-----\n"

echo "SECTION: contact maps, contacts"
python3 scripts/04_compute_contacts.py
echo "-----\n"
echo "-----\n"
echo "-----\n"

echo "SECTION: alignment of dimeric_pdb to single_chain"
python3 scripts/05_align_chains.py
echo "-----\n"
echo "-----\n"
echo "-----\n"

echo "SECTION: contact pairs remapped to single_chain"
python3 scripts/06_make_targetpairs.py
echo "-----\n"
echo "-----\n"
echo "-----\n"
