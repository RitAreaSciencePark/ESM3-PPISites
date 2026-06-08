#!/bin/bash
#SBATCH --job-name=sampling
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=35G
#SBATCH --time=12:00:00
#SBATCH --output=logs_eval/log_eval_%j.out
#SBATCH --error=logs_eval/log_eval_%j.err

source /orfeo/cephfs/scratch/area/ssenci/senci_projects/LASTRUN/env_repo/bin/activate
echo "Virtual environment activated!"

# Get number of available CPUs from SLURM
NPROC=$SLURM_CPUS_PER_TASK

base="data/haddock_units/"
ids=($(ls $base))  

for id in "${ids[@]}"
do
	echo $id "STARTED:"
	date

	config_repo=$base$id"/configs/"
	configs=($(ls "$config_repo" | grep '^patches'))

	for config in "${configs[@]}"
	do
		echo $config_repo$config
		
		# Replace ncores in config file with SLURM nproc
		sed -i "s/^ncores = .*/ncores = $NPROC/" $config_repo$config
		
		haddock3 $config_repo$config
	done
	echo $id "FINISHED:"
	date
done

