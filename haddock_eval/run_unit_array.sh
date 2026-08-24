#!/bin/bash
#SBATCH --job-name=haddock3
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=20G
#SBATCH --time=30:00:00
#SBATCH --output=logs/log_%A_%a_%x.out
#SBATCH --error=logs/log_%A_%a_%x.err

LISTFILE="$1"

if [[ -z "$LISTFILE" || ! -f "$LISTFILE" ]]; then
    echo "Usage: sbatch --array=0-N $0 <listfile>"
    exit 1
fi

mkdir -p logs

WORKING_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LISTFILE")
WORKING_DIR=$(realpath "$WORKING_DIR")

if [[ -z "$WORKING_DIR" || ! -d "$WORKING_DIR" ]]; then
    echo "ERROR: Directory not found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CONFIG_DIR="${WORKING_DIR}/configs"
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: configs/ not found in ${WORKING_DIR}"
    exit 1
fi

conda activate ppi

DIR_NAME=$(basename "$WORKING_DIR")

for CONFIG_FILE in "${CONFIG_DIR}"/*.cfg; do
    [[ -f "$CONFIG_FILE" ]] || continue

    CFG_NAME=$(basename "$CONFIG_FILE")
    LOG="logs/log_${DIR_NAME}_${CFG_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

    {
        echo "=== $(date '+%F %T') | ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | ${DIR_NAME} | ${CFG_NAME} ==="

        sed -i "s/^ncores.*/ncores=${SLURM_CPUS_PER_TASK:-25}/" "$CONFIG_FILE"

        start=$SECONDS
        haddock3 "$CONFIG_FILE"
        code=$?

        echo "=== Exit: $code | Duration: $((SECONDS - start))s ==="
    } > "$LOG" 2>&1

done

