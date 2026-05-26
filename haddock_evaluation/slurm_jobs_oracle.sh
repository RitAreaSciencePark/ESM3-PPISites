#!/bin/bash
#SBATCH --job-name=haddock3
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=40G
#SBATCH --time=30:00:00
#SBATCH --output=logs_oracle/log_oracle_rig100_top10_haddock_%j.out
#SBATCH --error=logs_oracle/log_oracle_rig100_top10_haddock_%j.err

BASE_DIR="data/haddock_units"
CONFIG_FILE="configs/config_oracle.cfg"

# Activate virtual environment once before the loop
source /orfeo/scratch/area/ssenci/venvs/haddock_env/bin/activate
echo "Virtual environment activated!"

OVERALL_EXIT=0

# Get the directory where the script was submitted from
SUBMIT_DIR=$(pwd)

# Iterate over each subdirectory
for RUN_DIR in "${BASE_DIR}"/*/; do

    # Skip if not a directory
    [ -d "$RUN_DIR" ] || continue

    WORKING_DIR=$(realpath "$RUN_DIR")
    DIR_NAME=$(basename "$WORKING_DIR")

    # Skip if config file is missing
    if [ ! -f "${WORKING_DIR}/${CONFIG_FILE}" ]; then
        echo "WARNING: ${CONFIG_FILE} not found in ${WORKING_DIR}, skipping."
        continue
    fi

    START_TIME=$(date +%s)
    START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo "============================================================"
    echo "=== Processing: ${DIR_NAME} | Started: ${START_TIME_READABLE} ==="
    echo "============================================================"

    cd "$WORKING_DIR" || { echo "ERROR: Cannot cd into ${WORKING_DIR}, skipping."; continue; }

    # Update ncores in config file
    sed -i "s/^ncores.*/ncores=${SLURM_CPUS_PER_TASK:-64}/" "$CONFIG_FILE"
    echo "Updated ncores to ${SLURM_CPUS_PER_TASK:-64} in ${CONFIG_FILE}"

    # Run HADDOCK3
    echo "Starting HADDOCK3 in ${DIR_NAME}..."
    haddock3 "$CONFIG_FILE"
    EXIT_CODE=$?

    # Capture end time and calculate duration
    END_TIME=$(date +%s)
    DURATION_MIN=$(awk "BEGIN {printf \"%.2f\", ($END_TIME - $START_TIME)/60}")

    # Determine job status
    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
        OVERALL_EXIT=1
    fi

    echo "=== ${DIR_NAME} Complete: ${STATUS} (${DURATION_MIN} min) | Exit: ${EXIT_CODE} ==="

    # Return to submit directory before next iteration
    cd "$SUBMIT_DIR"

done

echo ""
echo "=== All directories processed. Overall exit code: ${OVERALL_EXIT} ==="
exit $OVERALL_EXIT
