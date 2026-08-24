#!/bin/bash
BASEDIR="haddock_units"
LISTFILE="jobs_dirlist.txt"

find "$BASEDIR" -mindepth 1 -maxdepth 1 -type d | sort > "$LISTFILE"
N=$(wc -l < "$LISTFILE")

[[ "$N" -eq 0 ]] && { echo "No directories found in $BASEDIR"; exit 1; }

echo "Found $N directories"
sbatch --array=0-$((N-1)) run_unit_array.sh "$LISTFILE"
