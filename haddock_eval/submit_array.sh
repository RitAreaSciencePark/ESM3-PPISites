#!/bin/bash
BASEDIR="haddock_units"
LISTFILE="jobs_dirlist.txt"

find "$BASEDIR" -mindepth 1 -maxdepth 1 -type d | sort > "$LISTFILE"
N=$(wc -l < "$LISTFILE")

[[ "$N" -eq 0 ]] && { echo "No directories found in $BASEDIR"; exit 1; }

echo "Found $N directories"

VENV_DIR="${PWD}/.haddock_venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install haddock3
fi

sbatch --array=0-$((N-1)) run_unit_array.sh "$LISTFILE"

