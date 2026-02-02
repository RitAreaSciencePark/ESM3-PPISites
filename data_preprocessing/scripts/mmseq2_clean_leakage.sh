#!/bin/bash

# Usage: ./mmseqs2_clean_leakage.sh <fasta_dir> <csv_dir> <train_filename> <test_filename>

# Exit immediately if a command fails
set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <fasta_dir> <csv_dir> <train_file.fasta> <test_file.fasta>"
    exit 1
fi

FASTA_DIR="$1"
CSV_DIR="$2"

# 1. Inputs (Using basename ensures we get just the filename without extensions/paths)
# The script assumes inputs are just filenames (e.g. "train.fasta") present in the specified folders.
INPUT_NAME_1=$(basename "$3" .fasta)
INPUT_NAME_2=$(basename "$4" .fasta)

TRAIN_FASTA="$FASTA_DIR/$3.fasta"
TEST_FASTA="$FASTA_DIR/$4.fasta"
TRAIN_CSV="$CSV_DIR/$INPUT_NAME_1.csv"

# Check if files exist
if [[ ! -f "$TRAIN_FASTA" || ! -f "$TEST_FASTA" || ! -f "$TRAIN_CSV" ]]; then
    echo "Error: Could not find one of the input files."
    echo "Checked: $TRAIN_FASTA, $TEST_FASTA, $TRAIN_CSV"
    exit 1
fi

# 2. Setup Temporary Directory
TMP_DIR="mmseqs_tmp"
mkdir -p "$TMP_DIR"

# Define paths inside the temp folder
DB1="$TMP_DIR/train_db"
DB2="$TMP_DIR/test_db"
RES_DB="$TMP_DIR/result_db"
MATCHES_FILE="$TMP_DIR/matches.m8"
BLOCKLIST="$TMP_DIR/ids_to_remove.txt"
INTERNAL_TMP="$TMP_DIR/tmp_calc"

mkdir -p clean_output
CLEANED_CSV="clean_output/parsed_$INPUT_NAME_1.csv"

echo "--- Step 1: Creating MMseqs databases ---"
mmseqs createdb "$TRAIN_FASTA" "$DB1" 
mmseqs createdb "$TEST_FASTA" "$DB2" 

echo "--- Step 2: Searching for leakage (Identity > 0.25) ---"
# Searching DB1 (Query/Train) against DB2 (Target/Test)
mmseqs search "$DB1" "$DB2" "$RES_DB" "$INTERNAL_TMP" \
    --threads 2 -s 9 --min-seq-id 0.25

echo "--- Step 3: Converting results ---"
# Format output to tab-separated matches
mmseqs convertalis "$DB1" "$DB2" "$RES_DB" "$MATCHES_FILE"

echo "--- Step 4: Extracting Leaky IDs ---"
if [ -s "$MATCHES_FILE" ]; then
    # 1. Get column 1 (Query ID / Train ID)
    # 2. Sort and unique
    # 3. Remove Windows newlines (\r) just in case
    awk '{print $1}' "$MATCHES_FILE" | sort -u | tr -d '\r' > "$BLOCKLIST"

    NUM_LEAKS=$(wc -l < "$BLOCKLIST")
    echo "Found $NUM_LEAKS leaky sequences."
else
    echo "No matches found! (Clean split)"
    > "$BLOCKLIST"
fi

echo "--- Step 5: Filtering CSV ---"
if [ -s "$BLOCKLIST" ]; then
    # -F: Fixed string match (fast)
    # -v: Invert match (keep lines that DON'T match)
    # -f: Read patterns from file
    # -w: Match whole words only (CRITICAL FIX)
    grep -F -w -v -f "$BLOCKLIST" "$TRAIN_CSV" > "$CLEANED_CSV"

    # Verify the count
    ORIG_COUNT=$(grep -c "^" "$TRAIN_CSV" || true)
    NEW_COUNT=$(grep -c "^" "$CLEANED_CSV" || true)
    echo "Lines removed: $((ORIG_COUNT - NEW_COUNT))"
    echo "Saved cleaned dataset to: $CLEANED_CSV"
else
    cp "$TRAIN_CSV" "$CLEANED_CSV"
    echo "No sequences removed. Copy saved to: $CLEANED_CSV"
fi

# Cleanup
rm -rf "$TMP_DIR"