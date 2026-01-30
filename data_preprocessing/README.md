## [WIP]

The preprocessing is composed by the following pipeline
    
1) Download the original data
2) run MMSEQ for sequence similarity -> generate CSV files
3) Extract PDBs with ESM3
4) Run foldseek -> generate a list with PDBS which are in common and duplicates
5) Remove PDBS that are not correct

### Step 1: Download and preparation of the data 
**[WIP]**
### Step 2: Run MMSEQ

To run MMSEQ we have to setup the fasta files before with the script create_fasta.py, it will create a folder where the `<input_file>.csv` it's processed to `<input_file>.fasta` file.

```
python scripts/create_fasta.py <fasta_dir> <csv_dir> <train_filename> <test_filename>
```

After creating the `<output_dir>` with all the fasta files it is now possible to run MMSEQ with `mmseqs2_clean_leakage.sh`.

```
./scripts/mmseqs2_clean_leakage.sh  <fasta_dir> <csv_dir> <train_filename> <test_filename>
```

Note that train files are cleaned against zk488 and the validation files are validated against their respective training. **[WIP] BETTER DESCRIPTION** 

### Step3: Exraction of PDBS with ESM3