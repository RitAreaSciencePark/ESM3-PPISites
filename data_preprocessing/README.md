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

To extract the PDBS from a given csv to a target output folder run the command
```
python scripts/extraction.py <csv_path>.csv <target_folder>
```

### Step4 

Now run foldseek with validations datasets against training, and training dataset against ZK. 
Results are then saved in FolderA_vs_FolderB.m8 file.

```
foldseek easy-search FolderA FolderB  FolderA_vs_FolderB.m8 tmp/ \
    --format-output query,target,alntmscore,qtmscore,ttmscore,evalue \
    -s 9.5
```

After the extraction with ESM3 with the following script copy in a target folder the proteins which are not similary in the .m8 file
These proteins have qtmscore < 0.5 and evalue > 1e-3
```
python scripts/filter_foldseek.py <pdb_folder> <foldseek_file>.m8  <target_folder>
```