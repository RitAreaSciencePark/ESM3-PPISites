### Structure Similarity

With this step we filter the dataset by looking at the structure similarity between the proteins of the different datasets. To do that we use Foldseek [link to install it](https://github.com/steineggerlab/foldseek)

#### Step 1
With the following command-line instruction do a structural alignment between two folders with PDBs
```
foldseek easy-search  folderA/ folderB/ output-file.m8 tmp/ --format-output query,target,alntmscore,qtmscore,ttmscore,evalue -s 9.5
```
With this python script get a csv with a list of PDBs from the folderA that are in common with the folderB which are present in the **output-file.m8**

```
scripts/extract_commons.py output-file.m8 output-commons.csv
```