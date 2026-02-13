### Structure Similarity

With this step we filter the dataset by looking at the structure similarity between the proteins of the different datasets. To do that we use Foldseek [link to install it](https://github.com/steineggerlab/foldseek)

#### Step 1
With the following command-line instruction do a structural alignment between two folders with PDBs
```
foldseek easy-search  folderA/ folderB/ output-file.m8 tmp/ --format-output query,target,alntmscore,qtmscore,ttmscore,evalue -s 9.5
```
After the extraction with ESM3 with the following script copy in a target folder the proteins which are not similary in the .m8 file
These proteins have qtmscore < 0.5 and evalue > 1e-3
```
```