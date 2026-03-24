# ESM[TMP NAME]
[Temporary Description]  Protein Protein Interaction project

**Authors**: 

**Paper**: 


### Install 

To run the code be sure to have at least a version of python >= 3.10
``` 
python -m venv ppi
source ppi/bin/activate
pip install -r requirements.txt
```

### Project structure
- `data_preprocessing`: contains the script to run the necessary preprocessing to obtain the final csv necessary to train to finetune the models.
- `finetuning`: it contains the code to finetune the model, and inside the subfolder `data` the csv files already computed and used in the paper.
- `inference`: it contains the code to run the final models.