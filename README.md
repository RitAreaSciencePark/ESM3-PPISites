# ESM3-PPISites

## Decoding the Grammar of Protein–Protein Interaction Interfaces with Multimodal Representations

**Authors**: Yuri Gardinazzi, Edith Natalia Villegas Garcia, Sergio Senci, Davide Di Vora, Antonio
Feltrin, and Francesca Cuturello

**Paper**: [bioRxiv link](https://www.biorxiv.org/content/10.64898/2026.05.29.728739v1)

**doi**: https://doi.org/10.64898/2026.05.29.728739

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

### Prediction / Inference

To run predictions on your own protein sequences using the fine-tuned ESM3 model, you can use the `run_prediction.py` script. It performs residue-level classification to predict Protein-Protein Interaction (PPI) sites.

```
python run_prediction.py \
    --token "YOUR_HUGGINGFACE_TOKEN" \
    --input "path/to/input.csv" \
    --output "path/to/predictions.csv"
```

**Input Format**:
The input must be a valid CSV file containing at least the id and sequence columns. Additional columns are allowed and will be carried over to the output file.

```
complex,id,sequence
2AJF,1R42_W,STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKE...
```

**Output Format**:
The generated output file will contain all the original columns from your input file, appended with two new ones:

- probabilities: A space-separated list of float values representing the prediction probability (between 0 and 1) for each individual residue.

- prediction: A continuous binary string composed of 0s and 1s representing the final classification for each residue (based on a probability threshold of > 0.70).

#### Predictions for haddock
Run `pair_predictions.py` in the same way as run_prediction. it will do the inference with ESM3 to generate the PDBS and inside the output file there will be the field **hotspot_pred** with the indexes of the one predictions.
