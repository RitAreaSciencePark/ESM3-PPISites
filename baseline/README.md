# Baseline Model for Protein-Protein Interaction Prediction

This directory contains the baseline model for protein-protein interaction (PPI) site prediction. The baseline is a simple linear layer trained on pre-computed protein embeddings.

## Data Setup

Before training, you must download the embeddings from Zenodo (33 GB):

1. Download `esm3_data.zip` from Zenodo: [Link to be specified]
2. Extract it inside the `baseline/` directory:
   ```bash
   cd baseline/
   wget ...
   unzip esm3_data.zip
   ```

This will create the `esm3_data/` subdirectory with all embeddings and CSV files organized as shown in the directory structure below.

## Directory Structure

Once you download and extract the Zenodo data, the directory structure will be:

```
baseline/
├── train_baseline.py              # Training script
├── evaluation_baseline.py         # Evaluation script
├── models_baseline/               # Trained model checkpoints
│   └── {model_name}_{dataset_name}.pt
├── results_baseline/              # Evaluation results
│   ├── {model_name}_{dataset_name}_evaluation_results.csv
│   └── performance.csv
├── esm3_data/                     # Downloaded embeddings 
│   ├── big_model_data/            # Big model embeddings (for training)
│   │   ├── BioLiP-3693_train.pt
│   │   ├── BioLiP-3693_val.pt
│   │   ├── BioLiP-3693_test.pt
│   │   ├── PDBbind-1409_train.pt
│   │   ├── PDBbind-1409_val.pt
│   │   ├── PDBbind-1409_test.pt
│   │   └── zk448_test.pt
│   ├── small_model_data/          # Small model embeddings (for training)
│   │   ├── BioLiP-3693_train.pt
│   │   ├── BioLiP-3693_val.pt
│   │   ├── BioLiP-3693_test.pt
│   │   ├── PDBbind-1409_train.pt
│   │   ├── PDBbind-1409_val.pt
│   │   ├── PDBbind-1409_test.pt
│   │   └── zk448_test.pt
│   └── csv_data/
│       ├── BioLiP-3693_train.csv
│       ├── BioLiP-3693_val.csv
│       ├── BioLiP-3693_test.csv
│       ├── PDBbind-1409_train.csv
│       ├── PDBbind-1409_val.csv
│       ├── PDBbind-1409_test.csv
│       └── zk448_test.csv
├── .gitignore
└── README.md                      # This file
```

## Training

### Run Training

Train on BioLiP-3693 dataset with big model embeddings:
```bash
python train_baseline.py --model_name big_model --dataset_name BioLiP-3693
```

Train on PDBbind dataset with small model embeddings:
```bash
python train_baseline.py --model_name small_model --dataset_name PDBbind-1409
```

### Training Hyperparameters

The following hyperparameters can be modified in `train_baseline.py`:
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH`: Batch size (default: 96)
- `lr`: Learning rate (default: 5e-4)
- `THRESHOLD`: Classification threshold for hotspot prediction (default: 0.7)
- `PATIENCE`: Early stopping patience (default: 5 epochs)
- `MIN_DELTA`: Minimum loss improvement for early stopping (default: 1e-4)

### Supported Configurations

**Models:**
- `big_model`: Large pre-trained embedding model
- `small_model`: Small pre-trained embedding model

**Datasets:**
- `BioLiP-3693`: BioLiP database with 3693 proteins
- `PDBbind-1409`: PDBbind database with 1409 proteins

## Evaluation

### Run Evaluation

Evaluate a trained model on the test set:

```bash
python evaluation_baseline.py --model_name big_model --dataset_name BioLiP-3693
```

This will:
1. Load the trained model from `models_baseline/` 
2. Load the test embeddings from `esm3_data/`
3. Run inference on the test set
4. Compute metrics: F1, MCC, AUC, precision, recall
5. Save results to `results_baseline/{model_name}_{dataset_name}_evaluation_results.csv`
6. Save performance summary to `results_baseline/performance.csv`

### Output Files

- **Evaluation results CSV**: Contains per-protein predictions and metrics
- **Performance CSV**: Contains aggregate performance metrics across all test proteins

## Model Outputs

### Training Output
After training completes, the model checkpoint is saved to:
```
models_baseline/{model_name}_{dataset_name}.pt
```

The checkpoint contains:
- Model state dictionary (weights and biases)
- Optimizer state (for resuming training)
- Training epoch and final loss

### Prediction Output
Predictions are binary classifications:
- **Hotspot sites**: Residues predicted as interface (probability > threshold)
- **Probability scores**: Sigmoid output for each residue


