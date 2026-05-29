# Model Finetuning and Evaluation Pipeline

This repository contains the pipeline for fine-tuning machine learning models and evaluating their performances based on configurations managed via a CSV job queue.

---

## 📂 Repository Structure

* `src/`: Core source code directory.
* `results_finetuning/`: Directory where fine-tuned models, evaluation metrics, and performance results are saved.
* `train_jobs.csv`: Configuration file containing the queue of training jobs and their hyperparameters.
* `train_finetuning.py`: The main script to execute the model fine-tuning process.
* `evaluation_finetuning.py`: Script to evaluate the fine-tuned models.

---

## ⚙️ Configuration (`train_jobs.csv`)

The fine-tuning pipeline reads its parameters sequentially from `train_jobs.csv`. Each row represents a training job with the following column structure:

| Column | Description |
| :--- | :--- |
| `model` | The architecture or base model to be trained. |
| `train_file` | Path to the training dataset. |
| `val_file` | Path to the validation dataset. |
| `test_file` | Path to the test dataset. |
| `dataset_type` | The format or type of the dataset being used. |
| `epochs` | Number of training passes over the dataset. |
| `lr` | Learning rate for the optimizer. |
| `wd` | Weight decay for regularization. |
| `batch_size` | Number of samples processed per training batch. |
| `gradient_batch`| Gradient accumulation steps. |
| `done` | Status flag (`0` or `1`/`True` or `False`) to track completed jobs. |

---

## 🚀 Getting Started

### 1. Running the Fine-Tuning Process
To start the training pipeline, run the following command. The script will automatically parse the parameters defined in `train_jobs.csv`:

```bash
python train_finetuning.py

### 2. Running the evaluation
To run the evalutation of the model, run the following command.

```bash
python evaluation_finetuning.py
