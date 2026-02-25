#!/bin/bash
#SBATCH --job-name=fine2
#SBATCH --partition=H100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_15_epochs_no_early_stopping_%j.out

cd $SLURM_SUBMIT_DIR

source /u/area/ygardinazzi/scratch/miniconda/bin/activate
conda activate bio



# --- ADD THIS LINE ---
echo "Bash check: PYTORCH_ALLOC_CONF is set to: $PYTORCH_ALLOC_CONF"
# -----------------------------
# CSV configuration
# -----------------------------
CSV_FILE="train_jobs.csv"
TMP_FILE="train_jobs_tmp.csv"

# CSV expected columns (UPDATED):
# model,train_file,val_file,test_file,dataset_type,epochs,lr,wd,batch_size,gradient_batch,done

# -----------------------------
# Iterate over each line in CSV
# -----------------------------
# tail skips header. read expects val_file now.
tail -n +2 "$CSV_FILE" | while IFS=, read -r model train_file val_file test_file dataset_type epochs lr wd batch_size gradient_batch done_flag
do
    # Skip commented or empty string lines
    if [[ "$model" == \#* ]] || [[ -z "$model" ]]; then
        continue
    fi

    # Skip completed jobs
    if [[ "$done_flag" == "1" ]]; then
        echo "⏩ Skipping $model (already done)"
        continue
    fi

    echo "🚀 Starting training for $model at $(date) "
    # --- DEBUGGING PRINTS ---
    echo "   [DEBUG] model: $model"
    echo "   [DEBUG] train_file: $train_file"
    echo "   [DEBUG] val_file: $val_file"
    echo "   [DEBUG] test_file: $test_file"
    echo "   [DEBUG] dataset: $dataset_type"
    echo "   [DEBUG] epochs: $epochs"
    echo "   [DEBUG] lr: $lr"
    # ------------------------

    OUTPUT_DIR="results/"  
    mkdir -p "$OUTPUT_DIR"

    # -----------------------------
    # Run Python training script
    # -----------------------------
    python -u training_main.py \
        --model_name "$model" \
        --train_file "$train_file" \
        --val_file "$val_file" \
        --test_file "$test_file" \
        --dataset_type "$dataset_type" \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs "$epochs" \
        --batch_size "$batch_size" \
        --gradient_batch "$gradient_batch" \
        --weight_decay "$wd" \
        --lr "$lr"

    # Check success and mark as done
    if [[ $? -eq 0 ]]; then
        echo "✅ Completed training for $model (lr=$lr, wd=$wd, bs=$batch_size, grad_bs=$gradient_batch) successfully at $(date) — marking as done"
    
        # Updated awk to handle the extra column (val_file is column 3)
        # Columns:
        # 1: model
        # 2: train_file
        # 3: val_file  <-- NEW
        # 4: test_file
        # 5: dataset_type
        # 6: epochs
        # 7: lr
        # 8: wd
        # 9: batch_size
        # 10: grad_batch
        # 11: done_flag
        awk -v model="$model" \
            -v train_file="$train_file" \
            -v val_file="$val_file" \
            -v test_file="$test_file" \
            -v dataset_type="$dataset_type" \
            -v epochs="$epochs" \
            -v lr="$lr" \
            -v wd="$wd" \
            -v batch_size="$batch_size" \
            -v grad_batch="$gradient_batch" \
            -F, 'BEGIN{OFS=","}
                NR==1 {print; next}
                $1==model && $2==train_file && $3==val_file && $4==test_file && $5==dataset_type && \
                $6==epochs && $7==lr && $8==wd && $9==batch_size && $10==grad_batch { $NF=1 }
                {print}
            ' "$CSV_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$CSV_FILE"
    
    else
        echo "❌ Training failed for $model (lr=$lr, wd=$wd, bs=$batch_size, grad_bs=$gradient_batch) — not marking as done"
    fi
    
    echo "----------------------------------------------"
done

echo "🎯 All runs completed at $(date)"