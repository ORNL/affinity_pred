#!/usr/bin/env bash
#BSUB -P BIP214
#BSUB -W 6:00
#BSUB -q batch
#BSUB -nnodes 32
#BSUB -J finetune_bert[1-5]
#BSUB -o finetune_bert.o%J
#BSUB -e finetune_bert.e%J

module load cuda/11.0.3
module load open-ce/1.2.0-py38-0
conda activate /gpfs/alpine/world-shared/bip214/deepspeed-env

export CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export HF_DATASETS_CACHE=/gpfs/alpine/world-shared/bip214/affinity_pred/train/dataset-cache
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build

# undo some conda env variables
export CC=`which gcc`
export GCC=`which gcc`
export CXX=`which g++`

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1

# clear stale lock files
rm -f `find -name *lock`

export ENSEMBLE_ID=${LSB_JOBINDEX}

jsrun -r 1 -g 6 -a 6 -c 42 python ../affinity_pred/finetune.py \
    --deepspeed='ds_config.json'\
    --output_dir=./results_bert_mse_${ENSEMBLE_ID}\
    --model_type='bert'\
    --num_train_epochs=45\
    --per_device_train_batch_size=64\
    --per_device_eval_batch_size=64\
    --warmup_steps=5\
    --learning_rate=3e-05\
    --weight_decay=3e-7\
    --logging_dir=./logs_bert_mse_${ENSEMBLE_ID}\
    --logging_steps=1\
    --evaluation_strategy="epoch"\
    --gradient_accumulation_steps=1\
    --fp16=True\
    --run_name="seq_smiles_affinity"\
    --save_strategy="epoch"\
    --seed==$((42+${ENSEMBLE_ID}))
