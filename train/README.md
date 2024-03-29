*Prequisites*

- mpi4py
- transformers 4.6.1

```
# with deepspeed, the system cuda version should be the same as the one in the conda environment
module load cuda/10.2
```

*Training*

Before running on the compute nodes, download the models to the cache by
beginning training on a *single* login node GPUs (to avoid a race condition when
writing the files). Subsequently, the compute nodes can use the cached copy of the model
and the datasets.

Note: ZERO stage 2 appears much faster than stage 3 with many GPUs, potentially
because of less communication overhead

```
# cache dir
export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build

# load cuda compatible with conda installation
module load cuda/10.2

# this might crash with out of memory
CUDA_VISIBLE_DEVICES=1 deepspeed ../affinity_pred/finetune.py \
--deepspeed='ds_config_stage2.json' \
--output_dir='./results' \
--num_train_epochs=20 \
--fp16 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--warmup_steps=0 \
--learning_rate=3e-05 \
--weight_decay=3e-7 \
--logging_dir='./logs' \
--logging_steps=1 \
--evaluation_strategy="epoch" \
--gradient_accumulation_steps=1 \
--run_name="seq_smiles_affinity" \
--seed=42

# remove stale lock files
rm -f `find -name *lock`
```
