### Explain model for Mpro

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build

**fill cache**

CUDA_VISIBLE_DEVICES=1 deepspeed explain_mpro.py  --output_dir='tmp' --fp16

**With a compute allocation**

```
# does not work on 16 GB V100 (should work on 32GB though)

# (explanation does not work with ZERO v3)
OMP_NUM_THREADS=1 jsrun -r 1 -g 6 -a 6 -c 42 python explain_mpro.py \
    --deepspeed='../train/ds_config_stage2.json' \
    --output_dir='tmp/' \
    --fp16 \
    --per_device_eval_batch_size=1
```

