# choose a port in a valid (non-blocked) range, instead of letting NCCL pick a random one
export NCCL_SOCKET_IFNAME=enP5p1s0f0
IP=`ip addr show ${NCCL_SOCKET_IFNAME} | grep -o "inet [0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"`
PORT=$((32768+${OMPI_COMM_WORLD_LOCAL_RANK}))
export NCCL_COMM_ID=${IP}:${PORT}

# command line
python ../affinity_pred/finetune.py \
    --deepspeed='ds_config_scale.json'\
    --output_dir='./results'\
    --num_train_epochs=20\
    --per_device_train_batch_size=8\
    --per_device_eval_batch_size=8\
    --warmup_steps=0\
    --learning_rate=3e-05\
    --weight_decay=3e-7\
    --logging_dir='./logs'\
    --logging_steps=1\
    --evaluation_strategy="epoch"\
    --gradient_accumulation_steps=1\
    --fp16=True\
    --run_name="seq_smiles_affinity"\
    --seed=42
