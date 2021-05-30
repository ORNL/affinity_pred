import torch
import logging

import transformers
from transformers import AutoModelForSequenceClassification, BertModel, RobertaModel, BertTokenizer, RobertaTokenizer
from transformers import PreTrainedModel, BertConfig, RobertaConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from transformers import EvalPrediction

from transformers import AutoModelForMaskedLM
from transformers import AdamW

from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process

from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
import deepspeed
from sparse_self_attention import BertSparseSelfAttention

from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn import functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import re
import gc
import os
import json
import pandas as pd
import numpy as np
import requests
from tqdm.auto import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DD

logger = logging.getLogger(__name__)

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))

seq_model_name = "Rostlab/prot_bert_bfd" # for fine-tuning

# this logic is necessary because online-downloading and caching doesn't seem to work
if os.path.exists('seq_tokenizer'):
    seq_tokenizer = BertTokenizer.from_pretrained('seq_tokenizer/', do_lower_case=False)
else:
    seq_tokenizer = BertTokenizer.from_pretrained(seq_model_name, do_lower_case=False)
    seq_tokenizer.save_pretrained('seq_tokenizer/')

model_directory = '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/model'
tokenizer_directory =  '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/tokenizer'
tokenizer_config = json.load(open(tokenizer_directory+'/config.json','r'))

smiles_tokenizer =  BertTokenizer.from_pretrained(tokenizer_directory, **tokenizer_config)
max_smiles_length = min(200,BertConfig.from_pretrained(model_directory).max_position_embeddings)
max_seq_length = min(4096,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,ninput):
        super().__init__()
        self.layers = torch.nn.Sequential(
               torch.nn.Linear(ninput, 32),
               torch.nn.ReLU(),
               torch.nn.Linear(32, 32),
               torch.nn.ReLU(),
               torch.nn.Linear(32, 1)
#        torch.nn.Linear(ninput, 1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class EnsembleSequenceRegressor(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, *args, **kwargs):
        super().__init__()

        # enable gradient checkpointing
        seq_config = BertConfig.from_pretrained(seq_model_name)
        seq_config.gradient_checkpointing=True
        self.seq_model = BertModel.from_pretrained(seq_model_name,config=seq_config)
        smiles_config = BertConfig.from_pretrained(smiles_model_name)
        smiles_config.gradient_checkpointing=True
        self.smiles_model = BertModel.from_pretrained(smiles_model_name,config=smiles_config)

        # for deepspeed stage 3 (to estimate buffer sizes)
        self.config = BertConfig(hidden_size = self.seq_model.config.hidden_size + self.smiles_model.config.hidden_size)

        self.sparsity_config = None
        try:
            from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
            self.sparsity_config = STConfig(num_heads=self.seq_model.config.num_attention_heads)
        except:
            pass

        if self.sparsity_config is not None:
            # replace the self attention layer of the sequence model
            from deepspeed.ops.sparse_attention import SparseAttentionUtils
            self.sparse_attention_utils = SparseAttentionUtils

            config = seq_config
            sparsity_config = self.sparsity_config
            layers = self.seq_model.encoder.layer

            for layer in layers:
                deepspeed_sparse_self_attn = BertSparseSelfAttention(
                    config=config,
                    sparsity_config=sparsity_config,
                    max_seq_length=max_seq_length)
                deepspeed_sparse_self_attn.query = layer.attention.self.query
                deepspeed_sparse_self_attn.key = layer.attention.self.key
                deepspeed_sparse_self_attn.value = layer.attention.self.value

                layer.attention.self = deepspeed_sparse_self_attn

            self.pad_token_id = seq_config.pad_token_id if hasattr(
                seq_config, 'pad_token_id') and seq_config.pad_token_id is not None else 0

        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.cls = MLP(seq_config.hidden_size+smiles_config.hidden_size)
        else:
            self.cls = MLP(seq_config.hidden_size+smiles_config.hidden_size)

    def pad_to_block_size(self,
                          block_size,
                          input_ids,
                          attention_mask,
                          pad_token_id):
        batch_size, seq_len = input_ids.shape

        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            # pad attention mask without attention on the padding tokens
            attention_mask = F.pad(attention_mask, (0, pad_len), value=False)

        return pad_len, input_ids, attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = []
        input_ids_1 = input_ids[:,:max_seq_length]
        attention_mask_1 = attention_mask[:,:max_seq_length]

        # sequence model with sparse attention
        input_shape = input_ids_1.size()
        device = input_ids_1.device
        extended_attention_mask: torch.Tensor = self.seq_model.get_extended_attention_mask(attention_mask_1, input_shape, device)
        if self.seq_model.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.seq_model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.sparsity_config is not None:
            pad_len_1, input_ids_1, attention_mask_1 = self.pad_to_block_size(
                block_size=self.sparsity_config.block,
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                pad_token_id=self.pad_token_id)

        embedding_output = self.seq_model.embeddings(
                    input_ids=input_ids_1
                )
        encoder_outputs = self.seq_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
            )
        sequence_output = encoder_outputs[0]

        if self.sparsity_config is not None and pad_len_1 > 0:
            sequence_output = self.sparse_attention_utils.unpad_sequence_output(
                pad_len_1, sequence_output)

        pooled_output = self.seq_model.pooler(sequence_output) if self.seq_model.pooler is not None else None
        outputs.append((sequence_output, pooled_output))

        # smiles model with full attention
        input_ids_2 = input_ids[:,max_seq_length:]
        attention_mask_2 = attention_mask[:,max_seq_length:]
        outputs.append(self.smiles_model(input_ids=input_ids_2,
                                         attention_mask=attention_mask_2,
                                         return_dict=False))

        # output is a tuple (hidden_state, pooled_output)
        last_hidden_states = torch.cat([output[1] for output in outputs], dim=1)

        logits = self.cls(last_hidden_states).squeeze(-1)

        if labels is not None:
            # crossentropyloss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).half())
            return (loss, logits)
        else:
            return output


def expand_seqs(seqs):
    input_fixed = ["".join(seq.split()) for seq in seqs]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    return [list(seq) for seq in input_fixed]

# on-the-fly tokenization
def encode(item):
        seq_encodings = seq_tokenizer(expand_seqs(item['seq'])[0],
                                     is_split_into_words=True,
                                     return_offsets_mapping=False,
                                     truncation=True,
                                     padding='max_length',
                                     add_special_tokens=True,
                                     max_length=max_seq_length)

        smiles_encodings = smiles_tokenizer(item['smiles'][0],
                                            padding='max_length',
                                            max_length=max_smiles_length,
                                            add_special_tokens=True,
                                            truncation=True)

        item['input_ids'] = [torch.cat([torch.tensor(seq_encodings['input_ids']),
                                        torch.tensor(smiles_encodings['input_ids'])])]
        item['attention_mask'] = [torch.cat([torch.tensor(seq_encodings['attention_mask']),
                                            torch.tensor(smiles_encodings['attention_mask'])])]
        return item

class AffinityDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        #affinity = item['neg_log10_affinity_M']
        affinity = float(item['affinity'])
        item['labels'] = float(affinity)

        # drop the non-encoded input
        item.pop('smiles')
        item.pop('seq')
        item.pop('neg_log10_affinity_M')
        item.pop('affinity')
        return item

    def __len__(self):
        return len(self.dataset)

data_all = load_dataset("jglaser/binding_affinity",split='train')

f = 0.9
split = data_all.train_test_split(train_size=f)
train = split['train']
validation = split['test']
train.set_transform(encode)
validation.set_transform(encode)


train_dataset = AffinityDataset(train)
val_dataset = AffinityDataset(validation)

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = p.predictions, p.label_ids

    return {
        "mse": mean_squared_error(out_label_list, preds_list),
        "mae": mean_absolute_error(out_label_list, preds_list),
    }


def model_init():
    return EnsembleSequenceRegressor(seq_model_name, model_directory)

def main():
    # also handles --deepspeed
    parser = HfArgumentParser(TrainingArguments)

    (training_args,) = parser.parse_args_into_dataclasses()
    print(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process local rank: %s, world size: %d, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.world_size,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == transformers.training_args.ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    if training_args.local_rank == 0:
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    trainer = Trainer(
        model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                   # training arguments, defined above
        train_dataset=train_dataset,          # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics = compute_metrics,    # evaluation metric
    )

    all_metrics = {}
    logger.info("*** Train ***")
    train_result = trainer.train()
    trainer.save_model('ensemble_model_'+str(dist.get_world_size()))  # this also saves the tokenizer
    metrics = train_result.metrics

    if trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        tokenizer.save_pretrained(training_args.output_dir)
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics

if __name__ == "__main__":
    main()
