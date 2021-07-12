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
from transformers.trainer_utils import get_last_checkpoint

from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
import deepspeed

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

from model import EnsembleSequenceRegressor

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
        nhidden = 1000
        self.layers = torch.nn.Sequential(
               torch.nn.Linear(ninput, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, 1)
#        torch.nn.Linear(ninput, 1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

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

        smiles_encodings = smiles_tokenizer(item['smiles_can'][0],
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
        item.pop('smiles_can')
        item.pop('seq')
        item.pop('neg_log10_affinity_M')
        item.pop('affinity')
        return item

    def __len__(self):
        return len(self.dataset)

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = p.predictions[:,0], p.label_ids

    return {
        "mse": mean_squared_error(out_label_list, preds_list),
        "mae": mean_absolute_error(out_label_list, preds_list),
    }


def model_init():
    return EnsembleSequenceRegressor(seq_model_name, model_directory,  max_seq_length=max_seq_length,
                                     sparse_attention=True)

def main():
    # also handles --deepspeed
    parser = HfArgumentParser(TrainingArguments)

    (training_args,) = parser.parse_args_into_dataclasses()

    # seed the weight initialization
    torch.manual_seed(training_args.seed)

    # split the dataset
    data_all = load_dataset("jglaser/binding_affinity",split='train')

    # keep a small holdout data set
    split_test = data_all.train_test_split(train_size=0.99, seed=0)

    # further split the train set
    f = 0.9
    split = split_test['train'].train_test_split(train_size=f, seed=training_args.seed)
    train = split['train']
    validation = split['test']
    train.set_transform(encode)
    validation.set_transform(encode)

    train_dataset = AffinityDataset(train)
    val_dataset = AffinityDataset(validation)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(training_args.output_dir)  # this also saves the tokenizer
    metrics = train_result.metrics

    if trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics

if __name__ == "__main__":
    main()
