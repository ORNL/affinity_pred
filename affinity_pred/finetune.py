import torch
import logging

import transformers
from transformers import BertModel, BertTokenizer, AutoTokenizer
from transformers import PreTrainedModel, BertConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from transformers import EvalPrediction

from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex

from dataclasses import dataclass, field
from enum import Enum


from transformers import AdamW

from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import get_last_checkpoint

from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
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

max_seq_length = min(512,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

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
    return EnsembleSequenceRegressor(seq_model_name, smiles_model_directory,  max_seq_length=max_seq_length,
                                     sparse_attention=False)

@dataclass
class ModelArguments:
    model_type: str = field(
        default='bert',
        metadata = {'choices': ['bert','regex']},
    )


def main():
    global smiles_tokenizer, max_smiles_length, smiles_model_directory

    # also handles --deepspeed
    parser = HfArgumentParser([TrainingArguments,ModelArguments])

    (training_args, model_args) = parser.parse_args_into_dataclasses()

    # set up tokenizer and pre-trained model
    model_base_directory_dict = {
        "bert": ["/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean/",
                "/gpfs/alpine/world-shared/med106/gounley1/automatedmutations/pretraining/run/job_ikFsbI/output/"],
        "regex": ["/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean_regex/",
                  "/gpfs/alpine/world-shared/med106/blnchrd/automatedmutations/pretraining/run/job_86neeM/output/"]
    }

    smiles_tokenizer_directory = model_base_directory_dict[model_args.model_type][0] + 'tokenizer'
    smiles_model_directory = model_base_directory_dict[model_args.model_type][1]
    tokenizer_config = json.load(open(smiles_tokenizer_directory+'/config.json','r'))

    smiles_tokenizer =  AutoTokenizer.from_pretrained(smiles_tokenizer_directory, **tokenizer_config)

    if model_args.model_type == 'regex':
        smiles_tokenizer.backend_tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')])

    print('Tokenizer lower_case:', smiles_tokenizer.do_lower_case, flush=True)

    max_smiles_length = min(200,BertConfig.from_pretrained(smiles_model_directory).max_position_embeddings)

    # seed the weight initialization
    torch.manual_seed(training_args.seed)

    # split the dataset
    #data_all = load_dataset("jglaser/binding_affinity",split='train')
    data_all = load_dataset('parquet',data_files='/gpfs/alpine/world-shared/bip214/binding_affinity/data/all_ic50.parquet')['train']

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
