from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

import torch

import transformers
from transformers import AutoModelForSequenceClassification, BertModel, RobertaModel, BertTokenizerFast, RobertaTokenizer
from transformers import PreTrainedModel, BertConfig, RobertaConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from transformers import EvalPrediction

from transformers import AutoModelForMaskedLM
from transformers import AdamW

from transformers import HfArgumentParser
from dataclasses import dataclass, field


from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
import deepspeed

from torch.nn import functional as F

import toolz
import time
from functools import partial

import traceback

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd

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

seq_model_name = "Rostlab/prot_bert_bfd" # for fine-tuning

# this logic is necessary because online-downloading and caching doesn't seem to work
if os.path.exists('seq_tokenizer'):
    seq_tokenizer = BertTokenizerFast.from_pretrained('seq_tokenizer/', do_lower_case=False)
else:
    seq_tokenizer = BertTokenizerFast.from_pretrained(seq_model_name, do_lower_case=False)
    seq_tokenizer.save_pretrained('seq_tokenizer/')

model_directory = '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/model'
tokenizer_directory =  '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/tokenizer'
tokenizer_config = json.load(open(tokenizer_directory+'/config.json','r'))

smiles_tokenizer =  BertTokenizerFast.from_pretrained(tokenizer_directory, **tokenizer_config)
max_smiles_length = min(200,BertConfig.from_pretrained(model_directory).max_position_embeddings)

# Mpro has 306 residues
max_seq_length = min(4096,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

def expand_seqs(seqs):
    input_fixed = ["".join(seq.split()) for seq in seqs]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    return [list(seq) for seq in input_fixed]

# use distributed data parallel on a node-local basis for inference
#os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
#os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_LOCAL_SIZE']
#os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
#os.environ['MASTER_ADDR'] = '127.0.0.1'
#os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(29500+int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']))


#torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

@dataclass
class InferenceArguments:
    checkpoint: str = field(
        default=None
    )

    batch_size: int = field(
        default=1
    )

    input_path: str = field(
        default=None
    )

    output_path: str = field(
        default=None
    )

    seq: str = field(
        default=None
    )

    smiles_column: str = field(
        default='smiles'
    )

    seq_column: str = field(
        default='seq'
    )

#
# parser - used to handle deepspeed case as well
parser = HfArgumentParser([TrainingArguments,InferenceArguments])
training_args, inference_args = parser.parse_args_into_dataclasses()

def main(fn):
    try:
        def encode_canonical(item):
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
                item['input_ids'] = torch.cat([torch.tensor(seq_encodings['input_ids']),
                                                torch.tensor(smiles_encodings['input_ids'])])
                item['token_type_ids'] = torch.cat([torch.tensor(seq_encodings['token_type_ids']),
                                                torch.tensor(smiles_encodings['token_type_ids'])])
                item['attention_mask'] = torch.cat([torch.tensor(seq_encodings['attention_mask']),
                                                    torch.tensor(smiles_encodings['attention_mask'])])
                item.pop('smiles')
                item.pop('seq')
                return item

        def transform(seq, smiles_canonical):
            item = {'seq': [seq], 'smiles': [smiles_canonical]}
            return encode_canonical(item)

        def transform_df(df):
            if inference_args.seq is not None:
                return df[inference_args.smiles_column].apply(lambda x: transform(inference_args.seq, x)).values
            else:
                assert inference_args.seq is None
                return df[[inference_args.seq_column,inference_args.smiles_column]].apply(lambda x: transform(*x),axis=1).values

        # load the model and predict a batch
        def predict(df, return_dict=False):
            from affinity_pred.model import EnsembleSequenceRegressor

            def model_init():
                return EnsembleSequenceRegressor(seq_model_name, model_directory,  max_seq_length=max_seq_length, sparse_attention=True)

            trainer = Trainer(
                model_init=model_init,                # the instantiated <F0><9F><A4><97> Transformers model to be trained
                args=training_args,                   # training arguments, defined above
            )

            checkpoint = torch.load(inference_args.checkpoint,
                map_location=torch.device('cpu'))

            trainer.model.load_state_dict(checkpoint,strict=False)

            x = transform_df(df)
            out = trainer.predict(x)

            print('{} samples/second'.format(out.metrics['test_samples_per_second']))

            df['affinity_mean'] = pd.Series(data=out.predictions[:,0], index=df.index).astype('float32')
            df['affinity_var'] = pd.Series(data=out.predictions[:,1], index=df.index).astype('float32')
            return df

        df = pd.read_parquet(fn)

        df_pred = predict(df)

        base = os.path.basename(fn)
        df_pred.to_parquet(inference_args.output_path+'/'+base)
    except Exception as e:
        print(repr(e))
        traceback.print_exc()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            import glob
            fns = glob.glob(inference_args.input_path)
            fns = [f for f in fns if not os.path.exists(inference_args.output_path+'/'+os.path.basename(f))]

            executor.map(main, fns)
