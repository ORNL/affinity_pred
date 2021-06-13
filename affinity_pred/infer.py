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
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import get_last_checkpoint

from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
import deepspeed

from torch.nn import functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dask
import dask.dataframe as dd
from dask.distributed import Client

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

# Mpro has 203 residues
max_seq_length = min(256,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

def expand_seqs(seqs):
    input_fixed = ["".join(seq.split()) for seq in seqs]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    return [list(seq) for seq in input_fixed]

def main():
    # also handles --deepspeed
    parser = HfArgumentParser(TrainingArguments)

    (training_args,) = parser.parse_args_into_dataclasses()

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

    def transform(smiles_canonical):
        # 5R84
        seq = 'SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ'
        item = {'seq': [seq], 'smiles': [smiles_canonical]}
        return encode_canonical(item)

    def transform_df(df):
        return df['smiles_can'].apply(transform)

    # load the model and predict a batch
    def predict(df, return_dict=False):
        from affinity_pred.model import EnsembleSequenceRegressor

        def model_init():
            return EnsembleSequenceRegressor(seq_model_name, model_directory,  max_seq_length=max_seq_length, sparse_attention=False)

        trainer = Trainer(
            model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
            args=training_args,                   # training arguments, defined above
        )

        checkpoint = torch.load('../train/ensemble_model_4608/pytorch_model.bin')
        trainer.model.load_state_dict(checkpoint,strict=False)

        x = transform_df(df)
        pred = trainer.predict(x)
        df['affinity'] = pd.Series(data=pred.predictions, index=df.index)
        if return_dict:
            return df, pred
        else:
            return df

    client = Client(scheduler_file='my-scheduler.json')
    ddf = dd.read_parquet('/gpfs/alpine/world-shared/bip214/Enamine_SMILES_canonical')

    df = ddf.head(1000)
    meta, out = client.submit(predict,df,return_dict=True).result()
    print('{} samples/second'.format(out.metrics['test_samples_per_second']))

    ddf_pred = ddf.map_partitions(predict,meta=meta)
    ddf_pred.to_parquet('/gpfs/alpine/world-shared/bip214/Enamine_affinity')

if __name__ == "__main__":
    main()
