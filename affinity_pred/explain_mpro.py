import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer, BertConfig, Trainer
from transformers import EvalPrediction
from transformers import TrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled

import pandas as pd
import numpy as np
import json
import re
import os

from model import EnsembleSequenceRegressor
from explain import EnsembleExplainer

from transformers import HfArgumentParser

# this logic is necessary because online-downloading and caching doesn't seem to work
seq_model_name = "Rostlab/prot_bert_bfd" # for fine-tuning
if os.path.exists('seq_tokenizer'):
    seq_tokenizer = BertTokenizer.from_pretrained('seq_tokenizer/', do_lower_case=False)
else:
    seq_tokenizer = BertTokenizer.from_pretrained(seq_model_name, do_lower_case=False)
    seq_tokenizer.save_pretrained('seq_tokenizer/')

smiles_model_directory = '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/model'
tokenizer_directory =  '/gpfs/alpine/world-shared/bip214/maskedevolution/models/bert_large_1B/tokenizer'
tokenizer_config = json.load(open(tokenizer_directory+'/config.json','r'))
smiles_tokenizer =  BertTokenizer.from_pretrained(tokenizer_directory, **tokenizer_config)

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
        item['token_type_ids'] = [torch.cat([torch.tensor(seq_encodings['token_type_ids']),
                                        torch.tensor(smiles_encodings['token_type_ids'])])]
        item['attention_mask'] = [torch.cat([torch.tensor(seq_encodings['attention_mask']),
                                            torch.tensor(smiles_encodings['attention_mask'])])]
        return item

max_smiles_length = min(200,BertConfig.from_pretrained(smiles_model_directory).max_position_embeddings)
max_seq_length = min(4096,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

class PosteraDataset(Dataset):
    def __init__(self, df, mean, var, seq):
        self.df = df
        self.mean = mean
        self.var = var
        self.seq = seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {'smiles': [row.SMILES], 'seq': [self.seq]}
        item = encode(item)

        # get first (single) item
        item['input_ids'] = item['input_ids'][0]
        item['token_type_ids'] = item['token_type_ids'][0]
        item['attention_mask'] = item['attention_mask'][0]

        affinity = 6-np.log(row['f_avg_IC50'])/np.log(10)
        affinity = (affinity-self.mean)/np.sqrt(self.var)
        item['labels'] = float(affinity)
        # drop the non-encoded input
        item.pop('smiles')
        item.pop('seq')
        return item

    def __len__(self):
        return len(self.df)

def main():
    import deepspeed

    # handle --deepspeed argument
    parser = HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()

    model = EnsembleSequenceRegressor(seq_model_name, smiles_model_directory, max_seq_length=max_seq_length)

    fname = '../train/ensemble_model_4608/pytorch_model.bin'
    checkpoint = torch.load(fname, map_location="cpu")

    error_msgs = []
    def load(module: torch.nn.Module, prefix=""):
        # because zero3 puts placeholders in model params, this context
        # manager gathers (unpartitions) the params of the current layer, then loads from
        # the state dict and then re-partitions them again
        args = (checkpoint, '', {}, True, [], [], error_msgs)
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix="")
    device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    model.to(device)

    # un-normalize predictions [to -log_10 affinity[M] units]
    mean, var = (6.49685099, 2.43570803)
    def scale(x):
        return x*np.sqrt(var)+mean

    mpro_seq_5r84 = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"

    df = pd.read_csv('data/activity_data.csv')
    df = df[~df['f_avg_IC50'].isnull()].reset_index(drop=True)

    postera_dataset = PosteraDataset(df=df, mean=mean, var=var, seq=mpro_seq_5r84)

    # explain the predictions
    explainer = EnsembleExplainer(model,
                                  seq_tokenizer=seq_tokenizer,
                                  smiles_tokenizer=smiles_tokenizer,
                                  internal_batch_size=training_args.per_device_eval_batch_size)

    loader = torch.utils.data.DataLoader(postera_dataset)

    item = next(iter(loader))
    input_ids = item['input_ids'].to(device)
    attention_mask = item['attention_mask'].to(device)

    print(explainer(input_ids,attention_mask))

if __name__ == "__main__":
    main()


