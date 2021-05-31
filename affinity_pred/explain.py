import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
from captum.attr import LayerIntegratedGradients

from model import EnsembleSequenceRegressor

import numpy as np

class EnsembleExplainer(object):
    def __init__(
        self,
        model: EnsembleSequenceRegressor,
        seq_tokenizer,
        smiles_tokenizer,
    ):
        """
        Args:
            model (EnsembleSequenceRegressor): Pretrained ensemble regressor
        Raises:
            AttributionTypeNotSupportedError:
        """
        self.seq_attributions = None
        self.smiles_attributions = None

        self.model = model

        self.seq_tokenizer = seq_tokenizer
        self.smiles_tokenizer = smiles_tokenizer

    def _calculate_attributions(  # type: ignore
        self,
        input_ids,
        attention_mask,
    ):

        ref_input_ids = np.array(input_ids.cpu().numpy())
        ref_input_ids[:,0:self.model.max_seq_length] = [[self.seq_tokenizer.cls_token_id] + \
            [self.seq_tokenizer.pad_token_id for token in seq[:self.model.max_seq_length] 
                if token != self.seq_tokenizer.cls_token_id
                and token != self.seq_tokenizer.sep_token_id] + \
            [self.seq_tokenizer.sep_token_id] for seq in input_ids.cpu().numpy()]

        ref_input_ids[self.model.max_seq_length:] = [[self.smiles_tokenizer.cls_token_id] + \
             [self.smiles_tokenizer.pad_token_id for token in seq[self.model.max_seq_length:]
                if token != self.smiles_tokenizer.cls_token_id
                and token != self.smiles_tokenizer.sep_token_id] + \
            [self.smiles_tokenizer.sep_token_id] for seq in input_ids.cpu().numpy()]

        ref_input_ids = torch.tensor(ref_input_ids)
        ref_input_ids = ref_input_ids.to(input_ids.device)

        lig = LayerIntegratedGradients(self.model.forward,
            [self.model.seq_model.get_input_embeddings(), self.model.smiles_model.get_input_embeddings()],
        )

        (seq_attributions, smiles_attributions), self.delta = lig.attribute(
                    inputs=input_ids,
                    baselines=ref_input_ids,
                    return_convergence_delta=True,
                    additional_forward_args=attention_mask,
                )
        self.seq_attributions = seq_attributions[:self.model.max_seq_length].sum(dim=-1).squeeze(0)
        self.seq_attributions = self.seq_attributions / torch.norm(self.seq_attributions)
        self.smiles_attributions = smiles_attributions[:self.model.max_seq_length].sum(dim=-1).squeeze(0)
        self.smiles_attributions = self.smiles_attributions / torch.norm(self.smiles_attributions)

    def __call__(
        self,
        input_ids,
        attention_mask 
    ):
        """
        Calculates attribution for `input_ids` using the model.
        This explainer also allows for attributions with respect to a particlar embedding type.
        Returns:
            tuple: (List of associated attribution scores for protein sequence, and for SMILES)
        """

        self._calculate_attributions(
            input_ids,
            attention_mask
        )
        return self.seq_attribtuions, self.smiles_attributions
