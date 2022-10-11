from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class CrossAttentionLayer(nn.Module):
    def __init__(self, config, other_config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertAttention(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            past_key_value=None
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class EnsembleSequenceRegressor(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length=None, sparse_attention=False,
                 output_attentions=False, n_cross_attention_layers=3):
        super().__init__()

        # enable gradient checkpointing
        seq_config = BertConfig.from_pretrained(seq_model_name)
        seq_config.gradient_checkpointing=True
        self.seq_model = BertModel.from_pretrained(seq_model_name,config=seq_config)

        smiles_config = BertConfig.from_pretrained(smiles_model_name)
        smiles_config.gradient_checkpointing=True
        self.smiles_model = BertModel.from_pretrained(smiles_model_name,config=smiles_config)

        self.max_seq_length = max_seq_length

        # for deepspeed stage 3 (to estimate buffer sizes)
        self.config = BertConfig(hidden_size = self.seq_model.config.hidden_size + self.smiles_model.config.hidden_size)

        self.sparsity_config = None
        if sparse_attention:
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

            from affinity_pred.sparse_self_attention import BertSparseSelfAttention

            for layer in layers:
                deepspeed_sparse_self_attn = BertSparseSelfAttention(
                    config=config,
                    sparsity_config=sparsity_config,
                    max_seq_length=self.max_seq_length)
                deepspeed_sparse_self_attn.query = layer.attention.self.query
                deepspeed_sparse_self_attn.key = layer.attention.self.key
                deepspeed_sparse_self_attn.value = layer.attention.self.value

                layer.attention.self = deepspeed_sparse_self_attn

            self.pad_token_id = seq_config.pad_token_id if hasattr(
                seq_config, 'pad_token_id') and seq_config.pad_token_id is not None else 0

        # Cross-attention layers
        self.n_cross_attention_layers = n_cross_attention_layers
        self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(config=seq_config,other_config=smiles_config) for _ in range(n_cross_attention_layers)])
        self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(config=smiles_config,other_config=seq_config) for _ in range(n_cross_attention_layers)])

        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.linear = torch.nn.Linear(seq_config.hidden_size+smiles_config.hidden_size,1)
        else:
            self.linear = torch.nn.Linear(seq_config.hidden_size+smiles_config.hidden_size,1)

        self.output_attentions=output_attentions

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
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
    ):
        outputs = []

        if sparse_attention:
            input_ids_1 = input_ids[:,:self.max_seq_length]
            attention_mask_1 = attention_mask[:,:self.max_seq_length]

        # sequence model with sparse attention
        input_shape = input_ids_1.size()
        device = input_ids_1.device
        extended_attention_mask_1: torch.Tensor = self.seq_model.get_extended_attention_mask(attention_mask_1, input_shape, device)

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
            attention_mask=extended_attention_mask_1,
            head_mask=head_mask
            )
        sequence_output = encoder_outputs[0]

        if self.sparsity_config is not None and pad_len_1 > 0:
            sequence_output = self.sparse_attention_utils.unpad_sequence_output(
                pad_len_1, sequence_output)

        # smiles model with full attention

        if sparse_attention:
            input_ids_2 = input_ids[:,self.max_seq_length:]
            attention_mask_2 = attention_mask[:,self.max_seq_length:]

        input_shape = input_ids_2.size()

        encoder_outputs = self.smiles_model(input_ids=input_ids_2,
                                         attention_mask=attention_mask_2,
                                         return_dict=False)
        smiles_output = encoder_outputs[0]

        # cross attentions
        if self.output_attentions:
            output_attentions = True

        # cross-attention masks
        cross_attention_mask_1 = self.seq_model.invert_attention_mask(
            attention_mask_1[:,None,:]*attention_mask_2[:,:,None])
        cross_attention_mask_2 = self.smiles_model.invert_attention_mask(
            attention_mask_2[:,None,:]*attention_mask_1[:,:,None])

        hidden_seq = sequence_output
        hidden_smiles = smiles_output

        for i in range(self.n_cross_attention_layers):
            attention_output_1 = self.cross_attention_seq[i](
                hidden_states=hidden_seq,
                attention_mask=attention_mask_1,
                encoder_hidden_states=hidden_smiles,
                encoder_attention_mask=cross_attention_mask_2,
                output_attentions=output_attentions)

            attention_output_2 = self.cross_attention_smiles[i](
                hidden_states=hidden_smiles,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_seq,
                encoder_attention_mask=cross_attention_mask_1,
                output_attentions=output_attentions)

            hidden_seq = attention_output_1[0]
            hidden_smiles = attention_output_2[0]

        mean_seq = torch.mean(hidden_seq,axis=1)
        mean_smiles = torch.mean(hidden_smiles,axis=1)
        last_hidden_states = torch.cat([mean_seq, mean_smiles], dim=1)

        if output_attentions:
            attentions_seq = attention_output_1[1]
            attentions_smiles = attention_output_2[1]

        logits = self.linear(last_hidden_states).squeeze(-1)

        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).half())
            return (loss, logits)
        else:
            if output_attentions:
                return logits, (attentions_seq, attentions_smiles)
            else:
                return logits
