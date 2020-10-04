# coding=utf-8

"""Syntax-Augmented BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
from io import open

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_transformers import modeling_bert
from pytorch_transformers.modeling_utils import (PretrainedConfig, PreTrainedModel)

from model.graph_encoder import GNNRelationModel, pool
from model.crf import CRF
from model.loss import SequenceCriteriaCRF
from utils import constant
from model.tree import head_to_tree, tree_to_adj


logger = logging.getLogger(__name__)
ACT2FN = {"gelu": modeling_bert.gelu,
          "relu": torch.nn.functional.relu,
          "swish": modeling_bert.swish}


class SyntaxBertConfig(PretrainedConfig):
    pretrained_config_archive_map = modeling_bert.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 use_syntax=True,
                 syntax=None,
                 **kwargs):
        super(SyntaxBertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps

            # Syntax encoder layer parameter initializations
            self.use_syntax = use_syntax,
            self.syntax = syntax
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size,
                               self.all_head_size)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size)
        self.value = nn.Linear(config.hidden_size,
                               self.all_head_size)

    def forward(self, h, z, attention_mask, attention_probs_dropout_prob=0.1):
        query = self.query(h)
        key = self.key(h)
        value = self.value(h)
        return self.multi_head_attention(key,
                                         value,
                                         query,
                                         attention_mask,
                                         dropout_ratio=attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, key, value, query, attention_mask, dropout_ratio=0.1):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = F.dropout(attention_probs,
                                    p=dropout_ratio,
                                    training=self.training)
        context_layer = torch.matmul(attention_probs,
                                     value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class JointFusionAttention(BertSelfAttention):
    def __init__(self, config):
        super(JointFusionAttention, self).__init__(config)
        self.ukey = nn.Linear(config.syntax['hidden_size'],
                              self.all_head_size)
        self.uvalue = nn.Linear(config.syntax['hidden_size'],
                                self.all_head_size)

    def forward(self, h, z, attention_mask, attention_probs_dropout_prob=0.1):
        query = self.query(h)
        key = self.key(h) + self.ukey(z)
        value = self.value(h) + self.uvalue(z)
        return self.multi_head_attention(key,
                                         value,
                                         query,
                                         attention_mask,
                                         dropout_ratio=attention_probs_dropout_prob)


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.output = modeling_bert.BertSelfOutput(config)
        if config.model_type == 'm3':
            self.self = JointFusionAttention(config)
        else:
            self.self = BertSelfAttention(config)
        self.config = config
        self.pruned_heads = set()

    def forward(self, input_tensor, z, attention_mask):
        self_outputs = self.self(input_tensor,
                                 z,
                                 attention_mask)
        attention_output = self.output(self_outputs[0],
                                       input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = modeling_bert.BertIntermediate(config)
        self.output = modeling_bert.BertOutput(config)

    def forward(self, hidden_states, z, attention_mask):
        attention_outputs = self.attention(hidden_states,
                                           z,
                                           attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output,
                                   attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, z, attention_mask):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states,
                                         z,
                                         attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = SyntaxBertConfig
    pretrained_model_archive_map = modeling_bert.BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = modeling_bert.load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, modeling_bert.BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.config = config
        in_hidden_size = config.hidden_size
        if config.syntax['use_subj_obj']:
            in_hidden_size *= 3

        self.pool_type = self.config.syntax['pooling']

        # output MLP layers
        layers = [nn.Linear(in_hidden_size,
                            config.hidden_size),
                  nn.Tanh()]
        for _ in range(config.syntax['mlp_layers'] - 1):
            layers += [nn.Linear(config.hidden_size,
                                 config.hidden_size),
                       nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, hidden_states, token_mask=None,
                subj_pos=None, obj_pos=None):

        pool_mask = token_mask.eq(0).unsqueeze(2)
        h_out = pool(hidden_states,
                     pool_mask,
                     type=self.pool_type)

        if self.config.syntax['use_subj_obj']:
            subj_mask = subj_pos.eq(0).eq(0).unsqueeze(2)
            subj_out = pool(hidden_states,
                            subj_mask,
                            type=self.pool_type)

            obj_mask = obj_pos.eq(0).eq(0).unsqueeze(2)
            obj_out = pool(hidden_states,
                           obj_mask,
                           type=self.pool_type)

            h_out = torch.cat([h_out, subj_out, obj_out],
                              dim=1)

        pooled_output = self.out_mlp(h_out)
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config
        if config.model_type in {'late_fusion', 'joint_fusion'}:
            self.syntax_encoder = GNNRelationModel(config)
        self.embeddings = modeling_bert.BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(config)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings,
                                                      new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings

        if self.config.model_type not in {'bert_baseline', 'late_fusion'}:
            # Resizing token embeddings of the syntax-encoder
            self.syntax_encoder.resize_token_embeddings(new_num_tokens)
        return self.embeddings.word_embeddings

    def inputs_to_tree_reps(self, dep_head, seq_len, subj_pos, obj_pos, dep_rel, device):
        maxlen = max(seq_len)
        trees = [head_to_tree(dep_head[i],
                              seq_len[i],
                              self.config.syntax['prune_k'],
                              subj_pos[i],
                              obj_pos[i],
                              dep_rel[i]) for i in range(len(seq_len))]

        # Making "self_loop=True" as adj will be used as a masking matrix during graph attention
        adj_matrix_list, dep_rel_matrix_list = [], []
        for tree in trees:
            adj_matrix, dep_rel_matrix = tree_to_adj(maxlen,
                                                     tree,
                                                     directed=False,
                                                     self_loop=self.config.syntax['adj_self_loop'])
            adj_matrix = adj_matrix.reshape(1, maxlen, maxlen)
            adj_matrix_list.append(adj_matrix)

            dep_rel_matrix = dep_rel_matrix.reshape(1, maxlen, maxlen)
            dep_rel_matrix_list.append(dep_rel_matrix)

        batch_adj_matrix = torch.from_numpy(np.concatenate(adj_matrix_list,
                                                           axis=0))
        batch_dep_rel_matrix = torch.from_numpy(np.concatenate(dep_rel_matrix_list,
                                                               axis=0))
        return Variable(batch_adj_matrix.to(device)), \
               Variable(batch_dep_rel_matrix.to(device))

    def postprocess_attention_mask(self, mask):
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        return mask

    @classmethod
    def to_linguistic_space(cls, wp_tensor, wp_rows, align_sizes, wp_seq_lengths):
        device = wp_tensor.device
        new_tensor = []
        for i, (seq_len, size) in enumerate(zip(wp_seq_lengths, align_sizes)):
            wp_weighted = wp_tensor[i, :seq_len] / torch.FloatTensor(size).to(device).unsqueeze(1)
            new_row = []
            for j, word_piece_slice in enumerate(wp_rows[i]):
                tensor = torch.sum(wp_weighted[word_piece_slice],
                                   dim=0, keepdim=True)
                new_row.append(tensor)
            new_row = torch.cat(new_row)
            new_tensor.append(new_row)
        new_tensor = nn.utils.rnn.pad_sequence(new_tensor,
                                               batch_first=True)
        return new_tensor

    def forward(self, input_ids, token_type_ids=None,
                wp_token_mask=None, dep_head=None,
                dep_rel=None, wp_rows=None, align_sizes=None, seq_len=None,
                subj_pos=None, obj_pos=None, linguistic_token_mask=None):

        if wp_token_mask is None:
            wp_token_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if self.config.model_type in {"late_fusion", "joint_fusion"}:
            adj_matrix, dep_rel_matrix = self.inputs_to_tree_reps(dep_head,
                                                                  seq_len,
                                                                  subj_pos,
                                                                  obj_pos,
                                                                  dep_rel,
                                                                  input_ids.device)

        self_attention_mask = wp_token_mask[:, None, None, :]
        self_attention_mask = self.postprocess_attention_mask(self_attention_mask)

        if self.config.model_type == "joint_fusion":
            syntax_enc_outputs = self.syntax_encoder(input_ids,
                                                     adj_matrix,
                                                     dep_rel_matrix,
                                                     seq_len)
        else:
            syntax_enc_outputs = None

        # For all model types (late_fusion, joint_fusion, bert_baseline) the below two steps are same
        embedding_output = self.embeddings(input_ids,
                                           token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(embedding_output,
                                       syntax_enc_outputs,
                                       self_attention_mask)
        sequence_output = encoder_outputs[0]

        if self.config.model_type == "late_fusion":
            sequence_output = self.syntax_encoder(sequence_output,
                                                  adj_matrix,
                                                  dep_rel_matrix,
                                                  seq_len)

        # TODO: For token-level tasks like SRL or NER, pooled output doesn't matter but for sequence-level tasks
        # TODO: like Relation Extraction, it is important. Fix this hack so that the code works without
        # TODO: modification for both token-level and sequence-level tasks.
        pooled_output = self.pooler(sequence_output,
                                    wp_token_mask,
                                    subj_pos,
                                    obj_pos)
        # to linguistic space
        sequence_output = self.to_linguistic_space(sequence_output,
                                                   wp_rows,
                                                   align_sizes,
                                                   seq_len)
        outputs = (sequence_output, pooled_output)
        return outputs


class SyntaxBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(SyntaxBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size,
                                    self.config.num_labels)
        self.init_weights()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, wp_token_mask=None,
                labels=None, dep_head=None, dep_rel=None, wp_rows=None, align_sizes=None,
                seq_len=None, subj_pos=None, obj_pos=None, linguistic_token_mask=None,
                hidden_dropout_prob=0.1, **kwargs):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            wp_token_mask=wp_token_mask,
                            dep_head=dep_head,
                            dep_rel=dep_rel,
                            wp_rows=wp_rows,
                            align_sizes=align_sizes,
                            seq_len=seq_len,
                            subj_pos=subj_pos,
                            obj_pos=obj_pos,
                            linguistic_token_mask=linguistic_token_mask)
        pooled_output = outputs[1]
        pooled_output = F.dropout(pooled_output,
                                  p=hidden_dropout_prob,
                                  training=self.training)
        logits = self.classifier(pooled_output)

        loss = self.loss_func(logits.view(-1,
                                          self.config.num_labels),
                              labels.view(-1))
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        return {'loss': loss, 'predict': preds}


class SyntaxBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(SyntaxBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size,
                                    config.num_labels)
        self.loss_func = nn.CrossEntropyLoss()

        if config.crf:
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions = nn.Parameter(torch.randn(config.num_labels,
                                                        config.num_labels))
            nn.init.xavier_uniform_(self.transitions.data)
            self.crf = CRF(config.label_map,
                           config=config,
                           transitions=self.transitions)
            self.loss_func = SequenceCriteriaCRF(self.crf)
        self.config = config
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, wp_token_mask=None,
                labels=None, dep_head=None, dep_rel=None, wp_rows=None, align_sizes=None,
                seq_len=None, subj_pos=None, obj_pos=None, linguistic_token_mask=None,
                hidden_dropout_prob=0.1, **kwargs):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            wp_token_mask=wp_token_mask,
                            dep_head=dep_head,
                            dep_rel=dep_rel,
                            wp_rows=wp_rows,
                            align_sizes=align_sizes,
                            seq_len=seq_len,
                            subj_pos=subj_pos,
                            obj_pos=obj_pos,
                            linguistic_token_mask=linguistic_token_mask)
        sequence_output = outputs[0]

        sequence_output = F.dropout(sequence_output,
                                    p=hidden_dropout_prob,
                                    training=self.training)
        logits = self.classifier(sequence_output)

        if self.config.crf:
            l = [len(row) for row in wp_rows]
            batch_labels = nn.utils.rnn.pad_sequence(torch.split(labels, l),
                                                     batch_first=True,
                                                     padding_value=constant.OntoNotes_NER_LABEL_TO_ID['PAD_TOKEN'])
            loss = self.loss_func(logits, batch_labels, l)
            preds = []
            for feats, sequence_length in zip(logits, l):
                score, tag_seq = self.crf(feats[: sequence_length])
                preds.extend(torch.LongTensor(tag_seq).cpu().numpy())
            preds = torch.LongTensor(preds)
        else:
            # This attention mask acts as if the word tokens are in the linguistic space
            attn_mask = [torch.LongTensor([1]*len(row)).to(logits.device) for row in wp_rows]
            attn_mask = nn.utils.rnn.pad_sequence(attn_mask,
                                                  batch_first=True)
            attn_mask = attn_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[attn_mask]

            loss = self.loss_func(active_logits.view(-1,
                                                     self.config.num_labels),
                                  labels.view(-1))
            preds = active_logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
        return {'loss': loss, 'predict': preds}
