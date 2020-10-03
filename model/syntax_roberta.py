# coding=utf-8

"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_utils import (PretrainedConfig, add_start_docstrings)
from pytorch_transformers.modeling_bert import (BertConfig,
                                                BertLayerNorm,
                                                BertPreTrainedModel, gelu)

from model.crf import CRF
from model.loss import LabelSmoothingLoss, SequenceCriteriaCRF
from model.syntax_bert import BertModel
from utils import constant
import numpy as np

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
}


class SyntaxRobertaConfig(PretrainedConfig):
    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=50625,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=514,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-05,
                 use_syntax=True,
                 syntax=None,
                 **kwargs):
        super(SyntaxRobertaConfig, self).__init__(**kwargs)
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


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings_modified = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings_modified(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids, token_type_ids=token_type_ids,
                                                      position_ids=position_ids)


class RobertaConfig(BertConfig):
    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.

    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.

    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.RobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the RobertaTokenizer.encode function with 
            the ``add_special_tokens`` parameter set to ``True``.

            RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputing raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                wp_token_mask=None, dep_head=None,
                dep_rel=None, wp_rows=None, align_sizes=None, seq_len=None,
                subj_pos=None, obj_pos=None, linguistic_token_mask=None):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids,
                                                 token_type_ids,
                                                 wp_token_mask,
                                                 dep_head,
                                                 dep_rel,
                                                 wp_rows,
                                                 align_sizes,
                                                 seq_len,
                                                 subj_pos,
                                                 obj_pos,
                                                 linguistic_token_mask)


class SyntaxRobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, label_smoothing=0.):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if label_smoothing == 0.:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = LabelSmoothingLoss(config.num_labels, label_smoothing)
        if config.crf:
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions = nn.Parameter(torch.randn(config.num_labels, config.num_labels))
            nn.init.xavier_uniform_(self.transitions.data)
            self.crf = CRF(config.label_map, config=config, transitions=self.transitions)
            self.loss_func = SequenceCriteriaCRF(self.crf)
        self.config = config
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, wp_token_mask=None,
                labels=None, dep_head=None, dep_rel=None, wp_rows=None, align_sizes=None,
                seq_len=None, subj_pos=None, obj_pos=None, linguistic_token_mask=None, hidden_dropout_prob=0.1,
                **kwargs):
        outputs = self.roberta(input_ids,
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

        sequence_output = F.dropout(sequence_output, p=hidden_dropout_prob, training=self.training)
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
            attn_mask = [torch.LongTensor([1] * len(row)).to(logits.get_device()) for row in wp_rows]
            attn_mask = nn.utils.rnn.pad_sequence(attn_mask, batch_first=True)
            attn_mask = attn_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[attn_mask]

            loss = self.loss_func(active_logits.view(-1, self.config.num_labels), labels.view(-1))
            preds = active_logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
        return {'loss': loss, 'predict': preds}
