import logging
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import constant
from utils.utils import get_positions


logger = logging.getLogger(__name__)


def _wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel):
    wp_dep_head, wp_dep_rel = [], []
    # Default ROOT position is the [CLS] token
    # root_pos = 0
    for i, (idx, slen) in enumerate(zip(word_idxs, subword_token_len)):
        if i == 0 or i == len(subword_token_len) - 1:
            wp_dep_rel.append(constant.DEPREL_TO_ID['special_rel'])
            # Goal is to introduce an edge from ROOT token to [CLS] and [SEP] tokens.
            # As the position of the root token is unknown, we will fill these later, once it becomes known
            wp_dep_head.append(idx + 1)
        else:
            rel = dep_rel[i - 1]
            wp_dep_rel.append(rel)
            # This i - 1 takes care of the extra [CLS] token in front
            head = dep_head[i - 1]
            # ROOT token in the parse tree
            if head == 0:
                # This index is what the other words will refer to as the ROOT word
                # root_pos = i + 1
                wp_dep_head.append(0)
            else:
                # Stanford Dependency Parses are offset by 1 as 0 is the ROOT token
                # if head < len(word_idxs):
                if head < max(word_idxs):
                    # Obtain the index of the displaced version of the same head
                    new_pos = word_idxs[head - 1 + 1]
                else:  # TODO: Fix this hack arising due to long lengths
                    new_pos = idx + 1 # self-connection
                wp_dep_head.append(new_pos + 1)

            for _ in range(1, slen):
                # Add special DEP-REL for the subwords
                wp_dep_rel.append(constant.DEPREL_TO_ID['subtokens'])
                # Add special directed edges from the first subword to the next subword
                wp_dep_head.append(idx + 1)
    # wp_dep_head[0] = root_pos
    # wp_dep_head[-1] = root_pos
    return wp_dep_head, wp_dep_rel


def _handle_long_sent_parses(dep_head, dep_rel, length):
    dep_rel = dep_rel[:length]
    dep_head, truncated_head = dep_head[:length], dep_head[length:]

    # Check if the ROOT lies in the remaining part of the truncated sequence
    # And if so, make the last token in the truncated sequence as ROOT
    is_root = [True for x in truncated_head if x == 0]
    if is_root:
        dep_head[-1] = 0

    # Assert that there is only one ROOT in the parse tree
    dep_root_ = [i for i, x in enumerate(dep_head) if x == 0]
    assert len(dep_root_) == 1

    # If head word index is greater than max_length then connect the word to itself
    for i, head_word_index in enumerate(dep_head):
        if head_word_index > len(dep_head):
            dep_head[i] = i + 1
            # dep_head[i] = length

    return dep_head, dep_rel


def _tokenize_with_bert(sequence, tokenizer):
    """
    Wordpiece-tokenize a list of tokens and return vocab ids.
    """
    word_idxs, bert_tokens, subword_token_len = [], [], []
    idx = 0
    for s in sequence:
        tokens = tokenizer.tokenize(s)
        subword_token_len.append(len(tokens))
        word_idxs += [idx]
        bert_tokens += tokens
        idx += len(tokens)
    return bert_tokens, word_idxs, subword_token_len


def _compute_alignment(word_idxs, subword_token_len):
    alignment = []
    for i, l in zip(word_idxs, subword_token_len):
        assert l > 0
        aligned_subwords = []
        for j in range(l):
            aligned_subwords.append(i + j)
        alignment.append(aligned_subwords)
    return alignment


def _get_boundary_sensitive_alignment(word_pieces, raw_tokens, alignment):
    align_sizes = [0 for _ in range(len(word_pieces))]
    wp_rows = []
    for word_piece_slice in alignment:
        wp_rows.append(word_piece_slice)
        for i in word_piece_slice:
            align_sizes[i] += 1
    # To make this weighting work, we "align" the boundary tokens against
    # every token in their sentence. The boundary tokens are otherwise
    # unaligned, which is how we identify them.
    offset = 0
    for i in range(len(word_pieces)):
        if align_sizes[offset + i] == 0:
            align_sizes[offset + i] = len(raw_tokens)
            for j in range(len(raw_tokens)):
                wp_rows[j].append(offset + i)
    return wp_rows, align_sizes


class FeaturizedDataset(Dataset):
    def __init__(self, examples, opt=None, tokenizer=None, label_map=None, mask_padding_with_zero=True,
                 pad_token=0, cls_token_segment_id=1, pad_token_segment_id=0,
                 sequence_a_segment_id=0, cached_features=False):
        super().__init__()
        if cached_features:
            self.data = examples
        else:
            self.opt = opt
            self.cls_token = tokenizer.cls_token
            self.sep_token = tokenizer.sep_token
            self.label_map = label_map
            self.max_seq_length = opt.max_seq_length
            self.mask_padding_with_zero = mask_padding_with_zero
            self.pad_token = pad_token
            self.cls_token_segment_id = cls_token_segment_id
            self.pad_token_segment_id = pad_token_segment_id
            self.sequence_a_segment_id = sequence_a_segment_id
            if opt.sample_rate < 1: # subsample
                k = int(len(examples) * opt.sample_rate)
                examples = random.sample(examples, k)
            self.data = self.convert_examples_to_features(examples, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def map_to_label_id(self, label):
        if isinstance(label, str):
            return self.label_map[label]
        elif isinstance(label, list):
            return [self.label_map[x] for x in label]

    def convert_examples_to_features(self, examples, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
        """
        data = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            raw_tokens = example.text_a.split()
            dep_head, dep_rel = example.dep_head, example.dep_rel
            tokens, word_idxs, subword_token_len = _tokenize_with_bert(raw_tokens, tokenizer)
            while len(tokens) > self.max_seq_length - 2:  # Account for [CLS] and [SEP] with "- 2"
                # tokens = tokens[:(max_seq_length - 2)]
                # TODO: Maybe, find a better solution to deal with this problem
                ratio = len(tokens) / len(raw_tokens)
                est_len = int((self.max_seq_length - 2) / ratio)
                raw_tokens = raw_tokens[:est_len]
                # Takes care of edge cases for dependency head and dependency relation when truncating a sequence
                dep_head, dep_rel = _handle_long_sent_parses(dep_head, dep_rel, est_len)
                tokens, word_idxs, subword_token_len = _tokenize_with_bert(raw_tokens, tokenizer)

            alignment = _compute_alignment(word_idxs, subword_token_len)
            if example.verb_index:
                # This takes care of the [CLS] and [SEP] tokens
                verb_indicator = [0] * (sum(map(lambda x: len(x), alignment[:example.verb_index])) + 1) + \
                                 [1] * len(alignment[example.verb_index]) + \
                                 [0] * (sum(map(lambda x: len(x), alignment[example.verb_index + 1:])) + 1)

            if example.subj_pos and example.obj_pos:
                def get_aligned_idxs(ent_pos):
                    ent_pos = [i for i, p in enumerate(ent_pos) if p == 0]
                    start_idx = sum(map(lambda x: len(x), alignment[:ent_pos[0]])) + 1
                    end_idx = start_idx + sum(map(lambda x: len(x), alignment[ent_pos[0]: ent_pos[-1]]))
                    return get_positions(start_idx, end_idx, len(tokens) + 2)

                example.subj_pos = get_aligned_idxs(example.subj_pos)
                example.obj_pos = get_aligned_idxs(example.obj_pos)

            # The convention in BERT is:
            # For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.

            # Add [SEP] token at ending
            tokens = tokens + [self.sep_token]
            segment_ids = [self.sequence_a_segment_id] * len(tokens)
            subword_token_len = subword_token_len + [1]
            word_idxs.append(len(tokens) - 1)

            # Add [CLS] token at beginning
            tokens = [self.cls_token] + tokens
            segment_ids = [self.cls_token_segment_id] + segment_ids
            subword_token_len = [1] + subword_token_len
            word_idxs = [0] + [i + 1 for i in word_idxs]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # Increase alignment position by offset 1 i.e. because of the [CLS] token at the start
            alignment = [[val + 1 for val in list_] for list_ in alignment]
            wp_rows, align_sizes = _get_boundary_sensitive_alignment(input_ids, raw_tokens, alignment)

            if self.opt.wordpiece_aligned_dep_graph:
                dep_head, dep_rel = _wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel)
                assert len(dep_rel) == len(tokens)
                assert len(dep_head) == len(tokens)
                assert max(dep_head) <= len(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)

            input_ids = input_ids + ([self.pad_token] * padding_length)
            # dep_rel = dep_rel + ([self.pad_token] * (self.max_seq_length - len(dep_rel)))
            input_mask = input_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)

            if example.verb_index:
                segment_ids = verb_indicator
            segment_ids = segment_ids + ([self.pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            label = self.map_to_label_id(example.label)
            subj_pos = torch.LongTensor(example.subj_pos) if example.subj_pos else None
            obj_pos = torch.LongTensor(example.obj_pos) if example.obj_pos else None
            verb_index = example.verb_index if example.verb_index else None

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(f"label: {example.label} (id = {label})")

            data.append((example.text_a.split(),
                         input_ids,
                         input_mask,
                         segment_ids,
                         label,
                         dep_head,
                         dep_rel,
                         wp_rows,
                         align_sizes,
                         len(tokens),
                         subj_pos,
                         obj_pos,
                         verb_index,
                         example.guid))
        return data


class FeaturizedDataLoader(DataLoader):
    def __init__(self, dataset, opt, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        self.eval = eval
        self.opt = opt
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch_size = len(batch_data)
        batch = list(zip(*batch_data))
        assert len(batch) == 14

        maxlen = max(batch[9])
        tensorized = OrderedDict()
        tensorized['input_tokens'] = batch[0]
        tensorized['input_ids'] = torch.LongTensor(batch[1])[:, :maxlen].to(self.opt.device)
        tensorized['wp_token_mask'] = torch.LongTensor(batch[2])[:, :maxlen].to(self.opt.device)
        tensorized['token_type_ids'] = torch.LongTensor(batch[3])[:, :maxlen].to(self.opt.device)

        if self.opt.task_name in ("ontonotes_ner", "ontonotes_srl", "conll2005wsj_srl", "conll2005brown_srl"):
            labels = [x for l in batch[4] for x in l]
        elif self.opt.task_name in ("tacred",):
            labels = batch[4]
        tensorized['labels'] = torch.LongTensor(labels).to(self.opt.device)
        tensorized['dep_head'] = batch[5]
        tensorized['dep_rel'] = batch[6]
        tensorized['wp_rows'] = batch[7]
        tensorized['align_sizes'] = batch[8]
        tensorized['seq_len'] = batch[9]
        # Below, keeping padding value as a number different from zero as
        #  "subj_pos" and "obj_pos" tensors uses "0" to indicate their position
        if not all(x is None for x in batch[10]):
            tensorized['subj_pos'] = nn.utils.rnn.pad_sequence(batch[10],
                                                               batch_first=True,
                                                               padding_value=-100).to(self.opt.device)
        else:
            tensorized['subj_pos'] = [None] * batch_size
        if not all(x is None for x in batch[11]):
            tensorized['obj_pos'] = nn.utils.rnn.pad_sequence(batch[11],
                                                              batch_first=True,
                                                              padding_value=-100).to(self.opt.device)
        else:
            tensorized['obj_pos'] = [None] * batch_size

        if not all(x is None for x in batch[12]):
            tensorized['verb_index'] = batch[12]
        else:
            tensorized['verb_index'] = [None] * batch_size

        inter_attn_mask = []
        for row in batch[7]:
            inter_attn_mask.append(torch.LongTensor([1] * len(row)).to(self.opt.device))
        tensorized['linguistic_token_mask'] = nn.utils.rnn.pad_sequence(inter_attn_mask,
                                                                        batch_first=True,
                                                                        padding_value=0)
        if self.eval:
            tensorized["guid"] = batch[13]

        return tensorized
