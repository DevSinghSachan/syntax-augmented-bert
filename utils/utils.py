# coding=utf-8

from __future__ import absolute_import, division, print_function

import torch
import tempfile
import subprocess
import shutil
import csv
import os
import sys
import json
from io import open
import numpy as np
from collections import Counter, defaultdict

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from utils import constant
from utils.srl_utils import write_conll_formatted_tags_to_file, convert_bio_tags_to_conll_format


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None, dep_head=None, dep_rel=None, subj_pos=None, obj_pos=None, verb_index=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                    sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
                   specified for train and dev examples, but not for test examples.
            dep_head: The dependency head for each word
            dep_rel: The dependency relation between head and tail words
            subj_pos: Position of the subject words
            obj_pos: Position of the object words
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.dep_head = dep_head
        self.dep_rel = dep_rel
        self.subj_pos = subj_pos
        self.obj_pos = obj_pos
        self.verb_index = verb_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_index, input_ids, input_mask,
                 segment_ids, label_id, dep_head, dep_rel, wp_rows, align_sizes):
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dep_head = dep_head
        self.dep_rel = dep_rel
        self.wp_rows = wp_rows
        self.align_sizes = align_sizes

    def __iter__(self):
        return self

    def __next__(self):
        for k, v in self.__dict__.items():
            yield v


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class OntoNotesNERProcessor(DataProcessor):
    """Processor for OntoNotes-5.0 NER task"""

    def __init__(self):
        super().__init__()
        self.class_weight = Counter()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return constant.OntoNotes_NER_LABEL_TO_ID

    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f'{set_type}-{i}'
            tokens = list(line['token'])
            tags = line['tags']
            text_a = ' '.join(tokens)

            # Adding dependency tree information
            deprel = [t.lower() for t in line['deprel']]
            deprel = map_to_ids(deprel, constant.DEPREL_TO_ID)
            assert all([x != 1 for x in deprel]) # To make sure no relation gets mapped to UNK token

            head = [int(x) for x in line['head']]
            assert any([x == 0 for x in head])

            examples.append(InputExample(guid=guid, text_a=text_a, label=tags, dep_head=head, dep_rel=deprel))
        return examples


class TacredProcessor(DataProcessor):
    """Processor for the TACRED relation extraction data set (EMNLP 2017)"""

    def __init__(self):
        super(TacredProcessor, self).__init__()
        self.special_tokens_dict = {'additional_special_tokens': []}
        self.special_tokens_set = set()
        self.class_weight = Counter()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return constant.TACRED_LABEL_TO_ID

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f'{set_type}-{i}'
            tokens = list(line['token'])
            # anonymize tokens
            ss, se = line['subj_start'], line['subj_end']
            os, oe = line['obj_start'], line['obj_end']
            subj_type = 'SUBJ-' + line['subj_type']
            obj_type = 'OBJ-' + line['obj_type']
            label = line['relation']
            tokens[ss:se + 1] = [subj_type] * (se - ss + 1)
            tokens[os:oe + 1] = [obj_type] * (oe - os + 1)
            subj_pos = get_positions(line['subj_start'], line['subj_end'], len(tokens))
            obj_pos = get_positions(line['obj_start'], line['obj_end'], len(tokens))
            text_a = ' '.join(tokens)

            # Adding dependency tree information
            deprel = [t.lower() for t in line['stanford_deprel']]
            deprel = map_to_ids(deprel, constant.DEPREL_TO_ID)
            assert all([x != 1 for x in deprel])  # To make sure no relation gets mapped to UNK token
            head = [int(x) for x in line['stanford_head']]
            assert any([x == 0 for x in head])
            # TODO: Add information regarding NER and POS tags for each word

            if set_type == 'train':
                self.special_tokens_set.add(subj_type)
                self.special_tokens_set.add(obj_type)
                self.class_weight.update([label])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label,
                                         dep_head=head, dep_rel=deprel, subj_pos=subj_pos,
                                         obj_pos=obj_pos))
        self.special_tokens_dict['additional_special_tokens'] = list(self.special_tokens_set)
        return examples


class OntoNotesSRLProcessor(DataProcessor):
    def __init__(self):
        super(OntoNotesSRLProcessor, self).__init__()
        self.class_weight = Counter()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return constant.OntoNotes_SRL_LABEL_TO_ID

    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f'{set_type}-{i}'
            tokens = list(line['tokens'])
            tags = line['tags'][0]
            text_a = ' '.join(tokens)

            verb_index = line['metadata']['verb_index']
            # Adding dependency tree information
            deprel = [t.lower() for t in line['ontonotes_deprel']]
            deprel = map_to_ids(deprel, constant.DEPREL_TO_ID)
            assert all([x != 1 for x in deprel])  # To make sure no relation gets mapped to UNK token
            head = [int(x) for x in line['ontonotes_head']]
            assert any([x == 0 for x in head])

            if len(head) != len(tokens):
                print("Mismatch between lengths of dep_head and tokens")
                continue

            examples.append(InputExample(guid=guid, text_a=text_a, label=tags, dep_head=head, dep_rel=deprel, verb_index=verb_index))
        return examples


class CoNLL2005WSJSRLProcessor(OntoNotesSRLProcessor):
    def __init__(self):
        super(CoNLL2005WSJSRLProcessor, self).__init__()

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "wsj-test.json")), "test")

    def get_labels(self):
        """See base class."""
        return constant.CoNLL2005_SRL_LABEL_TO_ID

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f'{set_type}-{i}'
            tokens = list(line['tokens'])
            tags = line['tags'][0]
            text_a = ' '.join(tokens)

            verb_index = line['metadata']['verb_index']
            # Adding dependency tree information
            deprel = [t.lower() for t in line['dep_label']]
            deprel = map_to_ids(deprel, constant.DEPREL_TO_ID)
            assert all([x != 1 for x in deprel])  # To make sure no relation gets mapped to UNK token
            head = [int(x) for x in line['dep_head']]
            assert any([x == 0 for x in head])

            if len(head) != len(tokens):
                print("Mismatch between lengths of dep_head and tokens")
                continue

            examples.append(InputExample(guid=guid, text_a=text_a, label=tags, dep_head=head, dep_rel=deprel, verb_index=verb_index))
        return examples


class CoNLL2005BrownSRLProcessor(CoNLL2005WSJSRLProcessor):
    def __init__(self):
        super(CoNLL2005BrownSRLProcessor, self).__init__()

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "brown-test.json")), "test")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "tacred":
    #     return {"f1": score(preds, labels)}
    else:
        raise KeyError(task_name)


class TacredScore(object):
    def __init__(self, tag_to_idx, verbose=False):
        self.correct_by_relation = Counter()
        self.guessed_by_relation = Counter()
        self.gold_by_relation = Counter()
        self.verbose = verbose

    def update(self, prediction, key, *args, **kwargs):
        key = key.detach().cpu().numpy()
        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]
            if gold == 0 and guess == 0:
                pass
            elif gold == 0 and guess != 0:
                self.guessed_by_relation[guess] += 1
            elif gold != 0 and guess == 0:
                self.gold_by_relation[gold] += 1
            elif gold != 0 and guess != 0:
                self.guessed_by_relation[guess] += 1
                self.gold_by_relation[gold] += 1
                if gold == guess:
                    self.correct_by_relation[guess] += 1

    def get_stats(self, **kwargs):
        # Print verbose information
        if self.verbose:
            print("Per-relation statistics:")
            relations = self.gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = self.correct_by_relation[relation]
                guessed = self.guessed_by_relation[relation]
                gold = self.gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        all_metrics = {}
        if self.verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(self.guessed_by_relation.values()) > 0:
            prec_micro = float(sum(self.correct_by_relation.values())) / float(sum(self.guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(self.gold_by_relation.values()) > 0:
            recall_micro = float(sum(self.correct_by_relation.values())) / float(
                sum(self.gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        print("P (micro): {:.3%}| R (micro): {:.3%}| F1 (micro): {:.3%}".format(prec_micro, recall_micro, f1_micro))
        all_metrics["precision-overall"] = prec_micro
        all_metrics["recall-overall"] = recall_micro
        all_metrics["f1-measure-overall"] = f1_micro
        return all_metrics


class NERScore(object):
    def __init__(self, tag_to_idx, verbose=False, default_key='O'):
        self.idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
        self.default_key = default_key
        self.accs = []
        self.correct_preds = 0.
        self.total_correct = 0.
        self.total_preds = 0.

    def get_chunk_type(self, tok):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}
        Returns:
            tuple: "B", "PER"
        """
        tag_name = self.idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def get_chunks(self, seq):
        """Given a sequence of tags, group entities and their position
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]
        """
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == self.default_key and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != self.default_key:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)
        return chunks

    def update(self, prediction, key, *args, **kwargs):
        key = key.detach().cpu().numpy()
        key = key.tolist()
        prediction = prediction.tolist()
        self.accs += [k == p for k, p in zip(key, prediction)]
        lab_chunks = set(self.get_chunks(key))
        lab_pred_chunks = set(self.get_chunks(prediction))
        self.correct_preds += len(lab_chunks & lab_pred_chunks)
        self.total_preds += len(lab_pred_chunks)
        self.total_correct += len(lab_chunks)

    def get_stats(self, *args, **kwargs):
        # Print the aggregate score
        all_metrics = {}
        precision = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        recall = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1_measure = 2 * precision * recall / (precision + recall) if self.correct_preds > 0 else 0
        acc = np.mean(self.accs)
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        print("P (micro): {:.3%}| R (micro): {:.3%} | F1 (micro): {:.3%}".format(precision, recall, f1_measure))
        return all_metrics


DEFAULT_SRL_EVAL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "srl-eval.pl"))


class SRLScore(object):
    """
    This class uses the external srl-eval.pl script for computing the CoNLL SRL metrics.
    Note that this metric reads and writes from disk quite a bit. In particular, it
    writes and subsequently reads two files per __call__, which is typically invoked
    once per batch. You probably don't want to include it in your training loop;
    instead, you should calculate this on a validation set only.

    Parameters
    ----------
    srl_eval_path : ``str``, optional.
        The path to the srl-eval.pl script.
    ignore_classes : ``List[str]``, optional (default=``None``).
        A list of classes to ignore.
    """

    def __init__(self, tag_to_idx, srl_eval_path=DEFAULT_SRL_EVAL_PATH, ignore_classes=['V']):
        self.idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
        self._srl_eval_path = srl_eval_path
        self._ignore_classes = set(ignore_classes)
        # These will hold per label span counts.
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
        self._overall = []

    def update(self, pred_tags, gold_tags, verb_indices, sentences, guid=None, training=False):
        """
        Parameters
        ----------
        verb_indices : ``List[Optional[int]]``, required.
            The indices of the verbal predicate in the sentences which
            the gold labels are the arguments for, or None if the sentence
            contains no verbal predicate.
        sentences : ``List[List[str]]``, required.
            The word tokens for each instance in the batch.
        pred_tags : ``List[List[str]]``, required.
            A list of predicted CoNLL-formatted SRL tags (itself a list) to compute score for.

            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the tags into the metric,
            if applicable.
        gold_tags : ``List[List[str]]``, required.
            A list of gold CoNLL-formatted SRL tags (itself a list) to use as a reference.
            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the
            tags into the metric, if applicable.
        """
        if training:
            return

        if not os.path.exists(self._srl_eval_path):
            raise OSError(f"srl-eval.pl not found at {self._srl_eval_path}.")
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        pred_path = os.path.join(tempdir, "predicted.txt")

        l = [len(row) for row in sentences]
        pred_tags = torch.split(pred_tags, l)
        pred_tags = [[self.idx_to_tag[tag.item()] for tag in tag_row] for tag_row in pred_tags]
        batch_conll_pred_tags = [convert_bio_tags_to_conll_format(tags) for tags in pred_tags]

        gold_tags = torch.split(gold_tags, l)
        gold_tags = [[self.idx_to_tag[tag.item()] for tag in tag_row] for tag_row in gold_tags]
        batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for tags in gold_tags]

        with open(pred_path, "w", encoding="utf-8") as predicted_file, \
                open(gold_path, "w", encoding="utf-8") as gold_file:
            for verb_index, sentence, pred_tag_seq, gold_tag_seq in zip(verb_indices,
                                                                        sentences,
                                                                        batch_conll_pred_tags,
                                                                        batch_conll_gold_tags):
                write_conll_formatted_tags_to_file(predicted_file,
                                                   gold_file,
                                                   verb_index,
                                                   sentence,
                                                   pred_tag_seq,
                                                   gold_tag_seq)
        perl_script_command = ["perl", self._srl_eval_path, gold_path, pred_path]
        completed_process = subprocess.run(perl_script_command,
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True,
                                           check=True)

        for line in completed_process.stdout.split("\n"):
            stripped = line.strip().split()
            if len(stripped) == 7:
                tag = stripped[0]
                # Overall metrics are calculated in get_metric, skip them here.
                if tag == "Overall" or tag in self._ignore_classes:
                    if tag == "Overall":
                        result_dict = {"excess": int(stripped[2]),
                                       "missed": int(stripped[3]),
                                       "f1": float(stripped[6]),
                                       "gold_tags": gold_tags[0],
                                       "predicted_tags": pred_tags[0],
                                       "sentence": sentences[0],
                                       "guid": guid}
                        self._overall.append(result_dict)
                    continue
                # This line contains results for a span
                num_correct = int(stripped[1])
                num_excess = int(stripped[2])
                num_missed = int(stripped[3])
                self._true_positives[tag] += num_correct
                self._false_positives[tag] += num_excess
                self._false_negatives[tag] += num_missed
        shutil.rmtree(tempdir)

    def get_overall_list(self):
        return self._overall

    def get_stats(self, reset=False, training=False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        if training:
            return
        all_tags = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            if tag == "overall":
                raise ValueError(
                    "'overall' is disallowed as a tag type, "
                    "rename the tag type to something else if necessary."
                )
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()

        print("P (micro): {:.3%}| R (micro): {:.3%} | F1 (micro): {:.3%}".format(precision, recall, f1_measure))
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)


scorer = {'tacred': (TacredScore, constant.TACRED_LABEL_TO_ID),
          'ontonotes_ner': (NERScore, constant.OntoNotes_NER_LABEL_TO_ID),
          'ontonotes_srl': (SRLScore, constant.OntoNotes_SRL_LABEL_TO_ID),
          'conll2005wsj_srl': (SRLScore, constant.CoNLL2005_SRL_LABEL_TO_ID),
          'conll2005brown_srl': (SRLScore, constant.CoNLL2005_SRL_LABEL_TO_ID)}

processors = {
    "sst-2": Sst2Processor,
    "tacred": TacredProcessor,
    "ontonotes_ner": OntoNotesNERProcessor,
    "ontonotes_srl": OntoNotesSRLProcessor,
    "conll2005wsj_srl": CoNLL2005WSJSRLProcessor,
    "conll2005brown_srl": CoNLL2005BrownSRLProcessor
}

output_modes = {
    "sst-2": "classification",
    "tacred": "classification",
    "ontonotes_ner": "token_classification",
    "ontonotes_srl": "token_classification",
    "conll2005wsj_srl": "token_classification",
    "conll2005brown_srl": "token_classification"
}

GLUE_TASKS_NUM_LABELS = {
    "sst-2": 2,
    "tacred": 42,
    "ontonotes_ner": 37,
    "ontonotes_srl": 129,
    "conll2005wsj_srl": 106,
    "conll2005brown_srl": 106
}


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))
