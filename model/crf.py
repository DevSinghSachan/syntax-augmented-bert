# This code is inspired from here:
# http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/crf/python/ops/crf.py

import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# Computes the log sum exp of the incoming matrix
def log_sum_exp_matrix(matrix):
    max_val, _ = torch.max(matrix, dim=1)
    # max_value_broadcast = max_val.expand(matrix.size())
    max_value_broadcast = max_val.view(-1, 1).expand_as(matrix)
    return max_val + torch.log(torch.sum(torch.exp(matrix - max_value_broadcast), dim=1))


# Computes the log sum exp of the incoming matrix
def log_sum_exp_matrix_batch(matrix):
    max_val, _ = torch.max(matrix, dim=2)
    # max_value_broadcast = max_val.expand(matrix.size())
    R, C = max_val.size()
    max_value_broadcast = max_val.view(R, C, -1).expand_as(matrix)
    return max_val + torch.log(torch.sum(torch.exp(matrix - max_value_broadcast), dim=2))


class CRF(nn.Module):
    def __init__(self, label_map, config, transitions):
        super(CRF, self).__init__()
        self.label_map = label_map
        self.tagset_size = len(label_map)
        self.config = config

        # Matrix of transition parameters.  Entry i, j is the score of transitioning *to* i *from* j.
        self.transitions = transitions
        self.allowed_transitions()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.tagset_size) + ' -> ' + str(self.tagset_size) + ')'

    def allowed_transitions(self):
        self.transition_mask = nn.Parameter(torch.ones(self.tagset_size, self.tagset_size), requires_grad=False)
        self.transition_mask.data[0, 0] = 0. # Due to padding character
        for k1, k2 in itertools.permutations(self.label_map.keys(), 2):
            # transition from / to 'PAD_TOKEN' is not allowed
            if k1 == 'PAD_TOKEN' or k2 == 'PAD_TOKEN':
                self.transition_mask.data[self.label_map[k1], self.label_map[k2]] = 0.
            try:
                te_pos, te_type = k1.split('-')
                if k2 == 'O' and te_pos == 'I':
                    self.transition_mask.data[self.label_map[k1], self.label_map[k2]] = 0.
            except ValueError:
                pass
            try:
                te_pos, te_type = k1.split('-')
                fe_pos, fe_type = k2.split('-')
                if te_pos in ('I') and te_type != fe_type:
                    self.transition_mask.data[self.label_map[k1], self.label_map[k2]] = 0.
            except ValueError:
                pass

    def _forward_alg_batch(self, feats_batch, sequence_lengths):
        device = feats_batch.device
        batch_size = len(feats_batch)
        feats_transpose = torch.transpose(feats_batch, 0, 1).contiguous()
        max_length = len(feats_transpose)

        # Do the forward algorithm to compute the partition function
        # init_alphas = torch.Tensor(batch_size, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        # init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        init_alphas = feats_transpose[0]

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = autograd.Variable(init_alphas).cuda()
        all_alpha = Variable(torch.Tensor(max_length, batch_size)).to(device)
        sequence_lengths_tensor = torch.LongTensor(sequence_lengths).to(device) - 1
        one_hot_tags = torch.ByteTensor(batch_size, max_length).to(device)

        forward_var = init_alphas

        # Swap the time dimension with batch dimension for iteration along time dim
        trans_score = self.transitions.view(1, self.tagset_size, self.tagset_size).expand(batch_size,
                                                                                          self.tagset_size,
                                                                                          self.tagset_size)
        # Initializing the value of alpha
        alpha = log_sum_exp_matrix(forward_var)
        all_alpha[0] = alpha

        for index, feats_b in enumerate(feats_transpose[1:]):
            emit_score = feats_b.view(batch_size, self.tagset_size, 1).expand(batch_size, self.tagset_size, self.tagset_size)
            next_tag_var = forward_var.view(batch_size, 1, self.tagset_size).expand(batch_size, self.tagset_size, self.tagset_size) + emit_score + trans_score
            alphas_t = log_sum_exp_matrix_batch(next_tag_var)
            forward_var = alphas_t.view(batch_size, self.tagset_size)
            alpha = log_sum_exp_matrix(forward_var)
            all_alpha[index + 1] = alpha

        one_hot_tags.zero_()
        one_hot_tags.scatter_(1, sequence_lengths_tensor.view(-1, 1), 1)
        temp = torch.masked_select(torch.t(all_alpha), autograd.Variable(one_hot_tags).bool())
        return temp.view(-1, 1)

    def _score_sentence_batch(self, feats_batch, tags_batch, sequence_lengths):
        binary_mask = self._sequence_mask(sequence_lengths, device=feats_batch.device)

        unary_score = self._emission_potential_score(feats_batch, tags_batch)
        unary_score = torch.masked_select(unary_score, binary_mask)

        binary_score = self._transition_potential_score(tags_batch)
        binary_score = torch.masked_select(binary_score, binary_mask)

        score = unary_score + binary_score
        return score

    def _sequence_mask(self, seq_len, device):
        max_len = max(seq_len)
        batch_size = len(seq_len)
        seq_len_tensor = Variable(torch.FloatTensor(seq_len)).to(device)
        range_row = Variable(torch.arange(0, max_len).view(1, -1).expand(batch_size, max_len)).to(device)
        lengths = seq_len_tensor.view(-1, 1).expand(batch_size, max_len)
        binary_mask = torch.lt(range_row, lengths).view(-1, 1)
        return binary_mask.bool()

    def _emission_potential_score(self, feats_batch, tags_batch):
        # Gives the unary score associated with the tags
        feats = feats_batch.view(-1, self.tagset_size)
        tags = tags_batch.view(-1)
        unary_score = torch.gather(feats, 1, tags.view(-1, 1))
        return unary_score

    # Computes the binary potential score
    def _transition_potential_score(self, tags_batch):
        device = tags_batch.device
        batch_size = len(tags_batch)
        max_len = tags_batch.size()[1]
        dummy_start = Variable(torch.LongTensor([0] * batch_size).view(batch_size, 1)).to(device)
        tags = torch.cat((dummy_start, tags_batch), dim=1)
        mask = Variable(torch.FloatTensor([0] + [1]*(max_len-1))).expand_as(tags_batch).contiguous().view(-1).to(device)

        transitions = self.transitions.view(-1)
        tags_from = tags[:, :-1]
        tags_to = tags[:, 1: ]
        temp = ((self.tagset_size * tags_to) + tags_from).view(-1)
        binary_score = torch.gather(transitions, 0, temp)
        binary_score = binary_score * mask
        binary_score = binary_score.view(-1, 1)
        return binary_score

    # Viterbi Algorithm to decode the best path
    def _viterbi_decode(self, feats):
        backpointers_b = []
        # forward_var = feats[0]
        forward_var = feats[0].view(1, -1)

        constrained_transitions = (self.transitions + ((1 - self.transition_mask) * -10000.0))
        if len(feats) > 1:
            for feat in feats[1:]:
                # next_tag_var = forward_var.expand(self.tagset_size, self.tagset_size) + self.transitions
                next_tag_var = forward_var.expand(self.tagset_size, self.tagset_size) + constrained_transitions
                viterbivars_t, best_tag_id = torch.max(next_tag_var, dim=1)
                forward_var = (viterbivars_t.view(-1) + feat).view(1, -1)
                backpointers_b.append(best_tag_id.data.view(-1))

        # This line can be fixed
        terminal_var = forward_var
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers_b):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_path.reverse()
        return path_score, best_path

    # Find the best path, given the features.
    def forward(self, feats):
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
