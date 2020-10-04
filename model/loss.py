import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_size, label_smoothing=0.1):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (label_size - 1)
        one_hot = torch.full((label_size,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        self.one_hot = self.one_hot.to(output.device)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output), model_prob, reduction='batchmean')


class SequenceCriteriaCRF(nn.Module):
    def __init__(self, model):
        super(SequenceCriteriaCRF, self).__init__()
        self.model = model

    def forward(self, top_layer_output, labels, sent_len):
        batch_size = top_layer_output.size()[0]
        gold_score = self.model._score_sentence_batch(top_layer_output,
                                                      labels,
                                                      sent_len)
        forward_score = self.model._forward_alg_batch(top_layer_output,
                                                      sent_len)
        avg_log_likelihood = (torch.sum(forward_score) - torch.sum(gold_score)) / batch_size
        return avg_log_likelihood
