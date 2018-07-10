import numpy as np
import torch

from utils_data import *


class TensorsIndexer():
    """
    TensorsIndexer converts input and output sequences (lists of lists of integer indices) to the Torch tensors and back.
    Sequences usually have different length, so shorter sequences are zero-padded, zero values indicates not to include
    them for calculating derivatives in the loss function.
    """

    def vec2idx(self, vec, seq_len):
        return [int(vec[k]) for k in range(seq_len)]

    def indices2tensors(self, inputs_idx, targets_idx):
        data_num = len(inputs_idx)
        max_seq_len = max([len(seq) for seq in inputs_idx])
        inputs = torch.zeros(data_num, max_seq_len, dtype=torch.long)
        targets = torch.zeros(data_num, max_seq_len, dtype=torch.long)
        for k, curr_input_idx in enumerate(inputs_idx):
            curr_target_idx = targets_idx[k]
            curr_seq_len = len(curr_input_idx)
            curr_input = torch.FloatTensor(np.asarray(curr_input_idx))
            curr_target = torch.LongTensor(np.asarray(curr_target_idx))
            inputs[k, :curr_seq_len] = curr_input
            targets[k, :curr_seq_len] = curr_target
        return inputs, targets

    def tensors2indices(self, inputs, targets, masks):
        inputs_idx = list()
        targets_idx = list()
        data_num = inputs.shape[0]
        for k in range(data_num):
            curr_input = inputs[k, :]
            curr_target = targets[k, :]
            curr_mask = masks[k, :]
            curr_seq_len = torch.sum(curr_mask.int())
            curr_input_idx = self.vec2idx(curr_input, curr_seq_len)
            curr_target_idx = self.vec2idx(curr_target, curr_seq_len)
            inputs_idx.append(curr_input_idx)
            targets_idx.append(curr_target_idx)
        return inputs_idx, targets_idx


