import numpy as np
import torch


class Masker():
    """
    Masker converts input and output sequences (lists of lists of integer indices) to the Torch tensors and back.
    Sequences usually have different length, so shorter sequences are zero-padded, mask tensor is used to indicate
    these zero-padded values in order not to include them to the loss & other operations.
    """

    def __init__(self):
        pass

    def indices2tensors(self, inputs_idx, targets_idx):
        data_num = len(inputs_idx)
        max_seq_len = max([len(seq) for seq in inputs_idx])
        inputs = torch.zeros(data_num, max_seq_len).float()
        targets = torch.zeros(data_num, max_seq_len).long()
        masks = torch.zeros(data_num, max_seq_len).byte()
        for k, curr_input_idx in enumerate(inputs_idx):
            curr_target_idx = targets_idx[k]
            curr_seq_len = len(curr_input_idx)
            curr_input = torch.FloatTensor(np.asarray(curr_input_idx))
            curr_target = torch.LongTensor(np.asarray(curr_target_idx))
            curr_mask = torch.ones(1, curr_seq_len).byte()
            inputs[k, :curr_seq_len] = curr_input
            targets[k, :curr_seq_len] = curr_target
            masks[k, :curr_seq_len] = curr_mask
        return inputs, targets, masks

    def vec2idx(self, vec, seq_len):
        return [int(vec[k]) for k in range(seq_len)]

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




