import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils import *


class TaggerBase(nn.Module):
    def predict_idx_from_tensor(self, inputs_tensor):# inputs_tensor: batch_size x max_seq_len
        self.eval()
        outputs_tensor = self.forward(inputs_tensor) # batch_size x num_class+1 x max_seq_len
        outputs_idx = list()
        batch_size, max_seq_len = inputs_tensor.size()
        for k in range(batch_size):
            list_idx = list()
            for l in range(max_seq_len):
                curr_input = inputs_tensor[k, l].item()
                if curr_input > 0: # ignore zero-padded inputs of sequence
                    curr_output = outputs_tensor[k, 1:, l]
                    _, max_no = curr_output.max(0)
                    list_idx.append(max_no.item() + 1)
            outputs_idx.append(list_idx)
        return outputs_idx

    def predict_tags_from_tensor(self, inputs_tensor, sequences_indexer):
        outputs_idx = self.predict_idx_from_tensor(inputs_tensor)
        return sequences_indexer.idx2tag(outputs_idx)

    def predict_tags_from_idx(self, inputs_idx, sequences_indexer):
        inputs_tensor = sequences_indexer.idx2tensor(inputs_idx)
        return self.predict_tags_from_tensor(inputs_tensor, sequences_indexer)

    def predict_tags_from_tokens(self, token_sequences, sequences_indexer):
        inputs_idx = sequences_indexer.token2idx(token_sequences)
        return self.predict_tags_from_idx(inputs_idx, sequences_indexer)












