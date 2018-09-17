"""
.. module:: LayerBiGRU
    :synopsis: BiGRU layer implements standard bidirectional GRU recurrent layer

.. moduleauthor:: Artem Chernodub
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers.layer_birnn_base import LayerBiRNNBase

class LayerBiGRU(LayerBiRNNBase):
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiGRU, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, input_tensor, input_lens, pad_idx): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size = len(input_lens)
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        input_packed = pack_padded_sequence(input_tensor, lengths=input_lens, batch_first=True)
        output_packed, _ = self.rnn(input_packed, h0)
        output, _ = pad_packed_sequence(output_packed, batch_first=True, padding_value=pad_idx)
        return output # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
