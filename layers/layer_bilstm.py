"""
.. module:: LayerBiLSTM
    :synopsis: BiLSTM layer implements standard bidirectional LSTM recurrent layer

.. moduleauthor:: Artem Chernodub
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers.layer_birnn_base import LayerBiRNNBase

class LayerBiLSTM(LayerBiRNNBase):
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiLSTM, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        rnn = nn.LSTM(input_size=input_dim,
                      hidden_size=hidden_dim,
                      num_layers=1,
                      batch_first=True,
                      bidirectional=True)
        # Custom init
        '''nn.init.xavier_uniform_(rnn.weight_hh_l0)
        nn.init.xavier_uniform_(rnn.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(rnn.weight_ih_l0)
        nn.init.xavier_uniform_(rnn.weight_ih_l0_reverse)
        rnn.bias_hh_l0.data.fill_(0)
        rnn.bias_hh_l0_reverse.data.fill_(0)
        rnn.bias_ih_l0.data.fill_(0)
        rnn.bias_ih_l0_reverse.data.fill_(0)
        # Init forget gates to 1
        for names in rnn._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)'''
        self.rnn = rnn

    def forward(self, input_tensor, input_lens, pad_idx): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size = len(input_lens)
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        input_packed = pack_padded_sequence(input_tensor, lengths=input_lens, batch_first=True)
        output_packed, _ = self.rnn(input_packed, (h0, c0))
        output, _ = pad_packed_sequence(output_packed, batch_first=True, padding_value=pad_idx)
        return output # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
