"""class implements standard bidirectional GRU recurrent layer"""
import torch
import torch.nn as nn
from src.layers.layer_birnn_base import LayerBiRNNBase

class LayerBiGRU(LayerBiRNNBase):
    """BiGRU layer implements standard bidirectional GRU recurrent layer"""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiGRU, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        input_packed, reverse_sort_index = self.pack(input_tensor, mask_tensor)
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        output_packed, _ = self.rnn(input_packed, h0)
        output_tensor = self.unpack(output_packed, max_seq_len, reverse_sort_index)
        return output_tensor  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda

    def forward_old(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        # Init rnn's states by zeros
        rnn_forward_h = self.tensor_ensure_gpu(torch.zeros(batch_size, self.hidden_dim))
        rnn_backward_h = self.tensor_ensure_gpu(torch.zeros(batch_size, self.hidden_dim))
        # Forward pass in both directions
        output = self.tensor_ensure_gpu(torch.zeros(batch_size, max_seq_len, self.hidden_dim * 2))
        for l in range(max_seq_len):
            n = max_seq_len - l - 1
            rnn_forward_h = self.rnn_forward_layer(input_tensor[:, l, :], rnn_forward_h)
            rnn_backward_h = self.rnn_backward_layer(input_tensor[:, n, :], rnn_backward_h)
            #print('rnn_backward_h.shape', rnn_backward_h.shape)
            #print('mask_tensor.shape', mask_tensor.shape)
            output[:, l, :self.hidden_dim] = self.apply_mask(rnn_forward_h, mask_tensor[:, l])
            #output[:, n, self.hidden_dim:] = self.apply_mask(rnn_backward_h, mask_tensor[:, n])
            output[:, n, self.hidden_dim:] = self.tensor_ensure_gpu(torch.zeros(batch_size, self.hidden_dim))
        return output # shape: batch_size x max_seq_len x hidden_dim*2
