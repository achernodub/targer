"""class implements standard bidirectional Vanilla recurrent layer"""
import torch
import torch.nn as nn
from src.layers.layer_birnn_base import LayerBiRNNBase

class LayerBiVanilla(LayerBiRNNBase):
    """BiVanilla layer implements standard bidirectional Vanilla recurrent layer."""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiVanilla, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        output, _ = self.rnn(input_tensor, h0)
        return self.apply_mask(output, mask_tensor)  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
