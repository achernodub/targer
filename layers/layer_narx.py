import torch
import torch.nn as nn

from layers.layer_birnn_base import LayerBiRNNBase

class LayerNARX(LayerBiRNNBase):
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerNARX, self).__init__(input_dim, hidden_dim, gpu)
        #nn.init.xavier_uniform_(rnn.weight_hh_l0)

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        output, _ = self.rnn(input_tensor, (h0, c0))
        return self.apply_mask(output)  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda
