import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerTDL(LayerBase):
    def __init__(self, input_dim, tdl_seq_len, gpu):
        super(LayerTDL, self).__init__(gpu)
        self.input_dim = input_dim
        self.tdl_seq_len = tdl_seq_len
        self.output_dim = input_dim * tdl_seq_len

    def init(self, batch_size):
        self.tdl = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.tdl_seq_len, self.input_dim).fill_(0))

    def push(self, curr_input):
        for k in range(self.tdl_seq_len - 1, 0, -1):
            self.tdl[:, k, :] = self.tdl[:, k - 1, :]
        self.tdl[:, 0, :] = curr_input

    def forward(self, curr_input):
        batch_size = curr_input.shape[0]
        self.push(curr_input)
        return self.tdl.view(batch_size, self.output_dim)

    def is_cuda(self):
        return True #############################################################################################
