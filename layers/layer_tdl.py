import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerTDL(LayerBase):
    def __init__(self, input_dim, tdl_seq_len, gpu):
        super(LayerTDL, self).__init__(gpu)
        self.input_dim = input_dim
        self.tdl_seq_len = tdl_seq_len
        self.output_dim = input_dim * tdl_seq_len
        self.tdl = self.tensor_ensure_gpu(torch.Tensor(1, tdl_seq_len, input_dim).fill_(0)) # batch_size x tdl_seq_len x dim

    def push(self, input):
        for k in range(self.tdl_seq_len - 1, 0, -1):
            self.tdl[:, k, :] = self.tdl[:, k - 1, :]
        self.tdl[:, 0, :] = input

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len = mask_tensor.shape
        output = self.tensor_ensure_gpu(torch.Tensor(batch_size, max_seq_len, self.output_dim).fill_(0))
        print('input_tensor.shape', input_tensor.shape)
        for n in range(max_seq_len):
            self.push(input_tensor[:, n, :])
            curr_output = self.tdl.view(batch_size, self.output_dim)
            output[:, n, :] = curr_output
        return output

    def is_cuda(self):
        return True #############################################################################################
