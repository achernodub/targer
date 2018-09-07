import torch
import torch.nn as nn

from layers.layer_base import LayerBase
from layers.layer_tdl import LayerTDL

class LayerNARX(LayerBase):
    def __init__(self, input_dim, hidden_dim, output_dim, tdl_seq_len, gpu):
        super(LayerNARX, self).__init__(gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tdl_seq_len = tdl_seq_len
        self.tdl_z = LayerTDL(input_dim, tdl_seq_len, gpu)
        self.tdl_y = LayerTDL(output_dim, tdl_seq_len, gpu)
        self.lin_layer_1 = nn.Linear(in_features =self.tdl_z.output_dim + self.tdl_y.output_dim, out_features = hidden_dim)
        self.lin_layer_2 = nn.Linear(in_features = hidden_dim, out_features = output_dim)
        nn.init.xavier_uniform_(self.lin_layer_1.weight)
        nn.init.xavier_uniform_(self.lin_layer_2.weight)
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len = mask_tensor.shape
        curr_output = self.tensor_ensure_gpu(torch.zeros(batch_size, self.output_dim, dtype=torch.float))
        output = self.tensor_ensure_gpu(torch.Tensor(batch_size, max_seq_len, self.output_dim))
        self.tdl_z.init(batch_size)
        self.tdl_y.init(batch_size)
        for n in range(max_seq_len):
            curr_input = input_tensor[:, n, :]
            z = self.tdl_z(curr_input)
            y = self.tdl_y(curr_output)
            curr_output = self.lin_layer_2(self.tanh(self.lin_layer_1(torch.cat([z, y], dim=1))))
            output[:, n, :] = curr_output
        return self.apply_mask(output, mask_tensor)

    def is_cuda(self):
        return self.lin_layer_1.weight.is_cuda
