import torch.nn as nn


class LayerBiRNNBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiRNNBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.output_dim = hidden_dim * 2

    def make_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor
