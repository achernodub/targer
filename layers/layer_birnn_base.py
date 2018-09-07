import torch.nn as nn
from layers.layer_base import LayerBase

class LayerBiRNNBase(LayerBase):
    def __init__(self, input_dim, hidden_dim, gpu, bi=True):
        super(LayerBiRNNBase, self).__init__(gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if bi:
            self.output_dim = hidden_dim * 2
        else:
            self.output_dim = hidden_dim
