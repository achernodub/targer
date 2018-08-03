import torch.nn as nn


class LayerBase(nn.Module):
    def __init__(self, gpu):
        super(LayerBase, self).__init__()
        self.gpu = gpu

    def make_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor
