import torch.nn as nn


class LayerBase(nn.Module):
    def __init__(self, gpu):
        super(LayerBase, self).__init__()
        self.gpu = gpu

    def tensor_ensure_gpu(self, tensor):
        if self.is_cuda():
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()
