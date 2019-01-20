"""abstract base class for all type of layers"""
import torch.nn as nn

class LayerBase(nn.Module):
    """Abstract base class for all type of layers."""
    def __init__(self, gpu):
        super(LayerBase, self).__init__()
        self.gpu = gpu

    def tensor_ensure_gpu(self, tensor):
        if self.is_cuda():
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)

    def get_seq_len_list_from_mask_tensor(self, mask_tensor):
        batch_size = mask_tensor.shape[0]
        return [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
