"""abstract base class for all bidirectional recurrent layers"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.layers.layer_base import LayerBase


class LayerBiRNNBase(LayerBase):
    """LayerBiRNNBase is abstract base class for all bidirectional recurrent layers."""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiRNNBase, self).__init__(gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2

    def sort_by_seq_len_list(self, seq_len_list):
        data_num = len(seq_len_list)
        sort_indices = sorted(range(len(seq_len_list)), key=seq_len_list.__getitem__, reverse=True)
        reverse_sort_indices = [-1 for _ in range(data_num)]
        for i in range(data_num):
            reverse_sort_indices[sort_indices[i]] = i
        sort_index = self.tensor_ensure_gpu(torch.tensor(sort_indices, dtype=torch.long))
        reverse_sort_index = self.tensor_ensure_gpu(torch.tensor(reverse_sort_indices, dtype=torch.long))
        return sorted(seq_len_list, reverse=True), sort_index, reverse_sort_index

    def pack(self, input_tensor, mask_tensor):
        seq_len_list = self.get_seq_len_list_from_mask_tensor(mask_tensor)
        sorted_seq_len_list, sort_index, reverse_sort_index = self.sort_by_seq_len_list(seq_len_list)
        input_tensor_sorted = torch.index_select(input_tensor, dim=0, index=sort_index)
        return pack_padded_sequence(input_tensor_sorted, lengths=sorted_seq_len_list, batch_first=True), \
               reverse_sort_index

    def unpack(self, output_packed, max_seq_len, reverse_sort_index):
        output_tensor_sorted, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=max_seq_len)
        output_tensor = torch.index_select(output_tensor_sorted, dim=0, index=reverse_sort_index)
        return output_tensor



