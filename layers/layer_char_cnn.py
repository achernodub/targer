import string

import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerCharCNN(LayerBase):
    def __init__(self, gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size, max_char_pad_len):
        super(LayerCharCNN, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        self.filter_num = char_cnn_filter_num
        self.char_window_size = char_window_size
        self.max_char_pad_len = max_char_pad_len
        self.conv_feature_len = max_char_pad_len - char_window_size + 1
        self.output_dim = char_embeddings_dim * char_cnn_filter_num
        self.conv1d = nn.Conv1d(in_channels=char_embeddings_dim,
                                out_channels=char_embeddings_dim * char_cnn_filter_num,
                                kernel_size=char_window_size,
                                groups=char_embeddings_dim)

    def forward(self, char_embeddings_feature):
        batch_num, max_seq_len, _, _ = char_embeddings_feature.shape # shape: batch_num x max_seq_len x  char_embeddings_dim x max_char_pad_len
        conv_out = self.make_gpu(torch.zeros(batch_num, max_seq_len, self.output_dim, self.conv_feature_len,
                                             dtype=torch.float))
        for k in range(max_seq_len):
            curr_y = char_embeddings_feature[:, k, :]  # batch_num x  char_embeddings_dim x max_char_pad_len
            curr_z = self.conv1d(curr_y)  # batch_num x (char_embeddings_dim*char_cnn_filter_num) x conv_feature_len
            conv_out[:, k, :, :] = curr_z
        max_pooling_out, _ = torch.max(conv_out, dim=3)
        return max_pooling_out # shape: batch_num x max_seq_len x (char_embeddings_dim*char_cnn_filter_num)
