import string

import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerCharCNN(LayerBase):
    def __init__(self, gpu, char_embeddings_dim, filter_num, char_window_size, word_len):
        super(LayerCharCNN, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        self.char_cnn_filter_num = filter_num
        self.char_window_size = char_window_size
        self.word_len = word_len
        self.conv_feature_len = word_len - char_window_size + 1
        self.output_dim = char_embeddings_dim * filter_num
        self.conv1d = nn.Conv1d(in_channels=char_embeddings_dim,
                                out_channels=char_embeddings_dim * filter_num,
                                kernel_size=char_window_size,
                                groups=char_embeddings_dim)

    def is_cuda(self):
        return self.conv1d.weight.is_cuda

    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len
        batch_num, max_seq_len, char_embeddings_dim, word_len = char_embeddings_feature.shape
        #print('batch_num=%d, max_seq_len=%d, char_embeddings_dim=%d, word_len=%d' % (batch_num, max_seq_len, char_embeddings_dim, word_len))
        # curr_in: batch_num x  char_embeddings_dim x word_len
        # curr_out: batch_num x char_cnn_filter_num*char_embeddings_dim x conv_feature_len
        #conv_out = self.make_gpu(torch.zeros(batch_num, max_seq_len, self.char_cnn_filter_num*self.char_embeddings_dim,
        #                                     self.conv_feature_len, dtype=torch.float))
        max_pooling_out = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len,
                                                             self.char_cnn_filter_num * self.char_embeddings_dim, dtype=torch.float))
        for k in range(max_seq_len):
#            curr_y = char_embeddings_feature[:, k, :, :]  # batch_num x  char_embeddings_dim x word_len
#            curr_z = self.conv1d(curr_y)  # batch_num x filter_num*char_embeddings_dim x conv_feature_len
#            max_pooling_out[:, k, :], _ = torch.max(curr_z, dim=2)
            max_pooling_out[:, k, :], _ = torch.max(self.conv1d(char_embeddings_feature[:, k, :, :]), dim=2)
        #max_pooling_out, _ = torch.max(conv_out, dim=3)
        return max_pooling_out # shape: batch_num x max_seq_len x filter_num*char_embeddings_dim
