import numpy as np

import torch
import torch.nn as nn

from layers.layer_char_embeddings import LayerCharEmbeddings
from layers.layer_char_cnn import LayerCharCNN

print('Hello!')

if 2 == 1:
    data = np.array([0, 3, 4, 5])
    print('data', data)
    conv1d_filter = np.array([1,2])
    print('conv1d_filter', conv1d_filter)
    result = []
    for i in range(3):
        print(data[i:i+2], '*', conv1d_filter, '=', data[i:i+2] * conv1d_filter)
        result.append(np.sum(data[i:i+2] * conv1d_filter))
    print('Conv1d output', result)

max_char_pad_len = 11
char_embeddings_dim = 4
char_cnn_filter_num = 30
char_window_size = 3
gpu = 0

word_sequences = [['Paris', 'is', 'good'], ['London', 'is', 'a', 'capital', 'of', 'UK', '.']]

print(word_sequences)

layer_char_embeddings = LayerCharEmbeddings(gpu=gpu, char_embeddings_dim=char_embeddings_dim,
                                            freeze_char_embeddings=False,
                                            max_char_pad_len=max_char_pad_len)

layer_char_embeddings.cuda()

y = layer_char_embeddings(word_sequences) # num_batch x max_seq_len X max_char_pad_len x char_embeddings_dim
print('y.shape', y.shape)


'''




batch_num, max_seq_len, _, _ = y.shape
conv = torch.nn.Conv1d(in_channels=char_embeddings_dim, out_channels=char_embeddings_dim * char_cnn_filter_num, kernel_size=3, groups=char_embeddings_dim)
print('\nconv.weight.shape', conv.weight.shape, '\n')

conv_feature_dim = max_char_pad_len - char_window_size + 1
print('conv_feature_dim', conv_feature_dim)

conv_out = torch.zeros(batch_num, max_seq_len, char_embeddings_dim * char_cnn_filter_num, conv_feature_dim, dtype=torch.float)
print('conv_out.shape', conv_out.shape)

for k in range(max_seq_len):
    curr_y = y[:, k, :] # batch_num x  char_embeddings_dim x max_char_pad_len

    print('\ncurr_y', curr_y.shape)

    curr_z = conv(curr_y) # batch_num x (char_embeddings_dim*filter_num) x conv_feature_dim

    print('curr_z', curr_z.shape)

    conv_out[:, k, :, :] = curr_z

max_pooling_out, _ = torch.max(conv_out, dim=3)

print('max_pooling_out', max_pooling_out.shape) # batch_num x max_seq_len x (char_embeddings_dim*filter_num)

'''

layer_char_cnn = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size, max_char_pad_len)

layer_char_cnn.cuda()

o = layer_char_cnn(y)

print('o', o.shape)

print(o)

print('end.')