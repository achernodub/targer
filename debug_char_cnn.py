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

#word_sequences = [['Paris', 'is', 'good'], ['London', 'is', 'a', 'capital', 'of', 'UK', '.']]
#word_sequences = [['As', 'a', 'mass', 'media', 'majoring', 'student', ',', 'I', 'think', 'that', 'it', "'", 's', 'true', 'that', 'media', 'is', 'overemphasizing', 'the', 'personal', 'lives', 'of', 'famous', 'people', 'in', 'this', 'modern', 'society', '.', 'In', 'my', 'point', 'of', 'view', ',', 'disadvantages', 'of', 'overemphasizing', 'on', 'personal', 'lives', 'of', 'famous', 'people', 'by', 'media', 'outweigh', 'its', 'advantages', '.', 'My', 'reasons', 'include', 'the', 'following', ':']]

word_sequences = [['Let', "'", 's', 'then', 'take', 'a', 'look', 'at', 'the', 'impact', 'that', 'art', 'has', 'posed', 'on', 'individuals', '.', 'It', 'is', 'undeniable', 'that', 'some', 'of', 'the', 'art', 'work', 'may', 'contain', 'negative', 'implications', 'such', 'as', 'insanity', ',', 'violence', ',', 'eroticism', 'etc', '.', 'I', 'absolutely', 'agree', 'that', 'restrictions', 'on', 'exposure', 'to', 'adolescents', 'should', 'apply', 'as', 'adolescents', 'are', 'not', 'mature', 'enough', 'to', 'avoid', 'being', 'led', 'astray', 'by', 'these', 'contents', ';', 'however', ',', 'in', 'the', 'case', 'of', 'adults', 'being', 'the', 'audience', ',', 'there', "'", 's', 'really', 'nothing', 'to', 'hide', '–', 'these', 'negative', 'things', 'objectively', 'exist', 'on', 'this', 'planet', 'and', 'it', 'is', 'impossible', 'for', 'adults', 'to', 'be', 'brain', '-', 'washed', 'only', 'because', 'of', 'some', 'pieces', 'of', 'creative', 'art', '.']]

print('word_sequences', word_sequences)

layer_char_embeddings = LayerCharEmbeddings(gpu=gpu, char_embeddings_dim=char_embeddings_dim,
                                            freeze_char_embeddings=False,
                                            word_len=max_char_pad_len)

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