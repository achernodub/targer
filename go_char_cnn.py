import torch
import torch.nn

from layers.layer_char_embeddings import LayerCharEmbeddings
from layers.layer_char_cnn import LayerCharCNN

print('Hello!')

gpu = -1
char_embeddings_dim = 25
char_cnn_filter_num = 30
char_window_size = 3
word_len = 21

word_sequences = [['Paris', 'hi', '.'], ['London', 'is', 'a', 'capital', 'of', 'Great', 'Britain', '.']]
layer_char_emb = LayerCharEmbeddings(gpu=gpu, char_embeddings_dim=char_embeddings_dim, freeze_char_embeddings=False, word_len=word_len)
char_embeddings_feature = layer_char_emb(word_sequences)
print('char_embeddings_feature.shape =', char_embeddings_feature.shape)

# shape: batch_num x max_seq_len x word_len x char_embeddings_dim

layer_cnn = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size, word_len)

feature = layer_cnn(char_embeddings_feature)

print('feature', feature.shape)

print('The end.')