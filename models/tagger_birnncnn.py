import torch
import torch.nn as nn

from models.tagger_base import TaggerBase
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_bilstm import LayerBiLSTM
from layers.layer_bigru import LayerBiGRU
from layers.layer_char_embeddings import LayerCharEmbeddings
from layers.layer_char_cnn import LayerCharCNN

class TaggerBiRNNCNN(TaggerBase):
    """
    TaggerBiRNNCNN is a model for sequences tagging that includes recurrent network and character-level conv-1D layer.
    """
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, rnn_hidden_dim=100, freeze_word_embeddings=False,
                 dropout_ratio=0.5, rnn_type='GRU', gpu=-1, freeze_char_embeddings = False, char_embeddings_dim=25,
                 max_char_pad_len=20, char_cnn_filter_num=30, char_window_size=3):
        super(TaggerBiRNNCNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu)
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.freeze_char_embeddings = freeze_char_embeddings
        self.char_embeddings_dim = char_embeddings_dim
        self.max_char_pad_len = max_char_pad_len
        self.char_cnn_filter_num = char_cnn_filter_num
        self.char_window_size = char_window_size
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.char_embeddings_layer = LayerCharEmbeddings(gpu, char_embeddings_dim, freeze_char_embeddings,
                                                         max_char_pad_len)
        self.char_cnn_layer = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size,
                                           max_char_pad_len)
        self.dropout1 = torch.nn.Dropout(p=dropout_ratio)
        self.dropout2 = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def forward(self, word_sequences):
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_char_embed = self.char_embeddings_layer(word_sequences)
        z_char_cnn = self.char_cnn_layer(z_char_embed)
        z = torch.cat((z_word_embed, z_char_cnn), dim=2)
        z_d = self.dropout1(z)
        rnn_output_h = self.birnn_layer(z_d)
        rnn_output_h_d = self.dropout2(rnn_output_h) # shape: batch_size x max_seq_len x class_num + 1
        z_out = self.lin_layer(rnn_output_h_d).permute(0, 2, 1) # shape: batch_size x class_num + 1 x max_seq_len
        return self.log_softmax_layer(z_out)