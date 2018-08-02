import torch
import torch.nn as nn

from models.tagger_base import TaggerBase
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_bilstm import LayerBiLSTM
from layers.layer_bigru import LayerBiGRU

from classes.utils import *


class TaggerBiRNN(TaggerBase):
    """
    TaggerBiRNN is a basic pure recurrent network model for sequences tagging.
    """
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, rnn_hidden_dim=100, freeze_word_embeddings=False, dropout_ratio=0.5,
                 rnn_type='GRU', gpu=-1):
        super(TaggerBiRNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu)
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings = LayerWordEmbeddings(word_seq_indexer, freeze_word_embeddings)
        self.dropout1 = torch.nn.Dropout(p=dropout_ratio)
        self.dropout2 = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings.embeddings_dim, hidden_dim=rnn_hidden_dim, gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings.embeddings_dim, hidden_dim=rnn_hidden_dim, gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def forward(self, word_sequences):
        z_embed = self.word_embeddings(word_sequences)
        z_embed_d = self.dropout1(z_embed)
        rnn_output_h = self.birnn_layer(z_embed_d)
        rnn_output_h_d = self.dropout2(rnn_output_h) # shape: batch_size x max_seq_len x class_num + 1
        z = self.lin_layer(rnn_output_h_d).permute(0, 2, 1) # shape: batch_size x class_num + 1 x max_seq_len
        return self.log_softmax_layer(z)