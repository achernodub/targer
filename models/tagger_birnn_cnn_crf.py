import torch
import torch.nn as nn

from models.tagger_base import TaggerBase
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_bilstm import LayerBiLSTM
from layers.layer_bigru import LayerBiGRU
from layers.layer_char_embeddings import LayerCharEmbeddings
from layers.layer_char_cnn import LayerCharCNN
from layers.layer_crf import LayerCRF

class TaggerBiRNNCNNCRF(TaggerBase):
    """
    TaggerBiRNNCNN is a model for sequences tagging that includes recurrent network + char conv1D + CRF.
    """
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, rnn_hidden_dim=100, freeze_word_embeddings=False,
                 dropout_ratio=0.5, rnn_type='GRU', gpu=-1, freeze_char_embeddings = False, char_embeddings_dim=25,
                 word_len=20, char_cnn_filter_num=30, char_window_size=3):
        super(TaggerBiRNNCNNCRF, self).__init__(word_seq_indexer, tag_seq_indexer, gpu)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.freeze_char_embeddings = freeze_char_embeddings
        self.char_embeddings_dim = char_embeddings_dim
        self.word_len = word_len
        self.char_cnn_filter_num = char_cnn_filter_num
        self.char_window_size = char_window_size
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.char_embeddings_layer = LayerCharEmbeddings(gpu, char_embeddings_dim, freeze_char_embeddings,
                                                         word_len, word_seq_indexer.get_unique_characters_list())
        self.char_cnn_layer = LayerCharCNN(gpu, char_embeddings_dim, char_cnn_filter_num, char_window_size,
                                           word_len)
        self.layer_crf = LayerCRF(gpu, states_num=class_num)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
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
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num)
        #self.log_softmax_layer = nn.LogSoftmax(dim=1) ###############################################################
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def _forward_birnn_cnn(self, word_sequences):
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        z_char_embed = self.char_embeddings_layer(word_sequences)
        z_char_embed_d = self.dropout(z_char_embed)
        z_char_cnn = self.char_cnn_layer(z_char_embed_d)
        z = torch.cat((z_word_embed_d, z_char_cnn), dim=2)
        rnn_output_h = self.birnn_layer(z)
        rnn_output_h_d = self.dropout(rnn_output_h) # shape: batch_size x max_seq_len x rnn_hidden_dim*2
        features_rnn = self.lin_layer(rnn_output_h_d) # shape: batch_size x max_seq_len x class_num + 1
        return features_rnn

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        targets_sequences_train_idx = self.tag_seq_indexer.elements2idx(tag_sequences_train_batch)
        features_rnn = self._forward_birnn_cnn(word_sequences_train_batch) # batch_num x max_seq_len x class_num
        neg_log_likelihood = self.layer_crf.get_neg_loglikelihood(features_rnn, targets_sequences_train_idx)
        return neg_log_likelihood

    def forward(self, word_sequences):
        # outputs_tensor = self.forward(word_sequences)  # batch_size x num_class+1 x max_seq_len
        seq_lens = [len(word_seq) for word_seq in word_sequences]
        features_rnn = self._forward_birnn_cnn(word_sequences)  # batch_num x max_seq_len x class_num
        y = self.layer_crf(features_rnn, seq_lens)
        return y
