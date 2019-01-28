"""BiLSTM/BiGRU + char-level CNN tagger model"""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bivanilla import LayerBiVanilla
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_bigru import LayerBiGRU
from src.layers.layer_char_embeddings import LayerCharEmbeddings
from src.layers.layer_char_cnn import LayerCharCNN


class TaggerBiRNNCNN(TaggerBase):
    """TaggerBiRNNCNN is a model for sequences tagging that includes RNN and character-level conv-1D layer."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1,
                 freeze_char_embeddings = False, char_embeddings_dim=25, word_len=20, char_cnn_filter_num=30,
                 char_window_size=3, emb_bert=True):
        super(TaggerBiRNNCNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
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
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        elif rnn_type == 'Vanilla':
            self.birnn_layer = LayerBiVanilla(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss(ignore_index=0)  # "0" target values actually are zero-padded parts of sequences

    def forward(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_char_embed = self.char_embeddings_layer(word_sequences)
        z_char_embed_d = self.dropout(z_char_embed)
        z_char_cnn_d = self.dropout(self.char_cnn_layer(z_char_embed_d))
        z = torch.cat((z_word_embed, z_char_cnn_d), dim=2)
        rnn_output_h = self.birnn_layer(z, mask)
        rnn_output_h_d = self.dropout(rnn_output_h) # shape: batch_size x max_seq_len x rnn_hidden_dim*2
        z_rnn_out = self.apply_mask(self.lin_layer(rnn_output_h_d), mask)
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))
        return y

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss
