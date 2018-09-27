"""
.. module:: TaggerBiRNNCNNCRF
    :synopsis: TaggerBiRNNCNNCRF is a model for sequences tagging that includes recurrent network + conv layer + CRF.

.. moduleauthor:: Artem Chernodub
"""

import math

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
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1,
                 freeze_char_embeddings = False, char_embeddings_dim=25, word_len=20, char_cnn_filter_num=30,
                 char_window_size=3):
        super(TaggerBiRNNCNNCRF, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
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
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 2)
        self.crf_layer = LayerCRF(gpu, states_num=class_num + 2, pad_idx=tag_seq_indexer.pad_idx, sos_idx=class_num + 1,
                                  tag_seq_indexer=tag_seq_indexer)
        self.softmax = nn.Softmax(dim=2)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def _forward_birnn(self, word_sequences):
        mask = self.get_mask(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_char_embed_d = self.dropout(self.char_embeddings_layer(word_sequences))
        z_char_cnn = self.char_cnn_layer(z_char_embed_d)
        z_d = self.dropout(torch.cat((z_word_embed, z_char_cnn), dim=2))
        rnn_output_h_d = self.dropout(self.apply_mask(self.birnn_layer(z_d, mask), mask))
        features_rnn_compressed = self.lin_layer(rnn_output_h_d)
        return self.apply_mask(features_rnn_compressed, mask)

    '''def _forward_birnn(self, word_sequences):
        mask = self.get_mask(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        z_char_embed = self.char_embeddings_layer(word_sequences)
        z_char_embed_d = self.dropout(z_char_embed)
        z_char_cnn = self.char_cnn_layer(z_char_embed_d)
        z = torch.cat((z_word_embed_d, z_char_cnn), dim=2)
        rnn_output_h = self.apply_mask(self.birnn_layer(z, mask), mask)
        features_rnn_compressed = self.lin_layer(rnn_output_h)
        return self.apply_mask(features_rnn_compressed, mask)'''

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        features_rnn = self._forward_birnn(word_sequences_train_batch) # batch_num x max_seq_len x class_num
        mask = self.get_mask(word_sequences_train_batch)  # batch_num x max_seq_len
        numerator = self.crf_layer.numerator(features_rnn, targets_tensor_train_batch, mask)
        denominator = self.crf_layer.denominator(features_rnn, mask)
        nll_loss = -torch.mean(numerator - denominator)
        return nll_loss

    def predict_idx_from_words(self, word_sequences, no=-1):
        self.eval()
        features_rnn_compressed_masked  = self._forward_birnn(word_sequences)
        mask = self.get_mask(word_sequences)
        idx_sequences = self.crf_layer.decode_viterbi(features_rnn_compressed_masked, mask)
        #if no == 9:
        #    print('\nBB features_rnn_compressed_masked SINGLE', features_rnn_compressed_masked[:, :, :])
        #    print('\nfeatures_rnn_compressed_masked.shape SINGLE', features_rnn_compressed_masked.shape)
        #    print('\nidx_sequences SINGLE', idx_sequences)
        #elif no == -1:
        #    print('\nBB features_rnn_compressed_masked', features_rnn_compressed_masked[9, :, :])
        #    print('\nfeatures_rnn_compressed_masked.shape', features_rnn_compressed_masked.shape)
        #    print('\nidx_sequences', idx_sequences[9])
        return idx_sequences

    def __save_debug(self, no, X):
        if no == 9:
            #print('\nСС SINGLE', X[:, :, :])
            torch.save(X[:, 0, :], 'a.hdf5')
        elif no == -1:
            #print('\nСС BATCH', X[9, :, :])
            torch.save(X[9, 0, :], 'b.hdf5')

    def predict_tags_from_words(self, word_sequences, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            if batch_size == 1:
                curr_output_idx = self.predict_idx_from_words(word_sequences[i:j], n)
            else:
                curr_output_idx = self.predict_idx_from_words(word_sequences[i:j], -1)
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                  end='', flush=True)
        return output_tag_sequences
