"""
.. module:: TaggerBiRNNCRF
    :synopsis: TaggerBiRNNCRF is a model for sequences tagging that includes recurrent network + CRF.

.. moduleauthor:: Artem Chernodub
"""

import math

import torch
import torch.nn as nn

from models.tagger_base import TaggerBase
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_bilstm import LayerBiLSTM
from layers.layer_bigru import LayerBiGRU
from layers.layer_crf import LayerCRF

class TaggerBiRNNCRF(TaggerBase):
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1):
        super(TaggerBiRNNCRF, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings_layer.output_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 2)
        self.crf_layer = LayerCRF(gpu, states_num=class_num + 2, pad_idx=tag_seq_indexer.pad_idx, sos_idx=class_num + 1)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def _forward_birnn(self, word_sequences):
        mask = self.get_mask(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        #rnn_output_h_d = self.dropout(rnn_output_h) # shape: batch_size x max_seq_len x rnn_hidden_dim*2
        #features_rnn_compressed = self.lin_layer(rnn_output_h_d) # shape: batch_size x max_seq_len x class_num
        features_rnn_compressed = self.lin_layer(rnn_output_h) # shape: batch_size x max_seq_len x class_num
        return self.apply_mask(features_rnn_compressed, mask)

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        features_rnn = self._forward_birnn(word_sequences_train_batch) # batch_num x max_seq_len x class_num
        mask = self.get_mask(word_sequences_train_batch)  # batch_num x max_seq_len
        numerator = self.crf_layer.numerator(features_rnn, targets_tensor_train_batch, mask)
        denominator = self.crf_layer.denominator(features_rnn, mask)
        nll_loss = -torch.mean(numerator - denominator)
        return nll_loss

    def predict_idx_from_words(self, word_sequences):
        self.eval()
        features_rnn_compressed  = self._forward_birnn(word_sequences)
        mask = self.get_mask(word_sequences)
        idx_sequences = self.crf_layer.decode_viterbi(features_rnn_compressed, mask)
        return idx_sequences

    def predict_tags_from_words(self, word_sequences, batch_size=100):
        print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j])
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                  end='', flush=True)
        return output_tag_sequences

    '''
    def forward(self, word_sequences):
        # outputs_tensor = self.forward(word_sequences)  # batch_num x class_num x max_seq_len
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        # 1 batch
        features_rnn = self._forward_birnn_cnn([word_sequences[0]])
        mask_tensor = self.tensor_ensure_gpu(torch.Tensor(1, len(word_sequences[0])).fill_(1))
        outputs_tensor = self.crf(features_rnn, mask_tensor)
        #outputs_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, self.class_num + 1, max_seq_len))
        #for n, word_seq in enumerate(word_sequences):
        #    features_rnn = self._forward_birnn_cnn([word_seq])  # batch_num x max_seq_len x class_num
        #    mask_tensor = self.tensor_ensure_gpu(torch.Tensor(1, len(word_seq)).fill_(1))
        #    y = self.crf(features_rnn, mask_tensor)
        #    outputs_tensor[n, :, :len(word_seq)] = y
        #print('outputs_tensor.shape', outputs_tensor.shape)
        #exit()
        return outputs_tensor
    '''

