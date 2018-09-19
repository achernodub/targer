"""
.. module:: TaggerBiRNN
    :synopsis: TaggerBiRNN is a Vanilla recurrent network model for sequences tagging.

.. moduleauthor:: Artem Chernodub
"""

import torch
import torch.nn as nn

from models.tagger_base import TaggerBase
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_bilstm import LayerBiLSTM
from layers.layer_bigru import LayerBiGRU

class TaggerBiRNN(TaggerBase):
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1):
        super(TaggerBiRNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
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
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        #self.nll_loss = nn.NLLLoss(ignore_index=0) # "0" target values actually are zero-padded parts of sequences
        self.nll_loss = nn.NLLLoss()

    def forward(self, word_sequences):
        word_seq_lens = [len(word_seq) for word_seq in word_sequences]
        z_word_embed = self.word_embeddings_layer(word_sequences)

        #print('z_word_embed.shape=', z_word_embed.shape)
        #print(z_word_embed[0, :, 1])
        #print(z_word_embed[1, :, 1])

        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, input_lens=word_seq_lens, pad_idx=self.word_seq_indexer.pad_idx)
        z_rnn_out = self.lin_layer(rnn_output_h)  # shape: batch_size x class_num + 1 x max_seq_len
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))
        #y = self.log_softmax_layer(z_rnn_out)
        return y

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        #loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)

        mask = self.get_mask(word_sequences_train_batch)

        outputs_tensor_train_batch_one_hot = outputs_tensor_train_batch_one_hot.permute(0, 2, 1)

        #print('outputs_tensor_train_batch_one_hot.shape', outputs_tensor_train_batch_one_hot.shape)
        #print('targets_tensor_train_batch.shape', targets_tensor_train_batch.shape)
        #print('mask.shape', mask.shape)
        #print(mask[0, :])
        #print(mask[1, :])

        T = targets_tensor_train_batch.view(-1)
        Y = outputs_tensor_train_batch_one_hot.contiguous().view(-1, self.class_num)
        M = mask.view(-1)

        import numpy as np
        mask_idx = np.argwhere(np.asarray(M))[:, 0]
        #print('mask_idx.shape', mask_idx.shape)

        #print('T.shape', T.shape)
        #print('Y.shape', Y.shape)
        #print('M.shape', M.shape)
        #exit()

        #T_masked = torch.masked_select(T, M.byte())
        #Y_masked = torch.masked_select(Y, M.unsqueeze(-1).expand_as(Y).byte())

        T_masked = T[mask_idx]
        Y_masked = Y[mask_idx, :]
        T_masked = T_masked - self.tensor_ensure_gpu(torch.ones(T_masked.shape[0]).long())

        #print('T_masked.shape', T_masked.shape)
        #print('Y_masked.shape', Y_masked.shape)
        #exit()

        loss = self.nll_loss(Y_masked, T_masked)

        '''outputs_tensor_train_batch = outputs_tensor_train_batch_one_hot.view(-1, self.class_num+1)
        targets_tensor_train_batch = targets_tensor_train_batch.view(-1)
        mask = (targets_tensor_train_batch > 0).float()
        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])
        # pick the values for the label and zero out the rest with the mask
        outputs_tensor_train_batch_one_hot = outputs_tensor_train_batch_one_hot[range(outputs_tensor_train_batch_one_hot.shape[0]), targets_tensor_train_batch] * mask
        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(outputs_tensor_train_batch_one_hot) / nb_tokens'''

        return loss
