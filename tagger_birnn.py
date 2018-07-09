import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class TaggerBiRNN(nn.Module):
    def __init__(self, embeddings_tensor, class_num, rnn_hidden_size=100, freeze_embeddings=False, dropout_ratio=0.5,
                 rnn_type='LSTM'):
        super(TaggerBiRNN, self).__init__()
        self.class_num = class_num
        self.rnn_hidden_size = rnn_hidden_size
        self.freeze_embeddings = freeze_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_embeddings)
        self.dropout1 = torch.nn.Dropout(p=dropout_ratio)
        self.dropout2 = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'Vanilla':
            self.rnn_layer = nn.RNNCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        elif rnn_type == 'LSTM':
            self.rnn_layer = nn.LSTMCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        elif rnn_type == 'GRU':
            self.rnn_layer = nn.GRUCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        else:
            raise ValueError('Unknown RNN type! Currently support "Vanilla", "LSTM", "GRU" only.')
        self.lin_layer = nn.Linear(in_features=rnn_hidden_size, out_features=class_num)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, inputs_batch):
        batch_size, seq_len = inputs_batch.size()
        z_embed = self.embeddings(inputs_batch)
        z_embed_d = self.dropout1(z_embed)
        rnn_forward_hidden_state = torch.zeros(batch_size, self.rnn_hidden_size)
        outputs_batch = torch.zeros(batch_size, self.class_num, seq_len)
        for k in range(seq_len):
            curr_rnn_input = z_embed_d[:, k, :]
            rnn_forward_hidden_state = self.rnn_layer(curr_rnn_input, rnn_forward_hidden_state)
            rnn_forward_hidden_state_d = self.dropout2(rnn_forward_hidden_state)
            z = self.lin_layer(rnn_forward_hidden_state_d)
            y = self.log_softmax_layer(z)
            outputs_batch[:, :, k] = y
        return outputs_batch


