from models.tagger_base import TaggerBase
from utils import *

class TaggerBiRNN(TaggerBase):
    """
    TaggerBiRNN is a basic pure recurrent network model for sequences tagging.
    """
    def __init__(self, embeddings_tensor, class_num, rnn_hidden_size=100, freeze_embeddings=False, dropout_ratio=0.5,
                 rnn_type='GRU',gpu=-1):
        super(TaggerBiRNN, self).__init__()
        self.class_num = class_num
        self.rnn_hidden_size = rnn_hidden_size
        self.freeze_embeddings = freeze_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_embeddings)
        self.dropout1 = torch.nn.Dropout(p=dropout_ratio)
        self.dropout2 = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'Vanilla':
            self.rnn_forward_layer = nn.RNNCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.RNNCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.lin_layer = nn.Linear(in_features=rnn_hidden_size * 2, out_features=self.class_num + 1)
        elif rnn_type == 'LSTM':
            self.rnn_forward_layer = nn.LSTMCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.LSTMCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.lin_layer = nn.Linear(in_features=rnn_hidden_size * 4, out_features=self.class_num + 1)
        elif rnn_type == 'GRU':
            self.rnn_forward_layer = nn.GRUCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.GRUCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.lin_layer = nn.Linear(in_features=rnn_hidden_size * 2, out_features=self.class_num + 1)
        else:
            raise ValueError('Unknown RNN type! Currently support "Vanilla", "LSTM", "GRU" only.')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def forward(self, inputs_tensor):
        batch_size, max_seq_len = inputs_tensor.shape
        # init states by zeros
        rnn_forward_h = torch.zeros(batch_size, self.rnn_hidden_size)
        rnn_backward_h = torch.zeros(batch_size, self.rnn_hidden_size)
        rnn_forward_h_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
        rnn_backward_h_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
        outputs_tensor = torch.zeros(batch_size, self.class_num+1, max_seq_len)
        if self.rnn_type == 'LSTM': # for LSTM we have two types of states
            rnn_forward_c = torch.zeros(batch_size, self.rnn_hidden_size)
            rnn_backward_c = torch.zeros(batch_size, self.rnn_hidden_size)
            rnn_forward_c_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
            rnn_backward_c_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
        if self.gpu >= 0:
            rnn_forward_h = rnn_forward_h.cuda(device=self.gpu)
            rnn_backward_h = rnn_backward_h.cuda(device=self.gpu)
            rnn_forward_h_d = rnn_forward_h_d.cuda(device=self.gpu)
            rnn_backward_h_d = rnn_backward_h_d.cuda(device=self.gpu)
            outputs_tensor = outputs_tensor.cuda(device=self.gpu)
            if self.rnn_type == 'LSTM':
                rnn_forward_c = rnn_forward_c.cuda(device=self.gpu)
                rnn_backward_c = rnn_backward_c.cuda(device=self.gpu)
                rnn_forward_c_d = rnn_forward_c_d.cuda(device=self.gpu)
                rnn_backward_c_d = rnn_backward_c_d.cuda(device=self.gpu)
        # Forward pass for embeddings layer
        z_embed = self.embeddings(inputs_tensor)
        z_embed_d = self.dropout1(z_embed)
        # Forward pass for recurrent layer
        for l in range(max_seq_len):
            n = max_seq_len - l - 1
            curr_rnn_input_forward = z_embed_d[:, l, :]
            curr_rnn_input_backward = z_embed_d[:, n, :]
            if self.rnn_type == 'Vanilla' or self.rnn_type == 'GRU':
                rnn_forward_h = self.rnn_forward_layer(curr_rnn_input_forward, rnn_forward_h)
                rnn_backward_h = self.rnn_backward_layer(curr_rnn_input_backward, rnn_backward_h)
                rnn_forward_h_d[:, :, l] = self.dropout2(rnn_forward_h)
                rnn_backward_h_d[:, :, n] = self.dropout2(rnn_backward_h)
            elif self.rnn_type == 'LSTM':
                rnn_forward_h, rnn_forward_c = self.rnn_forward_layer(curr_rnn_input_forward, (rnn_forward_h, rnn_forward_c))
                rnn_backward_h, rnn_backward_c = self.rnn_backward_layer(curr_rnn_input_backward, (rnn_backward_h, rnn_backward_c))
                rnn_forward_h_d[:, :, l] = self.dropout2(rnn_forward_h)
                rnn_backward_h_d[:, :, n] = self.dropout2(rnn_backward_h)
                rnn_forward_c_d[:, :, l] = self.dropout2(rnn_forward_c)
                rnn_backward_c_d[:, :, n] = self.dropout2(rnn_backward_c)
            else:
                raise ValueError('Unknown type of recurrent layer')
        # Forward pass for output layer
        for l in range(max_seq_len):
            curr_rnn_forward_h_d = rnn_forward_h_d[:, :, l]
            curr_rnn_backward_h_d = rnn_backward_h_d[:, :, l]
            if self.rnn_type == 'Vanilla' or self.rnn_type == 'GRU':
                rnn_hidden_state_d = torch.cat((curr_rnn_forward_h_d, curr_rnn_backward_h_d), 1)
            elif self.rnn_type == 'LSTM':
                curr_rnn_forward_c_d = rnn_forward_c_d[:, :, l]
                curr_rnn_backward_c_d = rnn_backward_c_d[:, :, l]
                rnn_hidden_state_d = torch.cat((curr_rnn_forward_h_d, curr_rnn_backward_h_d, curr_rnn_forward_c_d, curr_rnn_backward_c_d), 1)
            z = self.lin_layer(rnn_hidden_state_d)
            y = self.log_softmax_layer(z)
            outputs_tensor[:, :, l] = y
        return outputs_tensor
