from models.tagger_base import TaggerBase
from utils import *

class TaggerBiRNN(TaggerBase):
    """
    TaggerBiRNN is a basic pure recurrent network model for sequences tagging.
    """
    def __init__(self, embeddings_tensor, class_num, rnn_hidden_size=100, freeze_embeddings=False, dropout_ratio=0.5,
                 rnn_type='GRU'):
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
            self.rnn_forward_layer = nn.RNNCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.RNNCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        elif rnn_type == 'LSTM':
            self.rnn_forward_layer = nn.LSTMCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.LSTMCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        elif rnn_type == 'GRU':
            self.rnn_forward_layer = nn.GRUCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
            self.rnn_backward_layer = nn.GRUCell(input_size=self.embeddings.embedding_dim, hidden_size=rnn_hidden_size)
        else:
            raise ValueError('Unknown RNN type! Currently support "Vanilla", "LSTM", "GRU" only.')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=rnn_hidden_size*2, out_features=self.class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, inputs_tensor):
        batch_size, max_seq_len = inputs_tensor.size()
        z_embed = self.embeddings(inputs_tensor)
        z_embed_d = self.dropout1(z_embed)
        rnn_forward_hidden_state = torch.zeros(batch_size, self.rnn_hidden_size)
        rnn_backward_hidden_state = torch.zeros(batch_size, self.rnn_hidden_size)
        rnn_forward_hidden_states_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
        rnn_backward_hidden_states_d = torch.zeros(batch_size, self.rnn_hidden_size, max_seq_len)
        for l in range(max_seq_len):
            n = max_seq_len - l - 1
            curr_rnn_input_forward = z_embed_d[:, l, :]
            curr_rnn_input_backward = z_embed_d[:, n, :]
            rnn_forward_hidden_state = self.rnn_forward_layer(curr_rnn_input_forward, rnn_forward_hidden_state)
            rnn_backward_hidden_state = self.rnn_backward_layer(curr_rnn_input_backward, rnn_backward_hidden_state)
            rnn_forward_hidden_states_d[:, :, l] = self.dropout2(rnn_forward_hidden_state)
            rnn_backward_hidden_states_d[:, :, n] = self.dropout2(rnn_backward_hidden_state)
        outputs_tensor = torch.zeros(batch_size, self.class_num+1, max_seq_len)
        for l in range(max_seq_len):
            rnn_forward_hidden_state_d = rnn_forward_hidden_states_d[:, :, l]
            rnn_backward_hidden_state_d = rnn_backward_hidden_states_d[:, :, l]
            rnn_hidden_state_d = torch.cat((rnn_forward_hidden_state_d, rnn_backward_hidden_state_d), 1)
            z = self.lin_layer(rnn_hidden_state_d)
            y = self.log_softmax_layer(z)
            outputs_tensor[:, :, l] = y
        return outputs_tensor
