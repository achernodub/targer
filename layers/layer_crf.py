from math import log

import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerCRF(LayerBase):
    def __init__(self, gpu, states_num, pad_idx, sos_idx, tag_seq_indexer, verbose=True):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.tag_seq_indexer = tag_seq_indexer
        self.tag_seq_indexer.add_tag('<sos>')
        self.verbose = verbose
        # Transition matrix contains log probabilities from state j to state i
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num, dtype=torch.float))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        # Default initialization
        self.transition_matrix.data[self.sos_idx, :] = -9999.0
        self.transition_matrix.data[:, self.pad_idx] = -9999.0
        self.transition_matrix.data[self.pad_idx, :] = -9999.0
        self.transition_matrix.data[self.pad_idx, self.pad_idx] = 0.0

    def get_empirical_transition_matrix(self, tag_sequences_train, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        empirical_transition_matrix = torch.zeros(self.states_num, self.states_num, dtype=torch.long)
        for tag_seq in tag_sequences_train:
            for n, tag in enumerate(tag_seq):
                if n + 1 >= len(tag_seq):
                    break
                next_tag = tag_seq[n + 1]
                j = tag_seq_indexer.item2idx_dict[tag]
                i = tag_seq_indexer.item2idx_dict[next_tag]
                empirical_transition_matrix[i, j] += 1
        return empirical_transition_matrix

    def init_transition_matrix_empirical(self, tag_sequences_train):
        # Calculate statistics for tag transitions
        empirical_transition_matrix = self.get_empirical_transition_matrix(tag_sequences_train)
        # Initialize
        for i in range(self.tag_seq_indexer.get_items_count()):
            for j in range(self.tag_seq_indexer.get_items_count()):
                if empirical_transition_matrix[i, j] == 0:
                    self.transition_matrix.data[i, j] = -9999.0
                #self.transition_matrix.data[i, j] = torch.log(empirical_transition_matrix[i, j].float() + 10**-32)
        if self.verbose:
            print('Empirical transition matrix from the train dataset:')
            self.pretty_print_transition_matrix(empirical_transition_matrix)
            print('\nInitialized transition matrix:')
            self.pretty_print_transition_matrix(self.transition_matrix.data)

    def pretty_print_transition_matrix(self, transition_matrix, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        str = '%10s' % ''
        for i in range(tag_seq_indexer.get_items_count()):
            str += '%10s' % tag_seq_indexer.idx2item_dict[i]
        str += '\n'
        for i in range(tag_seq_indexer.get_items_count()):
            str += '\n%10s' % tag_seq_indexer.idx2item_dict[i]
            for j in range(tag_seq_indexer.get_items_count()):
                str += '%10s' % ('%1.1f' % transition_matrix[i, j])
        print(str)

    def is_cuda(self):
        return self.transition_matrix.is_cuda

    def numerator(self, features_rnn_compressed, states_tensor, mask_tensor):
        #
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        #
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
        start_states_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, 1, dtype=torch.long).fill_(self.sos_idx))
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1)
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n]
            curr_emission = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            curr_transition = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            for k in range(batch_num):
                curr_emission[k] = features_rnn_compressed[k, n, states_tensor[k, n + 1]].unsqueeze(0)
                curr_states_seq = states_tensor[k]
                curr_transition[k] = self.transition_matrix[curr_states_seq[n + 1], curr_states_seq[n]].unsqueeze(0)
            score = score + curr_emission*curr_mask + curr_transition*curr_mask
        return score

    def denominator(self, features_rnn_compressed, mask_tensor): # forward algorithm
        # features_rnn_compressed: batch x max_seq_len x tags_num
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, self.states_num, dtype=torch.float).fill_(-9999.0))
        score[:, self.sos_idx] = 0.
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n].unsqueeze(-1).expand_as(score)
            curr_score = score.unsqueeze(1).expand(-1, *self.transition_matrix.size())
            curr_emission = features_rnn_compressed[:, n].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transition_matrix.unsqueeze(0).expand_as(curr_score)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)
        score = log_sum_exp(score)
        return score

    def save_debug(self, no, X):
        if no == 9:
            #print('\nСС SINGLE', X[:, :, :])
            torch.save(X[:, 0, :], 'a.hdf5')
        elif no == -1:
            #print('\nСС BATCH', X[9, :, :])
            torch.save(X[9, 0, :], 'b.hdf5')

    # curr_mask = mask_tensor[:, n].unsqueeze(-1).expand_as(score)
    def decode_viterbi(self, features_rnn_compressed, mask_tensor, no): # Viterbi decoding
        #print('A')
        batch_size, max_seq_len = mask_tensor.shape
        backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, max_seq_len, self.states_num))
        score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num).fill_(-9999.))
        score[:, self.sos_idx] = 0.0
        for n in range(max_seq_len):
            curr_emissions = features_rnn_compressed[:, n]
            curr_score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num))
            curr_backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, self.states_num))
            for curr_state in range(self.states_num):
                T = self.transition_matrix[curr_state, :].unsqueeze(0).expand(batch_size, self.states_num)
                max_values, max_indices = torch.max(score + T, 1)
                curr_score[:, curr_state] = max_values
                curr_backpointers[:, curr_state] = max_indices
            curr_mask = mask_tensor[:, n].unsqueeze(1).expand(batch_size, self.states_num)
            score = score * (1 - curr_mask) + (curr_score + curr_emissions) * curr_mask
            backpointers[:, n, :] = curr_backpointers
        best_score, best_tag = torch.max(score, 1)
        #print('B')
        # Find best sequence
        seq_len_list = [int(scalar(mask_tensor[k].sum())) for k in range(batch_size)]
        backpointers_list = backpointers.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for k in range(batch_size):
            best_tag_k = best_tag[k]
            seq_len = seq_len_list[k]
            for curr_backpointers in reversed(backpointers_list[k][:seq_len]):
                best_tag_k = curr_backpointers[best_tag_k]
                best_path[k].append(best_tag_k)
            best_path[k].pop()
            best_path[k].reverse()
        #print('C')
        #if no == 9:
        #    print('best_path SINGLE', best_path[0])
        #elif no == -1:
        #    print('best path BATCH', best_path[9])
        return best_path

CUDA = True

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x): # for 1D tensor
    return scalar(torch.max(x, 0)[1])

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))