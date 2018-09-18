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
            curr_score = torch.logsumexp(curr_score + curr_emission + curr_transition, dim=2)
            score = curr_score * curr_mask + score * (1 - curr_mask)
        score0 = log_sum_exp(score)
        score = torch.logsumexp(score, dim=1)
        diff = (score0 - score).view(-1, 1).sum().item()
        if diff > 0:
            print('BBLL')
            exit()
        return score

    def decode_viterbi(self, features_rnn_compressed, mask): # Viterbi decoding
        batch_num, max_seq_len = mask.shape
        bptr = self.tensor_ensure_gpu(torch.LongTensor())
        score = self.tensor_ensure_gpu(torch.Tensor(batch_num, self.states_num).fill_(-9999.))
        score[:, self.sos_idx] = 0.0
        for n in range(max_seq_len):
            curr_bptr = self.tensor_ensure_gpu(torch.LongTensor())
            curr_score = self.tensor_ensure_gpu(torch.Tensor())
            for curr_state in range(self.states_num):
                max_value = [e.unsqueeze(1) for e in torch.max(score + self.transition_matrix[curr_state], 1)]
                curr_bptr = torch.cat((curr_bptr, max_value[1]), 1)
                curr_score = torch.cat((curr_score, max_value[0]), 1)
            bptr = torch.cat((bptr, curr_bptr.unsqueeze(1)), 1)
            curr_emission = features_rnn_compressed[:, n]
            score = curr_score + curr_emission
        best_score, best_tag = torch.max(score, 1)
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for k in range(batch_num):
            best_tag_k = best_tag[k]
            seq_len = int(mask[k].sum().item())
            for curr_bptr in reversed(bptr[k][:seq_len]):
                best_tag_k = curr_bptr[best_tag_k]
                best_path[k].append(best_tag_k)
            best_path[k].pop()
            best_path[k].reverse()
        return best_path

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))