import torch
import torch.nn as nn

from layers.layer_base import LayerBase

class LayerCRF(LayerBase):
    def __init__(self, gpu, states_num, pad_idx, sos_idx):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num, dtype=torch.float)) # transition scores from j to i
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        self.transition_matrix.data[sos_idx, :] = -9999.0
        self.transition_matrix.data[:, pad_idx] = -9999.0
        self.transition_matrix.data[pad_idx, :] = -9999.0
        self.transition_matrix.data[pad_idx, pad_idx] = 0.0

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

    def decode_viterbi(self, features_rnn_compressed, mask): # Viterbi decoding
        batch_num, max_seq_len = mask.shape
        bptr = self.tensor_ensure_gpu(torch.LongTensor())
        score = self.tensor_ensure_gpu(torch.Tensor(batch_num, self.states_num).fill_(-9999.))
        score[:, self.sos_idx] = 0.0
        for n in range(max_seq_len):
            curr_bptr = LongTensor()
            curr_score = Tensor()
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
            seq_len = int(scalar(mask[k].sum()))
            for curr_bptr in reversed(bptr[k][:seq_len]):
                best_tag_k = curr_bptr[best_tag_k]
                best_path[k].append(best_tag_k)
            best_path[k].pop()
            best_path[k].reverse()
        return best_path

'''
    def numerator0(self, features_input_tensor, states_tensor, mask_tensor):
        #
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        #
        batch_num, max_seq_len = mask_tensor.shape
        start_states_tensor = self.tensor_ensure_gpu(torch.LongTensor(batch_num, 1).fill_(self.sos_idx))
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1) # batch_num x max_seq_len + 1
        scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
        for k in range(max_seq_len):
            curr_emission_scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            curr_transition_scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            for n in range(batch_num):
                curr_state = states_tensor[n, k].item()
                next_state = states_tensor[n, k + 1].item()
                curr_emission_scores_batch[n] = features_input_tensor[n, k, next_state]
                curr_transition_scores_batch[n] = self.transition_matrix[next_state, curr_state]
            curr_mask = mask_tensor[:, k]
            scores_batch += curr_emission_scores_batch * curr_mask + curr_transition_scores_batch * curr_mask
        return scores_batch

    def denominator0(self, features_rnn_compressed, mask): # forward algorithm
        batch_num, max_seq_len = mask.shape
        # y = batch x max_seq_len x tags_num = 11 x 263 x 10
        # initialize forward variables in log space
        score = Tensor(batch_num, self.states_num).fill_(-10000.)
        score[:, self.sos_idx] = 0.
        for t in range(features_rnn_compressed.size(1)): # iterate through the sequence
            mask_t = mask[:, t].unsqueeze(-1).expand_as(score)
            score_t = score.unsqueeze(1).expand(-1, *self.transition_matrix.size())
            emit = features_rnn_compressed[:, t].unsqueeze(-1).expand_as(score_t)
            trans = self.transition_matrix.unsqueeze(0).expand_as(score_t)
            score_t = log_sum_exp(score_t + emit + trans)
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score)
        return score # partition function

    def denominator1(self, features_input_tensor, mask_tensor):
        #
        # features_input_tensor: batch_num x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        #
        batch_num, max_seq_len = mask_tensor.shape
        scores = self.tensor_ensure_gpu(torch.zeros(batch_num, self.states_num, dtype=torch.float))
        for k in range(max_seq_len):
            curr_scores = scores.unsqueeze(1).expand(batch_num, self.states_num, self.states_num)
            curr_emission_scores_batch = features_input_tensor[:, k].unsqueeze(-1).expand(batch_num, self.states_num,
                                                                                          self.states_num)
            curr_transition_scores_batch = self.transition_matrix.unsqueeze(0).expand(batch_num, self.states_num,
                                                                                      self.states_num)
            curr_mask = mask_tensor[:, k].unsqueeze(-1).expand(batch_num, self.states_num)
            curr_scores = torch.logsumexp(curr_scores + curr_emission_scores_batch + curr_transition_scores_batch, dim=2)
            scores = curr_scores * curr_mask + scores * (1 - curr_mask)
        return torch.logsumexp(scores, 1)
'''


    #def forward(self, y, mask):
    #    # y = batch x max_seq_len x tags_num
    #    batch_num, max_seq_len, _ = y.shape
    #    idx_sequences = self.decode(y, mask)
    #    out = torch.zeros(1, self.num_tags, max_seq_len)  # # batch_size x num_class+1 x max_seq_len
    #    for idx_seq in idx_sequences:
    #        for k, idx in enumerate(idx_seq):
    #            out[0, idx, k] = 1
    #    out.cuda()
    #    return out


''' def __init__(self, gpu, states_num):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.START_STATE = 0 # the first one
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        self.transition_matrix.data[self.START_STATE, :] = -10000.

    def is_cuda(self):
        return self.transition_matrix.is_cuda

    def get_neg_loglikelihood(self, features_input_tensor, targets_tensor):
        mask_tensor = targets_tensor.gt(0).float() # shape: batch_num x max_seq_len
        log_likelihood_numerator = self._compute_loglikelihood_numerator(features_input_tensor, targets_tensor, mask_tensor)
        log_likelihood_denominator = self._compute_log_likelihood_denominator(features_input_tensor, mask_tensor)
        return -torch.sum(log_likelihood_numerator - log_likelihood_denominator)

    def _compute_loglikelihood_numerator(self, features_input_tensor, states_tensor, mask_tensor):
        #
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        #
        batch_num, max_seq_len = mask_tensor.shape
        start_states_tensor = self.tensor_ensure_gpu(torch.LongTensor(batch_num, 1).fill_(self.START_STATE))
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1) # batch_num x max_seq_len + 1
        scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
        for k in range(max_seq_len):
            curr_emission_scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            curr_transition_scores_batch = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            for n in range(batch_num):
                curr_state = states_tensor[n, k].item()
                next_state = states_tensor[n, k + 1].item()
                curr_emission_scores_batch[n] = features_input_tensor[n, k, next_state]
                curr_transition_scores_batch[n] = self.transition_matrix[next_state, curr_state]
            curr_mask = mask_tensor[:, k]
            scores_batch += curr_emission_scores_batch * curr_mask + curr_transition_scores_batch * curr_mask
        return scores_batch

    def _compute_log_likelihood_denominator(self, features_input_tensor, mask_tensor):
        #
        # features_input_tensor: batch_num x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        #
        batch_num, max_seq_len = mask_tensor.shape
        scores = self.tensor_ensure_gpu(torch.Tensor(batch_num, self.states_num).fill_(-100500.))
        for k in range(max_seq_len):
            curr_scores = scores.unsqueeze(1).expand(batch_num, self.states_num, self.states_num)
            curr_emission_scores_batch = features_input_tensor[:, k].unsqueeze(-1).expand(batch_num, self.states_num,
                                                                                          self.states_num)
            curr_transition_scores_batch = self.transition_matrix.unsqueeze(0).expand(batch_num, self.states_num,
                                                                                      self.states_num)
            curr_mask = mask_tensor[:, k].unsqueeze(-1).expand(batch_num, self.states_num)
            curr_scores = torch.logsumexp(curr_scores + curr_emission_scores_batch + curr_transition_scores_batch, dim=2)
            scores = curr_scores * curr_mask + scores * (1 - curr_mask)
        return torch.logsumexp(scores, 1)

    def _decode_viterbi(self, features_input_tensor, mask_tensor):
        # initialize backpointers and viterbi variables in log space
        batch_num, max_seq_len = mask_tensor.shape
        bptr = self.tensor_ensure_gpu(torch.LongTensor())
        score = self.tensor_ensure_gpu(torch.Tensor(batch_num, self.states_num).fill_(-100500.))
        score[:, self.START_STATE] = 0.
        for k in range(max_seq_len):
            curr_bptr = self.tensor_ensure_gpu(torch.LongTensor())
            curr_score = self.tensor_ensure_gpu(torch.Tensor())
            for i in range(self.states_num): # for each next tag
                m = [e.unsqueeze(1) for e in torch.max(score + self.transition_matrix[i], 1)]
                curr_bptr = torch.cat((curr_bptr, m[1]), 1) # best previous tags
                curr_score = torch.cat((curr_score, m[0]), 1) # best transition scores
            bptr = torch.cat((bptr, curr_bptr.unsqueeze(1)), 1)
            score = curr_score + features_input_tensor[:, k] # plus emission scores
        best_score, best_tag = torch.max(score, 1)
        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for n in range(batch_num):
            x = best_tag[n] # best tag
            l = int((mask_tensor[n].sum()).item())
            for curr_bptr in reversed(bptr[n][:l]):
                x = curr_bptr[x]
                best_path[n].append(x)
            best_path[n].pop()
            best_path[n].reverse()
        return best_path

    def _viterbi_most_likely_states(self, inputs_seq, seq_len):
        argmaxes = list()
        # 1) Init stage
        prev_delta = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
        for start_state in range(0, self.states_num):
            prev_delta[start_state] = self._score_func(self.START_STATE, start_state, inputs_seq, k=0)
        # 2) Regular stage
        for k in range(1, seq_len):
            local_argmaxes = list()
            curr_delta = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
            for curr_state in range(self.states_num):
                scores = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
                for next_state in range(self.states_num):
                    scores[next_state] = prev_delta[curr_state] + self._score_func(curr_state, next_state, inputs_seq, k)
                local_argmaxes.append(scores.argmax().item())
                curr_delta[curr_state] = scores.max().item()
            argmaxes.append(local_argmaxes)
            prev_delta = curr_delta
        # 3) Find the most likely sequence
        final_state = prev_delta.argmax().item()
        most_likely_states = [final_state]
        for states in reversed(argmaxes):
            final_state = states[final_state]
            most_likely_states.append(final_state)
        return most_likely_states

    def forward(self, features_input_tensor, mask_tensor):
        idx_sequences = self._decode_viterbi(features_input_tensor, mask_tensor)
        y = self.idx2tensor(idx_sequences)
        return y

    def idx2tensor(self, idx_sequences):
        batch_num = len(idx_sequences)
        seq_lens = [len(seq) for seq in idx_sequences]
        max_seq_len = max(seq_lens)
        class_num = self.states_num
        outputs_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, class_num, max_seq_len, dtype=torch.float))
        for n in range(batch_num):
            for seq_len in range(seq_lens[n]):
                for k in range(seq_len):
                    c = idx_sequences[n][k]
                    outputs_tensor[n, c, k] = 1
        return outputs_tensor'''

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