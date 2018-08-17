import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layer_base import LayerBase

class LayerCRF(LayerBase):
    def __init__(self, gpu, states_num):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.START_STATE = 0 # the first one
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        self.transition_matrix.data[self.START_STATE, :] = -100500.

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
        return outputs_tensor
