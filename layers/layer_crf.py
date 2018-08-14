import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layer_base import LayerBase

class LayerCRF(LayerBase):
    def __init__(self, gpu, states_num):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.START_STATE = states_num # the last one
        self.transition_matrix = nn.Parameter(torch.zeros(states_num + 1, states_num))
        nn.init.normal_(self.transition_matrix, -1, 0.1)

    def _score_func(self, prev_state, curr_state, inputs, k):
        # inputs: max_seq_len x states_num
        # 0 <= k < seq_len
        return self.transition_matrix[prev_state, curr_state] + inputs[k, curr_state]

    def _log_sum_exp(self, inputs): # inputs: states_num
        return (inputs - F.log_softmax(inputs, dim=0)).mean(dim=0, keepdim=False)

    def _compute_log_likelihood_numerator(self, curr_inputs_tensor, states):
        # inputs: max_seq_len x states_num
        # states: seq_len
        prev_state = self.START_STATE
        score = torch.Tensor([0])
        score = self.tensor_ensure_gpu(score)
        for k, curr_state in enumerate(states):
            score += self._score_func(prev_state, curr_state, curr_inputs_tensor, k)
            prev_state = curr_state
        return score

    def _compute_log_likelihood_denominator(self, inputs_seq, seq_len):
        # alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * T(y_t | y{t-1} = i)
        # inputs: max_seq_len x states_num
        # 1) Init stage
        prev_alpha = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
        for start_state in range(0, self.states_num):
            prev_alpha[start_state] = self._score_func(self.START_STATE, start_state, inputs_seq, k=0)
        # 2) Regular stage
        for k in range(1, seq_len):
            curr_alpha = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
            for curr_state in range(self.states_num):
                scores = self.tensor_ensure_gpu(torch.zeros(self.states_num, dtype=torch.float))
                for next_state in range(self.states_num):
                    scores[next_state] = prev_alpha[curr_state] + self._score_func(curr_state, next_state, inputs_seq, k)
                curr_alpha[curr_state] = self._log_sum_exp(scores)
            prev_alpha = curr_alpha
        return self._log_sum_exp(prev_alpha)

    def _compute_neg_loglikelihood(self, curr_inputs_tensor, states):
        seq_len = len(states)
        log_likelihood_numerator = self._compute_log_likelihood_numerator(curr_inputs_tensor, states)
        log_likelihood_denominator = self._compute_log_likelihood_denominator(curr_inputs_tensor, seq_len)
        return -log_likelihood_numerator + log_likelihood_denominator

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

    def is_cuda(self):
        return self.transition_matrix.is_cuda

    def idx2states(self, sequences_idx): # idx are in range [1 .. class_num], whereas states in range [0 .. class_num-1]
        return [[i - 1 for i in idx_seq] for idx_seq in sequences_idx]

    def states2idx(self, sequences_states): # idx are in range [1 .. class_num], whereas states in range [0 .. class_num-1]
        return [[i + 1 for i in states] for states in sequences_states]

    def get_neg_loglikelihood(self, features_input_tensor, targets_sequences_idx):
        # By default, in targets_idx tags are from 1 to class_num, because 0 is reserved for <pad> words.
        # Here for convenience we temporarily make them from 0 to class_num -1
        sequences_states = self.idx2states(targets_sequences_idx)
        neg_loglikelihood = self.tensor_ensure_gpu(torch.Tensor([0]))
        for n, states in enumerate(sequences_states):
            curr_input_tensor = features_input_tensor[n, :, :]
            neg_loglikelihood += self._compute_neg_loglikelihood(curr_input_tensor, states)
        return neg_loglikelihood

    def forward(self, features_input_tensor, seq_lens):
        states_sequences = list()
        for n, seq_len in enumerate(seq_lens):
            curr_input_tensor = features_input_tensor[n, :, :]
            output_states = self._viterbi_most_likely_states(curr_input_tensor, seq_len)
            states_sequences.append(output_states)
        idx_sequences = self.states2idx(states_sequences)
        y = self.idx2tensor(idx_sequences)
        return y

    def idx2tensor(self, idx_sequences):
        batch_size = len(idx_sequences)
        seq_lens = [len(seq) for seq in idx_sequences]
        max_seq_len = max(seq_lens)
        class_num = self.states_num
        outputs_tensor = self.tensor_ensure_gpu(torch.zeros(batch_size, class_num + 1, max_seq_len, dtype=torch.float))
        for n in range(batch_size):
            for seq_len in range(seq_lens[n]):
                for k in range(seq_len):
                    c = idx_sequences[n][k]
                    outputs_tensor[n, c, k] = 1
        return outputs_tensor
