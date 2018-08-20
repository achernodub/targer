import random
import torch
import torch.nn as nn
import numpy as np

from classes.utils import info
import torch.nn.functional as F


START_STATE = 2

class Dice():
    def generate_sequence_x(self, seq_len):
        sequence_x = torch.zeros(seq_len, dtype=torch.long)
        for k in range(seq_len):
            curr_p = torch.zeros(self.observations_num, 1)
            for n in range(self.observations_num):
                curr_p[n] = self.probs[n]*random.random()
            _, max_idx = curr_p.max(0)
            sequence_x[k] = max_idx
        return sequence_x

class FairDice(Dice): # type 0
    def __init__(self):
        self.type = 0
        self.observations_num = 6
        self.probs = torch.FloatTensor(self.observations_num, 1).fill_(1/6)

class LiarDice(Dice): # type 1
    def __init__(self):
        self.type = 1
        self.observations_num = 6
        self.probs = torch.FloatTensor(self.observations_num, 1).fill_(0.04)
        self.probs[5] = 0.8

def sequence_x_to_loglikelihoods(sequence_x, prior_prob):
    # sequence_x : seq_len
    # prior_prob : observations_num x states_num (6 x 2)
    seq_len = sequence_x.shape[0]
    observations_num, states_num = prior_prob.shape
    loglikelihoods = torch.zeros(seq_len, states_num, dtype=torch.float)
    for k in range(seq_len):
        x_no = sequence_x[k].item() # from 0 to 5
        loglikelihoods[k, 0] = torch.log(prior_prob[x_no, 0])
        loglikelihoods[k, 1] = torch.log(prior_prob[x_no, 1])
    return loglikelihoods # seq_len x states_num (seq_len x 2)

def score_func(prev_state, curr_state, transition_matrix, loglikelihoods, k):
    return transition_matrix[prev_state, curr_state] + loglikelihoods[k, curr_state]



def compute_likelihood_numerator(transition_matrix, loglikelihoods, states):
    prev_state = START_STATE # we have 2 but 3 means "start"
    score = torch.Tensor([0])
    for k, curr_state in enumerate(states):
        score += score_func(prev_state, curr_state, transition_matrix, loglikelihoods, k)
        prev_state = curr_state
    return score

def log_sum_exp(inputs):
    return (inputs - F.log_softmax(inputs, dim=1)).mean(dim=1, keepdim=False)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp_D(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def compute_likelihood_denominator(transition_matrix, loglikelihoods):
    # loglikelihoods: seq_len x states_num (seq_len x 2)
    # alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * C(y_t | y{t-1} = i)
    print('compute_likelihood_denominator')
    seq_len, states_num = loglikelihoods.shape
    alpha = torch.zeros(seq_len, states_num, dtype=torch.float)

    # 1)
    for curr_state in range(states_num):
        alpha[0, curr_state] = score_func(START_STATE, curr_state, transition_matrix, loglikelihoods, k=0)

    print('alpha', alpha)

    # 2)
    for k in range(1, seq_len):
        for curr_state in range(states_num):
            curr_alpha = torch.zeros(1, states_num, dtype=torch.float)
            for prev_state in range(states_num):
                curr_alpha[0, prev_state] = alpha[k - 1, prev_state] * score_func(prev_state, curr_state,
                                                                             transition_matrix,
                                                                             loglikelihoods,
                                                                             k)
                print('curr_alpha', curr_alpha)
                lse = log_sum_exp(curr_alpha)
                print('lse', lse)
                alpha[k, prev_state] = lse
                print('----')
    print('alpha', alpha)
    Z = log_sum_exp(alpha[-1, :].view(1, -1)) ############## !!!
    print('Z', Z)
    return Z

def neg_log_likelihood(transition_matrix, loglikelihoods, states):
    numerator = compute_likelihood_numerator(transition_matrix, loglikelihoods, states)
    denominator = compute_likelihood_denominator(transition_matrix, loglikelihoods)
    return -(numerator - denominator)


print('Hello MyCRF!\n')

random.seed(42)
torch.manual_seed(42)

prior_prob = torch.cat((FairDice().probs, LiarDice().probs), dim=1)

'''
tensor([[ 0.1667,  0.0400],
        [ 0.1667,  0.0400],
        [ 0.1667,  0.0400],
        [ 0.1667,  0.0400],
        [ 0.1667,  0.0400],
        [ 0.1667,  0.8000]])'''


dice = FairDice()
#dice = LiarDice()

sequence_x = dice.generate_sequence_x(10)
loglikelihoods = sequence_x_to_loglikelihoods(sequence_x, prior_prob)

#print('sequence_x', sequence_x)
#print('loglikelihoods', loglikelihoods)

labels = ['FAIR', 'LIAR'] # + '<start>'
states_num = len(labels)

#print('labels', labels)
#print('states_num =', states_num)

#transition_matrix = torch.zeros(states_num, states_num - 1, dtype=torch.float, requires_grad=True)

transition_matrix = nn.Parameter(torch.randn(states_num + 1, states_num))
nn.init.normal_(transition_matrix, -1, 0.1)

info(transition_matrix, 'transition_matrix')
print(transition_matrix)

print('sequence_x ', sequence_x )
print('loglikelihoods', loglikelihoods)

exit()

compute_likelihood_denominator(transition_matrix, loglikelihoods)

# seq_len x states_num (seq_len x 2)

#neg_log_likelihood(transition_matrix, loglikelihoods, states):




print('\nThe end.')