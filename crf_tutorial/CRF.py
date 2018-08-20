"""Inspired by http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np


class CRF(nn.Module):
    
    def __init__(self, n_dice, log_likelihood):
        super(CRF, self).__init__()
        
        self.n_states = n_dice
        self.transition = torch.nn.init.normal(nn.Parameter(torch.randn(n_dice, n_dice + 1)), -1, 0.1)
        self.loglikelihood = log_likelihood
    

    def to_scalar(self, var):
        return var.view(-1).data.tolist()[0]


    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return self.to_scalar(idx)
        
    # numerically stable log sum exp
    # Source: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
    
    def _data_to_likelihood(self, rolls):
        """Converts a numpy array of rolls (integers) to log-likelihood.

        Input is one [1, n_rolls]
        """
        return Variable(torch.FloatTensor(self.loglikelihood[rolls]), requires_grad=False)
        
    
    def _compute_likelihood_numerator(self, loglikelihoods, states):
        """Computes numerator of likelihood function for a given sequence.
        
        We'll iterate over the sequence of states and compute the sum 
        of the relevant transition cost with the log likelihood of the observed
        roll. 

        Input:
            loglikelihoods: torch Variable. Matrix of n_obs x n_states. 
                            i,j entry is loglikelihood of observing roll i given state j
            states: sequence of labels
        Output:
            score: torch Variable. Score of assignment. 
        """
        prev_state = self.n_states
        score = Variable(torch.Tensor([0]))
        for index, state in enumerate(states):
            score += self.transition[state, prev_state] + loglikelihoods[index, state]
            prev_state = state
        return score
    
    def _compute_likelihood_denominator(self, loglikelihoods):
        """Implements the forward pass of the forward-backward algorithm.
        
        We loop over all possible states efficiently using the recursive
        relationship: alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * C(y_t | y{t-1} = i)

        Input:
            loglikelihoods: torch Variable. Same input as _compute_likelihood_numerator.
                            This algorithm efficiently loops over all possible state sequences
                            so no other imput is needed.
        Output:
            torch Variable. 
        """

        # Stores the current value of alpha at timestep t
        prev_alpha = self.transition[:, self.n_states] + loglikelihoods[0].view(1, -1)

        for roll in loglikelihoods[1:]:
            alpha_t = []

            # Loop over all possible states
            for next_state in range(self.n_states):
                
                # Compute all possible costs of transitioning to next_state
                feature_function = self.transition[next_state,:self.n_states].view(1, -1) +\
                                   roll[next_state].view(1, -1).expand(1, self.n_states) # 1 x n_states

                alpha_t_next_state = prev_alpha + feature_function # 1 x n_states

                alpha_t.append(self.log_sum_exp(alpha_t_next_state)) # alpha_t - list with n_states

            prev_alpha = torch.cat(alpha_t).view(1, -1) # 1 x n_states
        return self.log_sum_exp(prev_alpha)
    
    def _viterbi_algorithm(self, loglikelihoods):
        """Implements Viterbi algorithm for finding most likely sequence of labels.
        
        Very similar to _compute_likelihood_denominator but now we take the maximum
        over the previous states as opposed to the sum. 

        Input:
            loglikelihoods: torch Variable. Same input as _compute_likelihood_denominator.

        Output:
            tuple. First entry is the most likely sequence of labels. Second is
                   the loglikelihood of this sequence. 
        """

        argmaxes = []

        # prev_delta will store the current score of the sequence for each state
        prev_delta = self.transition[:, self.n_states].contiguous().view(1, -1) +\
                      loglikelihoods[0].view(1, -1)

        for roll in loglikelihoods[1:]:
            local_argmaxes = []
            next_delta = []
            for next_state in range(self.n_states):
                feature_function = self.transition[next_state,:self.n_states].view(1, -1) + roll.view(1, -1) + prev_delta
                most_likely_state = self.argmax(feature_function)
                score = feature_function[0][most_likely_state]
                next_delta.append(score)
                local_argmaxes.append(most_likely_state)
            prev_delta = torch.cat(next_delta).view(1, -1)
            argmaxes.append(local_argmaxes)
        
        final_state = self.argmax(prev_delta)
        final_score = prev_delta[0][final_state]
        path_list = [final_state]

        # Backtrack through the argmaxes to find most likely state
        for states in reversed(argmaxes):
            final_state = states[final_state]
            path_list.append(final_state)
        
        return np.array(path_list), final_score
        
    def neg_log_likelihood(self, rolls, states):
        """Compute neg log-likelihood for a given sequence.
        
        Input: 
            rolls: numpy array, dim [1, n_rolls]. Integer 0-5 showing value on dice.
            states: numpy array, dim [1, n_rolls]. Integer 0, 1. 0 if dice is fair.
        """
        loglikelihoods = self._data_to_likelihood(rolls)
        states = torch.LongTensor(states)
        
        sequence_loglik = self._compute_likelihood_numerator(loglikelihoods, states)
        denominator = self._compute_likelihood_denominator(loglikelihoods)
        return denominator - sequence_loglik
               
    
    def forward(self, rolls):
        loglikelihoods = self._data_to_likelihood(rolls)
        return self._viterbi_algorithm(loglikelihoods)
                
    