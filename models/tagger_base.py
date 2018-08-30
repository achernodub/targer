import math
import os.path

import torch
import torch.nn as nn


class TaggerBase(nn.Module):
    """
    TaggerBase is an abstract class for tagger models. It implements the tagging functionality for different types of
     inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class SequencesIndexer is used
     for input and output data formats conversions. Abstract method `forward` is used in order to make these predictions,
     it have to be implemented in ancestors.
    """
    def __init__(self,  word_seq_indexer, tag_seq_indexer, gpu):
        super(TaggerBase, self).__init__()
        self.word_seq_indexer = word_seq_indexer
        self.tag_seq_indexer = tag_seq_indexer
        self.gpu = gpu

    def tensor_ensure_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def self_ensure_gpu(self):
        if self.gpu >= 0:
            self.cuda(device=self.gpu)
        else:
            self.cpu()

    def clip_gradients(self, clip_grad):
        pass
        #nn.utils.clip_grad_norm_(self.parameters(), clip_grad) ###############################

    def save(self, checkpoint_fn):
        self.cpu()
        torch.save(self, checkpoint_fn)
        self.self_ensure_gpu()

    @staticmethod
    def load(checkpoint_fn, gpu=-1):
        if not os.path.isfile(checkpoint_fn):
            raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                             "--save_best_path" param to create it.' % checkpoint_fn)
        tagger = torch.load(checkpoint_fn)
        tagger.gpu = gpu
        #tagger.self_ensure_gpu()
        return tagger

    def predict_idx_from_words(self, word_sequences):
        self.eval()
        outputs_tensor = self.forward(word_sequences) # batch_size x num_class+1 x max_seq_len
        output_idx_sequences = list()
        for k in range(len(word_sequences)):
            idx_seq = list()
            for l in range(len(word_sequences[k])):
                curr_output = outputs_tensor[k, 1:, l] # ignore the first component of output
                max_no = curr_output.argmax(dim=0)
                idx_seq.append(max_no.item() + 1)
            output_idx_sequences.append(idx_seq)
        return output_idx_sequences

    def predict_tags_from_words(self, word_sequences, batch_size=1):
        batch_num = math.floor(len(word_sequences) / batch_size)
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j])
            curr_output_tag_sequences = self.tag_seq_indexer.idx2elements(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
        return output_tag_sequences
