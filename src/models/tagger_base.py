"""
.. module:: TaggerBase
    :synopsis:     TaggerBase is an abstract class for tagger models. It implements the tagging functionality for
    different types of inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class
    SequencesIndexer is used for input and output data formats conversions. Abstract method `forward` is used in order
    to make these predictions, it have to be implemented in ancestors.

.. moduleauthor:: Artem Chernodub
"""
import math
import torch
import torch.nn as nn


class TaggerBase(nn.Module):
    def __init__(self,  word_seq_indexer, tag_seq_indexer, gpu, batch_size):
        super(TaggerBase, self).__init__()
        self.word_seq_indexer = word_seq_indexer
        self.tag_seq_indexer = tag_seq_indexer
        self.gpu = gpu
        self.batch_size = batch_size

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

    def save_tagger(self, checkpoint_fn):
        self.cpu()
        torch.save(self, checkpoint_fn)
        self.self_ensure_gpu()

    def forward(self, *input):
        pass

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

    def predict_tags_from_words(self, word_sequences, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j])
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                  end='', flush=True)
        return output_tag_sequences

    def get_mask_from_word_sequences(self, word_sequences):
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        mask_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, dtype=torch.float))
        for k, word_seq in enumerate(word_sequences):
            mask_tensor[k, :len(word_seq)] = 1
        return mask_tensor # batch_size x max_seq_len

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)
