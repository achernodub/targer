"""
.. module:: TaggerBase
    :synopsis:     TaggerBase is an abstract class for tagger models. It implements the tagging functionality for
    different types of inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class
    SequencesIndexer is used for input and output data formats conversions. Abstract method `forward` is used in order
    to make these predictions, it have to be implemented in ancestors.

.. moduleauthor:: Artem Chernodub
"""

import math
import os.path

import torch
import torch.nn as nn

from models.tagger_birnn import TaggerBiRNN
from models.tagger_birnn_cnn import TaggerBiRNNCNN
from models.tagger_birnn_crf import TaggerBiRNNCRF
from models.tagger_birnn_cnn_crf import TaggerBiRNNCNNCRF


class TaggerBase(nn.Module):
    """
    """
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
        tagger.self_ensure_gpu()
        return tagger

    @staticmethod
    def create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train):
        if args.model == 'BiRNN':
            tagger = TaggerBiRNN(word_seq_indexer=word_seq_indexer,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 rnn_hidden_dim=args.rnn_hidden_dim,
                                 freeze_word_embeddings=args.freeze_word_embeddings,
                                 dropout_ratio=args.dropout_ratio,
                                 rnn_type=args.rnn_type,
                                 gpu=args.gpu)
        elif args.model == 'BiRNNCNN':
            tagger = TaggerBiRNNCNN(word_seq_indexer=word_seq_indexer,
                                    tag_seq_indexer=tag_seq_indexer,
                                    class_num=tag_seq_indexer.get_class_num(),
                                    batch_size=args.batch_size,
                                    rnn_hidden_dim=args.rnn_hidden_dim,
                                    freeze_word_embeddings=args.freeze_word_embeddings,
                                    dropout_ratio=args.dropout_ratio,
                                    rnn_type=args.rnn_type,
                                    gpu=args.gpu,
                                    freeze_char_embeddings=args.freeze_char_embeddings,
                                    char_embeddings_dim=args.char_embeddings_dim,
                                    word_len=args.word_len,
                                    char_cnn_filter_num=args.char_cnn_filter_num,
                                    char_window_size=args.char_window_size)
        elif args.model == 'BiRNNCRF':
            tagger = TaggerBiRNNCRF(word_seq_indexer=word_seq_indexer,
                                    tag_seq_indexer=tag_seq_indexer,
                                    class_num=tag_seq_indexer.get_class_num(),
                                    batch_size=args.batch_size,
                                    rnn_hidden_dim=args.rnn_hidden_dim,
                                    freeze_word_embeddings=args.freeze_word_embeddings,
                                    dropout_ratio=args.dropout_ratio,
                                    rnn_type=args.rnn_type,
                                    gpu=args.gpu)
            tagger.crf_layer.init_transition_matrix_empirical(tag_sequences_train)
        elif args.model == 'BiRNNCNNCRF':
            tagger = TaggerBiRNNCNNCRF(word_seq_indexer=word_seq_indexer,
                                       tag_seq_indexer=tag_seq_indexer,
                                       class_num=tag_seq_indexer.get_class_num(),
                                       batch_size=args.batch_size,
                                       rnn_hidden_dim=args.rnn_hidden_dim,
                                       freeze_word_embeddings=args.freeze_word_embeddings,
                                       dropout_ratio=args.dropout_ratio,
                                       rnn_type=args.rnn_type,
                                       gpu=args.gpu,
                                       freeze_char_embeddings=args.freeze_char_embeddings,
                                       char_embeddings_dim=args.char_embeddings_dim,
                                       word_len=args.word_len,
                                       char_cnn_filter_num=args.char_cnn_filter_num,
                                       char_window_size=args.char_window_size)
            tagger.crf_layer.init_transition_matrix_empirical(tag_sequences_train)
        else:
            raise ValueError('Unknown tagger model, must be one of "BiRNN"/"BiRNNCNN"/"BiRNNCRF"/"BiRNNCNNCRF".')
        return tagger

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

    def get_mask(self, word_sequences):
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
