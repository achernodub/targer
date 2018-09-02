import string

import numpy as np
import re
import torch
#from jellyfish import soundex
from autocorrect import spell

class SeqIndexerBase():
    """
    SeqIndexer converts list of lists of strings to to the list of lists of integer indices and back.
    Strings could be either words, tags or characters. Indices are stored into internal vocabularies.
    """

    def __init__(self, gpu=-1, check_for_lowercase=True, zero_digits=False, pad='<pad>', unk='<unk>',
                 load_embeddings=False, embeddings_dim=0, verbose=False):
        self.gpu = gpu
        self.check_for_lowercase = check_for_lowercase
        self.zero_digits = zero_digits
        self.pad = pad
        self.unk = unk
        self.load_embeddings = load_embeddings
        self.embeddings_dim = embeddings_dim
        self.verbose = verbose
        self.out_of_vocabulary_list = list()
        self.element2idx_dict = dict()
        self.idx2element_dict = dict()
        if load_embeddings:
            self.embeddings_loaded = False
            self.embedding_vectors_list = list()
        if pad is not None:
            self.pad_idx = self.add_element(pad)
            if load_embeddings:
                self.add_emb_vector(self.get_zero_emb_vector())
        if unk is not None:
            self.unk_idx = self.add_element(unk)
            if load_embeddings:
                self.add_emb_vector(self.get_random_emb_vector())

    def get_elements_list(self):
        return list(self.element2idx_dict.keys())

    def get_elements_num(self):
        return len(self.get_elements_list())

    def element_exists(self, element):
        return element in self.element2idx_dict.keys()

    def add_element(self, element):
        idx = len(self.get_elements_list())
        self.element2idx_dict[element] = idx
        self.idx2element_dict[idx] = element
        return idx

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return self.get_elements_num() - 2
        if self.pad is not None or self.unk is not None:
            return self.get_elements_num() - 1
        return self.get_elements_num()

    def add_emb_vector(self, emb_vector):
        self.embedding_vectors_list.append(emb_vector)

    def get_zero_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.embeddings_dim)]

    def get_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim), self.embeddings_dim).tolist()

    def load_emb_dict_from_file(self, emb_fn, emb_delimiter):
        emb_dict = dict()
        for line in open(emb_fn, 'r'):
            values = line.split(emb_delimiter)
            word = values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            emb_dict[word] = emb_vector
        return emb_dict

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))

    def elements2idx(self, element_sequences):
        idx_sequences = []
        for element_seq in element_sequences:
            idx_seq = list()
            for element in element_seq:
                if element in self.element2idx_dict:
                    idx_seq.append(self.element2idx_dict[element])
                else:
                    if self.unk is not None:
                        idx_seq.append(self.element2idx_dict[self.unk])
                    else:
                        idx_seq.append(self.element2idx_dict[self.pad])
            idx_sequences.append(idx_seq)
        return idx_sequences

    def idx2elements(self, idx_sequences):
        element_sequences = []
        for idx_seq in idx_sequences:
            element_seq = [self.idx2element_dict[idx] for idx in idx_seq]
            element_sequences.append(element_seq)
        return element_sequences

    def idx2tensor(self, idx_sequences, align='left', word_len=-1):
        batch_size = len(idx_sequences)
        if word_len == -1:
            word_len = max([len(idx_seq) for idx_seq in idx_sequences])
        # tensor = torch.zeros(batch_size, word_len, dtype=torch.long)
        if self.gpu >= 0:
            tensor = torch.cuda.LongTensor(batch_size, word_len).fill_(0)
        else:
            tensor = torch.LongTensor(batch_size, word_len).fill_(0)
        for k, idx_seq in enumerate(idx_sequences):
            curr_seq_len = len(idx_seq)
            if curr_seq_len > word_len:
                idx_seq = [idx_seq[i] for i in range(word_len)]
                curr_seq_len = word_len
            if align == 'left':
                tensor[k, :curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            elif align == 'center':
                start_idx = (word_len - curr_seq_len) // 2
                tensor[k, start_idx:start_idx+curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            else:
                raise ValueError('Unknown align string.')
        if self.gpu >= 0:
            tensor = tensor.cuda(device=self.gpu)
        return tensor

    def elements2tensor(self, element_sequences, align='left', word_len=-1):
        idx = self.elements2idx(element_sequences)
        return self.idx2tensor(idx, align, word_len)

    def get_unique_characters_list(self, verbose=False, init_by_printable_characters=True):
        if init_by_printable_characters:
            unique_characters_set = set(string.printable)
        else:
            unique_characters_set = set()
        if verbose:
            cnt = 0
        for n, word in enumerate(self.get_elements_list()):
            len_delta = len(unique_characters_set)
            unique_characters_set = unique_characters_set.union(set(word))
            if verbose and len(unique_characters_set) > len_delta:
                cnt += 1
                print('n = %d/%d (%d) %s' % (n, len(self.get_elements_list), cnt, word))
        return list(unique_characters_set)
