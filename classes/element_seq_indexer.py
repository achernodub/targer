import string

import numpy as np
import re
import torch

class ElementSeqIndexer():
    """
    ElementsSeqIndexer converts list of lists of strings to to the list of lists of integer indices and back.
        Strings could be either words, tags or characters. Indices are stored into internal vocabularies.
        Index 0 stands for the unknown string.
    """

    def __init__(self, gpu=-1, caseless=True, load_embeddings=False, verbose=False, pad='<pad>', unk='<unk>',
                 zero_digits=False):
        self.gpu = gpu
        self.caseless = caseless
        self.load_embeddings = load_embeddings
        self.verbose = verbose
        self.pad = pad
        self.unk = unk
        self.zero_digits = zero_digits
        self.out_of_vocabulary_list = list()
        self.element2idx_dict = dict()
        self.idx2element_dict = dict()
        if pad is not None:
            self.add_element(pad)
        if load_embeddings:
            self.embeddings_loaded = False
            self.embeddings_dim = 0
            self.embeddings_list = list()
        if unk is not None:
            self.add_element(unk)

    def elements_list(self):
        return self.element2idx_dict.keys()

    def add_element(self, element):
        if self.caseless:
            element = element.lower()
        if self.zero_digits:
            element = re.sub('\d', '0', element)
        idx = len(self.elements_list())
        self.element2idx_dict[element] = idx
        self.idx2element_dict[idx] = element

    def __element_exists(self, element):
        if self.caseless:
            element = element.lower()
        if self.zero_digits:
            element = re.sub('\d', '0', element)
        return (element in self.elements_list())

    def __add_emb_vector(self, emb_vector):
        self.embeddings_list.append(emb_vector)

    def __get_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim), self.embeddings_dim).tolist()

    def __get_zero_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.embeddings_dim)]

    def load_vocabulary_from_embeddings_file(self, emb_fn, emb_delimiter):
        if not self.load_embeddings:
            raise ValueError('load_embeddings == False')
        # 0) Get dimensionality of embeddings
        for line in open(emb_fn, 'r'):
            values = line.split(emb_delimiter)
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.embeddings_dim = len(emb_vector)
            break
        # 1) Generate random embedding which will be correspond to the index 0 that in used in the batches instead of mask.
        if self.pad is not None:
            self.__add_emb_vector(self.__get_zero_emb_vector()) # zeros for <pad>
        if self.unk is not None:
            self.__add_emb_vector(self.__get_random_emb_vector()) # randoms for <unk>
        # 2) Add embeddings from file
        for line in open(emb_fn, 'r'):
            values = line.split(emb_delimiter)
            element = values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.add_element(element)
            self.__add_emb_vector(emb_vector)
        self.embeddings_loaded = True
        # 3) Generate random embedding for 'unknown' token
        self.add_element(self.unk)
        self.__add_emb_vector(self.__get_random_emb_vector())
        if self.verbose:
            print('%s embeddings file was loaded, %d vectors, dim = %d.' % (emb_fn, len(self.embeddings_list), self.embeddings_dim))

    def get_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embeddings_list))

    def load_vocabulary_from_element_sequences(self, element_sequences, verbose=False):
        if self.load_embeddings and not self.embeddings_loaded:
            raise ValueError('Embeddings are not loaded.')
        for element_seq in element_sequences:
            for element in element_seq:
                if not self.__element_exists(element):
                    self.add_element(element)
                    self.out_of_vocabulary_list.append(element)
                    if self.load_embeddings:
                        self.__add_emb_vector(self.__get_random_emb_vector())
        if verbose:
            print('%d elements not found:' % len(self.out_of_vocabulary_list))
            for k, element in enumerate(self.out_of_vocabulary_list):
                print(' -= %d/%d out of vocabulary token: %s' % (k, len(self.out_of_vocabulary_list), element))
            print('%d embeddings loaded/generated.' % len(self.embeddings_list))

    def get_elements_num(self):
        return len(self.elements_list())

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return len(self.elements_list()) - 2
        if self.pad is not None or self.unk is not None:
            return len(self.elements_list()) - 1
        return len(self.elements_list())

    def elements2idx(self, element_sequences):
        idx_sequences = []
        for element_seq in element_sequences:
            element_caseless_seq = [element.lower() if self.caseless else element for element in element_seq]
            idx_seq = list()
            #idx_seq = [self.element2idx_dict.get(element, self.element2idx_dict[self.unk])
            #           for element in element_caseless_seq]
            for element in element_caseless_seq:
                if element in self.element2idx_dict:
                    idx_seq.append(self.element2idx_dict[element])
                else:
                    if self.unk is not None:
                        idx_seq.append(self.element2idx_dict[self.unk])
                    else:
                        idx_seq.append(0) # pad
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
        for n, token in enumerate(self.elements_list()):
            len_delta = len(unique_characters_set)
            unique_characters_set = unique_characters_set.union(set(token))
            if verbose and len(unique_characters_set) > len_delta:
                cnt += 1
                print('n = %d/%d (%d) %s' % (n, len(self.elements_list), cnt, token))
        return list(unique_characters_set)
