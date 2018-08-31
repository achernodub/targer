import string

import numpy as np
import re
import torch
#from jellyfish import soundex
from autocorrect import spell

class ElementSeqIndexer():
    """
    ElementsSeqIndexer converts list of lists of strings to to the list of lists of integer indices and back.
        Strings could be either words, tags or characters. Indices are stored into internal vocabularies.
        Index 0 stands for the unknown string.
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
                self.__add_emb_vector(self.__get_zero_emb_vector())
        if unk is not None:
            self.unk_idx = self.add_element(unk)
            if load_embeddings:
                self.__add_emb_vector(self.__get_random_emb_vector())

    def get_elements_list(self):
        return list(self.element2idx_dict.keys())

    def element_exists(self, element):
        return element in self.element2idx_dict.keys()

    def add_element(self, element):
        idx = len(self.get_elements_list())
        self.element2idx_dict[element] = idx
        self.idx2element_dict[idx] = element
        return idx

    def __add_emb_vector(self, emb_vector):
        self.embedding_vectors_list.append(emb_vector)

    def __get_zero_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.embeddings_dim)]

    def __get_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim), self.embeddings_dim).tolist()

    @staticmethod
    def __load_emb_dict_from_file(emb_fn, emb_delimiter):
        emb_dict = dict()
        for line in open(emb_fn, 'r'):
            values = line.split(emb_delimiter)
            word = values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            emb_dict[word] = emb_vector
        return emb_dict

    def load_vocabulary_from_embeddings_file_and_unique_words_list(self, emb_fn, emb_delimiter, unique_words_list):
        emb_dict = ElementSeqIndexer.__load_emb_dict_from_file(emb_fn, emb_delimiter)
        original_words_num = 0
        lowercase_words_num = 0
        zero_digits_replaced_num = 0
        zero_digits_replaced_lowercase_num = 0
        soundex_replaced_num = 0
        soundex_dict = dict()
        for word_emb in emb_dict.keys():
            try:
                word_emb_soundex_hash = soundex(word_emb)
                soundex_dict[word_emb_soundex_hash] = word_emb
            except:
                continue
        self.out_of_vocabulary_words_list = list()
        for word in unique_words_list:
            if word in emb_dict.keys():
                self.add_element(word)
                self.__add_emb_vector(emb_dict[word])
                original_words_num += 1
            elif self.check_for_lowercase and word.lower() in emb_dict.keys():
                self.add_element(word)
                self.__add_emb_vector(emb_dict[word.lower()])
                lowercase_words_num += 1
            elif self.zero_digits and re.sub('\d', '0', word) in emb_dict.keys():
                self.add_element(word)
                self.__add_emb_vector(emb_dict[re.sub('\d', '0', word)])
                zero_digits_replaced_num += 1
            elif self.check_for_lowercase and self.zero_digits and re.sub('\d', '0', word.lower()) in emb_dict.keys():
                self.add_element(word)
                self.__add_emb_vector(emb_dict[re.sub('\d', '0', word.lower())])
                zero_digits_replaced_lowercase_num += 1
            elif 2 == 1 and spell(word) in emb_dict.keys():
                self.add_element(word)
                self.__add_emb_vector(emb_dict[spell(word)])
                print('word = %s, spell(word) = %s' % (word, spell(word)))
                soundex_replaced_num += 1
            #elif soundex(word) in soundex_dict.keys():
            #        soundex_word = soundex_dict[soundex(word)]
            #        self.add_element(word)
            #        self.__add_emb_vector(emb_dict[soundex_word])
            #        #print('word=%s, soundex_word=%s' % (word, soundex_word))
            #        soundex_replaced_num += 1
            else:
                self.out_of_vocabulary_words_list.append(word)
                continue
        if self.verbose:
            print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
            print(' -- original_words_num = %d' % original_words_num)
            print(' -- lowercase_words_num = %d' % lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' % zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' % zero_digits_replaced_lowercase_num)
            print(' -- soundex_replaced_num = %d' % soundex_replaced_num)
            print(' -- len(out_of_vocabulary_words_list) = %d' % len(self.out_of_vocabulary_words_list))
            print('    First 50 OOV words:')
            for i, oov_word in enumerate(self.out_of_vocabulary_words_list):
                print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                if i > 49:
                    break

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
        if self.verbose:
            print('%s embeddings file was loaded, %d vectors, dim = %d.' % (emb_fn, len(self.embedding_vectors_list), self.embeddings_dim))


    def load_vocabulary_from_element_sequences(self, element_sequences, verbose=False):
        if self.load_embeddings and not self.embeddings_loaded:
            raise ValueError('Embeddings are not loaded.')
        for element_seq in element_sequences:
            for element in element_seq:
                if not self.element_exists(element):
                    self.add_element(element)
                    self.out_of_vocabulary_list.append(element)
                    if self.load_embeddings:
                        self.__add_emb_vector(self.__get_random_emb_vector())
        if verbose:
            print('%d elements not found:' % len(self.out_of_vocabulary_list))
            for k, element in enumerate(self.out_of_vocabulary_list):
                print(' -= %d/%d out of vocabulary token: %s' % (k, len(self.out_of_vocabulary_list), element))
            print('%d embeddings loaded/generated.' % len(self.embedding_vectors_list))


    def load_vocabulary_from_tag_sequences(self, tag_sequences):
        assert self.load_embeddings == False
        for tag_seq in tag_sequences:
            for tag in tag_seq:
                if not self.element_exists(tag):
                    self.add_element(tag)
        if self.verbose:
            print('\nload_vocabulary_from_tag_sequences:')
            print(' -- class_num = %d' % self.get_class_num())
            print(' --', self.element2idx_dict)

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))

    def get_elements_num(self):
        return len(self.get_elements_list())

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return self.get_elements_num() - 2
        if self.pad is not None or self.unk is not None:
            return self.get_elements_num() - 1
        return self.get_elements_num()

    def elements2idx(self, element_sequences):
        idx_sequences = []
        for element_seq in element_sequences:
            #element_normalized_seq = map(self.__element_normalization, element_seq)
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
