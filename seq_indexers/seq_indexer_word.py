"""
.. module:: SeqIndexerWord
    :synopsis: SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back.

.. moduleauthor:: Artem Chernodub
"""

import string
import numpy as np
import re
import torch
#from jellyfish import soundex
from autocorrect import spell

from seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings

class SeqIndexerWord(SeqIndexerBaseEmbeddings):
    def __init__(self, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose)
        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0

    def get_embeddings_word(self, word, embeddings_word_list):
        if word in embeddings_word_list:
            self.original_words_num += 1
            return word
        elif self.check_for_lowercase and word.lower() in embeddings_word_list:
            self.lowercase_words_num += 1
            return word.lower()
        elif self.zero_digits and re.sub('\d', '0', word) in embeddings_word_list:
            self.zero_digits_replaced_num += 1
            return re.sub('\d', '0', word)
        elif self.check_for_lowercase and self.zero_digits and re.sub('\d', '0', word.lower()) in embeddings_word_list:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub('\d', '0', word.lower())
        return None

    def load_items_from_embeddings_file_and_unique_words_list(self, emb_fn, emb_delimiter, unique_words_list):
        embeddings_word_list = SeqIndexerBaseEmbeddings.load_embeddings_word_list_from_file(emb_fn, emb_delimiter,
                                                                                            verbose=True)
        emb_word2unique_word_dict = dict()
        out_of_vocabulary_words_list = list()

        for unique_word in unique_words_list:
            emb_word = self.get_embeddings_word(unique_word, embeddings_word_list)
            if emb_word is not None:
                if emb_word not in emb_word2unique_word_dict:
                    emb_word2unique_word_dict[emb_word] = unique_word
            else:
                out_of_vocabulary_words_list.append(unique_word)

        print('len(unique_words_list)=', len(unique_words_list))
        print('len(emb_word2unique_word_dict.keys())=', len(emb_word2unique_word_dict.keys()))
        A=0
        for emb_word, vec in SeqIndexerBaseEmbeddings.load_emb_for_words_list(emb_fn, emb_delimiter,
                                                                              emb_words_list=emb_word2unique_word_dict.keys(),
                                                                              verbose=True):
            self.add_item(emb_word2unique_word_dict[emb_word])
            self.add_emb_vector(vec)
            A+=1
        if self.verbose:
            print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
            print('    First 50 OOV words:')
            for i, oov_word in enumerate(out_of_vocabulary_words_list):
                print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                if i > 49:
                    break
            print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
            print(' -- original_words_num = %d' % self.original_words_num)
            print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)
            print('A=',A)

    def get_unique_characters_list(self, verbose=False, init_by_printable_characters=True):
        if init_by_printable_characters:
            unique_characters_set = set(string.printable)
        else:
            unique_characters_set = set()
        if verbose:
            cnt = 0
        for n, word in enumerate(self.get_items_list()):
            len_delta = len(unique_characters_set)
            unique_characters_set = unique_characters_set.union(set(word))
            if verbose and len(unique_characters_set) > len_delta:
                cnt += 1
                print('n = %d/%d (%d) %s' % (n, len(self.get_items_list), cnt, word))
        return list(unique_characters_set)
