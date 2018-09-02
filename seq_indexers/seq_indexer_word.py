import string
import numpy as np
import re
import torch
#from jellyfish import soundex
from autocorrect import spell

from seq_indexers.seq_indexer_base import SeqIndexerBase

class SeqIndexerWord(SeqIndexerBase):
    def __init__(self, gpu, check_for_lowercase, embeddings_dim, verbose):
        super(SeqIndexerWord, self).__init__(gpu, check_for_lowercase, embeddings_dim, verbose)
        SeqIndexerBase.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True, pad='<pad>',
                                unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim, verbose=verbose)

    def load_vocabulary_from_embeddings_file_and_unique_words_list(self, emb_fn, emb_delimiter, unique_words_list):
        emb_dict = SeqIndexerBase.load_emb_dict_from_file(self, emb_fn, emb_delimiter)
        original_words_num = 0
        lowercase_words_num = 0
        zero_digits_replaced_num = 0
        zero_digits_replaced_lowercase_num = 0
        soundex_replaced_num = 0
        #soundex_dict = dict()
        #for word_emb in emb_dict.keys():
        #    try:
        #        word_emb_soundex_hash = soundex(word_emb)
        #        soundex_dict[word_emb_soundex_hash] = word_emb
        #    except:
        #        continue
        self.out_of_vocabulary_words_list = list()
        for word in unique_words_list:
            if word in emb_dict.keys():
                self.add_item(word)
                self.add_emb_vector(emb_dict[word])
                original_words_num += 1
            elif self.check_for_lowercase and word.lower() in emb_dict.keys():
                self.add_item(word)
                self.add_emb_vector(emb_dict[word.lower()])
                lowercase_words_num += 1
            elif self.zero_digits and re.sub('\d', '0', word) in emb_dict.keys():
                self.add_item(word)
                self.add_emb_vector(emb_dict[re.sub('\d', '0', word)])
                zero_digits_replaced_num += 1
            elif self.check_for_lowercase and self.zero_digits and re.sub('\d', '0', word.lower()) in emb_dict.keys():
                self.add_item(word)
                self.add_emb_vector(emb_dict[re.sub('\d', '0', word.lower())])
                zero_digits_replaced_lowercase_num += 1
            elif 2 == 1 and spell(word) in emb_dict.keys():
                pass
                #self.add_element(word)
                #self.__add_emb_vector(emb_dict[spell(word)])
                #print('word = %s, spell(word) = %s' % (word, spell(word)))
                #soundex_replaced_num += 1
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
