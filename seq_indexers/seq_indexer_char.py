import string
import numpy as np
import re
import torch

from seq_indexers.seq_indexer_base import SeqIndexerBase

class SeqIndexerChar(SeqIndexerBase):
    def __init__(self, gpu):
        super(SeqIndexerChar, self).__init__(gpu)
        SeqIndexerBase.__init__(self, gpu=gpu, check_for_lowercase=False, zero_digits=False, pad='<pad>',
                                unk='<unk>', load_embeddings=False, verbose=True)

    def add_char(self, c):
        if not self.item_exists(c):
            self.add_item(c)

    def get_char_tensor(self, curr_char_seq, word_len):
        return SeqIndexerBase.items2tensor(self, curr_char_seq, align='center', word_len=word_len)  # curr_seq_len x word_len

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
