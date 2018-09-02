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

    def get_char_tensor(self, curr_char_seq, word_len):
        return SeqIndexerBase.elements2tensor(self, curr_char_seq, align='center', word_len=word_len)  # curr_seq_len x word_len
