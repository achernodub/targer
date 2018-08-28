import string

import torch
import torch.nn as nn

from layers.layer_base import LayerBase
from classes.element_seq_indexer import ElementSeqIndexer
from classes.utils import flatten

class LayerCharEmbeddings(LayerBase):
    def __init__(self, gpu, char_embeddings_dim, freeze_char_embeddings=False, word_len=20, unique_characters_list=None):
        super(LayerCharEmbeddings, self).__init__(gpu)
        self.gpu = gpu
        self.char_embeddings_dim = char_embeddings_dim
        self.freeze_char_embeddings = freeze_char_embeddings
        self.word_len = word_len # standard len to pad
        # Init character sequences indexer
        self.char_seq_indexer = ElementSeqIndexer(gpu = gpu, check_for_lowercase=True, load_embeddings=False, pad='<pad>',
                                                  unk='<unk>')
        if unique_characters_list is None:
            unique_characters_list = list(string.printable)
        for c in unique_characters_list:
            self.char_seq_indexer.__add_element(c)
        self.char_seq_indexer.__add_element(self.char_seq_indexer.unk)
        # Init character embedding
        self.embeddings = nn.Embedding(num_embeddings=self.char_seq_indexer.get_elements_num(),
                                       embedding_dim=char_embeddings_dim,
                                       padding_idx=0)
        # nn.init.uniform_(self.embeddings.weight, -0.5, 0.5) # Option: Ma, 2016

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        char_sequences = [[[c for c in word] for word in word_seq] for word_seq in word_sequences]
        input_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, self.word_len, dtype=torch.long))
        for n, curr_char_seq in enumerate(char_sequences):
            curr_seq_len = len(curr_char_seq)
            curr_char_seq_tensor = self.char_seq_indexer.elements2tensor(curr_char_seq,
                                                                         align='center',
                                                                         word_len=self.word_len) # curr_seq_len x word_len
            input_tensor[n, :curr_seq_len, :] = curr_char_seq_tensor
        char_embeddings_feature = self.embeddings(input_tensor)
        return char_embeddings_feature.permute(0, 1, 3, 2) # shape: batch_num x max_seq_len x char_embeddings_dim x word_len
