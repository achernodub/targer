import string

import torch
import torch.nn as nn

from layers.layer_base import LayerBase
from classes.element_seq_indexer import ElementSeqIndexer

class LayerCharEmbeddings(LayerBase):
    def __init__(self, gpu, char_embeddings_dim, freeze_char_embeddings=False, max_char_pad_len=20):
        super(LayerCharEmbeddings, self).__init__(gpu)
        self.gpu = gpu
        self.char_embeddings_dim = char_embeddings_dim
        self.freeze_char_embeddings = freeze_char_embeddings
        self.max_char_pad_len = max_char_pad_len # standard len to pad
        self.out_dim = self.max_char_pad_len # self.max_char_pad_len * self.char_embeddings_dim
        # Init character sequences indexer
        self.char_seq_indexer = ElementSeqIndexer(gpu = gpu, caseless=True, load_embeddings=False)
        for c in list(string.printable):
            self.char_seq_indexer.add_element(c)
        self.char_seq_indexer.add_element(self.char_seq_indexer.unk)
        # Init character embedding
        self.embeddings = nn.Embedding(num_embeddings=self.char_seq_indexer.get_elements_num() + 1, # + <unk>
                                       embedding_dim=char_embeddings_dim,
                                       padding_idx=0)
        #nn.init.uniform_(self.embeddings.weight, -0.5, 0.5) # Ma, 2016

    def forward(self, word_sequences):
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        char_embeddings_feature = self.make_gpu(torch.zeros(batch_num, max_seq_len, self.char_embeddings_dim,
                                                            self.max_char_pad_len, dtype=torch.float))
        for n, word_seq in enumerate(word_sequences):
            curr_seq_len = len(word_seq)
            char_sequences = [[c for c in word] for word in word_seq]
            input_tensor = self.char_seq_indexer.elements2tensor(char_sequences, align='center', max_seq_len=self.max_char_pad_len)
            curr_char_emb = self.embeddings(input_tensor).permute(0, 2, 1)  # curr_seq_len x  char_embeddings_dim x max_char_pad_len
            char_embeddings_feature[n, :curr_seq_len, :, :] = curr_char_emb
        return char_embeddings_feature # shape: batch_num x max_seq_len x  char_embeddings_dim x max_char_pad_len
