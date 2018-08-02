import string

import torch
import torch.nn as nn

from classes.element_seq_indexer import ElementSeqIndexer

class LayerCharEmbeddings(nn.Module):
    def __init__(self, char_embeddings_dim, gpu, freeze_char_embeddings=False, max_pad_len=20):
        super(LayerCharEmbeddings, self).__init__()
        self.char_embeddings_dim = char_embeddings_dim
        self.gpu = gpu
        self.freeze_char_embeddings = freeze_char_embeddings
        self.max_pad_len = max_pad_len # standard len to pad
        self.feature_dim = self.max_pad_len*self.char_embeddings_dim
        # Init character sequences indexer
        self.char_seq_indexer = ElementSeqIndexer(gpu = gpu, caseless=True, load_embeddings=False)
        for c in list(string.printable):
            self.char_seq_indexer.add_element(c)
        self.char_seq_indexer.add_element(self.char_seq_indexer.unk)
        # Init character embedding
        self.embeddings = nn.Embedding(num_embeddings=self.char_seq_indexer.get_elements_num() + 1, # + <unk>
                                       embedding_dim=char_embeddings_dim,
                                       padding_idx=0)
        nn.init.uniform_(self.embeddings.weight, -0.5, 0.5) # Ma, 2016

    def forward(self, word_sequences):
        print(word_sequences)
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        char_embeddings_feature = torch.zeros(batch_num, max_seq_len, self.feature_dim, dtype=torch.float)
        for n, word_seq in enumerate(word_sequences):
            curr_seq_len = len(word_seq)
            input_tensor = self.char_seq_indexer.elements2tensor(word_seq, align='center', max_seq_len=self.max_pad_len)
            z = self.embeddings(input_tensor).view(curr_seq_len, self.feature_dim)
            for k in range(curr_seq_len):
                char_embeddings_feature[n, k] = z[k, :]
        return char_embeddings_feature
