"""
.. module:: SeqIndexerBaseEmbeddings
    :synopsis: SeqIndexerBaseEmbeddings is a basic abstract sequence indexers class that implements work qith embeddings.

.. moduleauthor:: Artem Chernodub
"""

import numpy as np
import torch

from seq_indexers.seq_indexer_base import SeqIndexerBase

class SeqIndexerBaseEmbeddings(SeqIndexerBase):
    def __init__(self, gpu, check_for_lowercase, zero_digits, pad, unk, load_embeddings, embeddings_dim, verbose):
        SeqIndexerBase.__init__(self, gpu, check_for_lowercase, zero_digits, pad, unk, load_embeddings, embeddings_dim,
                                verbose)

    def load_emb_dict_from_file(self, emb_fn, emb_delimiter):
        emb_dict = dict()
        for line in open(emb_fn, 'r'):
            values = line.split(emb_delimiter)
            word = values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            emb_dict[word] = emb_vector
        return emb_dict

    def generate_zero_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.embeddings_dim)]

    def generate_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim),
                                 self.embeddings_dim).tolist()

    def add_emb_vector(self, emb_vector):
        self.embedding_vectors_list.append(emb_vector)

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))
