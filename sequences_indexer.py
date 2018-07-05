import numpy as np
import torch


class SequencesIndexer():
    """
    SequencesIndexer creates dictionaries of integer indices for:
      1) tokens - uses dataset's input sequences and embeddings
      2) tags - uses dataset's outputs
    When dictionaries were created, sequences of tokens or tags can be converted to the sequences
     of integer indices and back.
    """

    def __init__(self, caseless=True, unk='<UNK>'):
        print('SequencesIndexer has been started.')
        self.caseless = caseless
        self.unk = unk
        self.embeddings_loaded = False
        self.embeddings_list = list()
        self.tags_list = list()
        self.token2idx_dict = dict()
        self.tag2idx_dict = dict()
        self.idx2token_dict = dict()
        self.idx2tag_dict = dict()
        self.tokens_num = 0
        self.embeddings_dim = 0
        self.tags_num = 0

    def load_embeddings(self, emb_fn, delimiter):
        if self.embeddings_loaded:
            raise ValueError('Embeddings are already loaded!')
        for line in open(emb_fn, 'r'):
            values = line.split(delimiter)
            token = values[0].lower() if self.caseless else values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.add_emb_vector(token, emb_vector)
        self.tokens_num = len(self.embeddings_list)
        self.embeddings_dim = len(emb_vector)
        self.embeddings_loaded = True
        print('%s embeddings file was loaded, %d vectors, dim = %d.' % (emb_fn, len(self.embeddings_list), self.embeddings_dim))
        # Generate random embedding for 'unknown' token
        self.add_emb_vector(self.unk, self.get_random_emb_vector())

    def get_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim was not set.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim), self.embeddings_dim).tolist()

    def add_emb_vector(self, token, emb_vector):
        idx = len(self.token2idx_dict)
        self.token2idx_dict[token] = idx
        self.idx2token_dict[idx] = token
        self.embeddings_list.append(emb_vector)

    def add_token_sequences(self, token_sequences):
        if not self.embeddings_loaded:
            raise ValueError('Embeddings were not loaded, please, load them first!')
        for tokens in token_sequences:
            for token in tokens:
                if self.caseless:
                    token = token.lower()
                if token not in self.token2idx_dict:
                    self.add_emb_vector(token, self.get_random_emb_vector())
        self.tokens_num = len(self.embeddings_list)

    def add_tag_sequences(self, tag_sequences):
        for tags in tag_sequences:
            for tag in tags:
                if tag not in self.tags_list:
                    self.tags_list.append(tag)
                    idx = len(self.tag2idx_dict)
                    self.tag2idx_dict[tag] = idx
                    self.idx2tag_dict[idx] = tag

    def token2idx(self, token_sequences):
        idx_sequences = list()
        for tokens in token_sequences:
            curr_idx_seq = list()
            for token in tokens:
                if self.caseless:
                    token = token.lower()
                curr_idx_seq.append(self.token2idx_dict[token])
            idx_sequences.append(curr_idx_seq)
        return idx_sequences

    def idx2token(self, idx_sequences):
        token_sequences = list()
        for indices in idx_sequences:
            curr_token_seq = list()
            for idx in indices:
                curr_token_seq.append(self.idx2token_dict[idx])
            token_sequences.append(curr_token_seq)
        return token_sequences

    def tag2idx(self, tag_sequences):
        idx_sequences = list()
        for tags in tag_sequences:
            curr_idx_seq = list()
            for tag in tags:
                if tag in self.tags_list:
                    curr_idx_seq.append(self.tag2idx_dict[tag])
                else:
                    curr_idx_seq.append(self.tag2idx_dict[self.unk])
            idx_sequences.append(curr_idx_seq)
        return idx_sequences

    def get_token_list(self):
        return self.token2idx_dict.keys()

    def get_tags_list(self):
        return self.tag2idx_dict.keys()

    def get_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embeddings_list))