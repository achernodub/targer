import numpy as np
import torch


class SequencesIndexer():
    """
    SequencesIndexer creates dictionaries of integer indices for:
      1) tokens - uses dataset's input sequences and embeddings
      2) tags - uses dataset's outputs
    When dictionaries were created, sequences of tokens or tags can be converted to the sequences
     of integer indices and back. Minimum value for any index is "1".
    """

    def __init__(self, caseless=True, verbose=False, unk='<UNK>'):
        self.caseless = caseless
        self.verbose = verbose
        self.unk = unk
        self.embeddings_loaded = False
        self.embeddings_list = list()
        self.tags_list = list()
        self.token2idx_dict = dict()
        self.tag2idx_dict = dict()
        self.idx2token_dict = dict()
        self.idx2tag_dict = dict()
        self.embeddings_dim = 0
        self.tokens_out_of_vocabulary_list = list()
        if self.verbose:
            print('SequencesIndexer has been started.')


    def load_embeddings(self, emb_fn, delimiter):
        if self.embeddings_loaded:
            raise ValueError('Embeddings are already loaded!')
        for line in open(emb_fn, 'r'):
            values = line.split(delimiter)
            token = values[0].lower() if self.caseless else values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.add_emb_vector(token, emb_vector)
        self.embeddings_dim = len(emb_vector)
        self.embeddings_loaded = True
        # Generate random embedding for 'unknown' token
        self.add_emb_vector(self.unk, self.get_random_emb_vector())
        if self.verbose:
            print('%s embeddings file was loaded, %d vectors, dim = %d.' % (emb_fn, len(self.embeddings_list), self.embeddings_dim))

    def get_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim was not set.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim), self.embeddings_dim).tolist()

    def add_emb_vector(self, token, emb_vector):
        idx = len(self.token2idx_dict) + 1
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
                    self.tokens_out_of_vocabulary_list.append(token)
        if self.verbose:
            print('%d tokens not found, random embeddings were generated:' % len(self.tokens_out_of_vocabulary_list))
            for k, out_of_vocabulary_token in enumerate(self.tokens_out_of_vocabulary_list):
                print(' -= %d/%d out of vocabulary token: %s' % (k, len(self.tokens_out_of_vocabulary_list), out_of_vocabulary_token))

    def add_tag_sequences(self, tag_sequences):
        for tags in tag_sequences:
            for tag in tags:
                if tag not in self.tags_list:
                    self.tags_list.append(tag)
                    idx = len(self.tag2idx_dict) + 1
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

    def get_tokens_num(self):
        return len(self.embeddings_list)

    def get_tags_num(self):
        return len(self.tags_list)

    def idx2tensor(self, list_idx):
        data_num = len(list_idx)
        max_seq_len = max([len(seq) for seq in list_idx])
        tensor = torch.zeros(data_num, max_seq_len, dtype=torch.long)
        for k, curr_input_idx in enumerate(list_idx):
            curr_seq_len = len(curr_input_idx)
            tensor[k, :curr_seq_len] = torch.FloatTensor(np.asarray(curr_input_idx))
        return tensor

    def tensor2idx(self, tensor):
        list_idx = list()
        data_num = tensor.shape[0]
        for k in range(data_num):
            curr_row = tensor[k, :]
            nonzerrro = curr_row.nonzero()
            print('curr_row.nonzero()', curr_row.nonzero())
            curr_row_nonzero = curr_row[curr_row.nonzero()]
            curr_seq_len = curr_row_nonzero.shape[0]
            list_idx.append([int(curr_row_nonzero[k]) for k in range(curr_seq_len)])
        return list_idx