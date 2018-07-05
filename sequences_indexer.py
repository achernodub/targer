class SequencesIndexer():
    """
    Indexer creates dictionaries of integer indices for:
      1) tokens - uses dataset's input sequences and embeddings
      2) tags - uses dataset's outputs
    When dictionaries were created, sequences of tokens or tags can be converted to the sequences
     of integer indices and back.
    """

    def __init__(self, caseless=True, unk='<UNK>'):
        print('Indexer has been launched.')
        self.caseless = caseless
        self.unk = unk
        self.embeddings_loaded = False
        self.embeddings_list = list()
        self.tokens_list = list()
        self.tags_list = list()
        self.token2idx_dict = dict()
        self.tag2idx_dict = dict()
        self.tokens_num = 0
        self.tags_num = 0
        self.token2idx_dict[unk] = 0

    def load_embeddings(self, emb_fn, delimiter):
        if self.embeddings_loaded:
            raise ValueError('Embeddings are already loaded!')
        for line in open(emb_fn, 'r'):
            values = line.split(delimiter)
            token = values[0].lower() if self.caseless else values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.embeddings_list.append(emb_vector)
            self.token2idx_dict[token] = len(self.token2idx_dict)
        self.embeddings_dim = len(emb_vector)
        self.embeddings_loaded = True
        print('%s embeddings file was loaded, %d vectors, dim = %d.' % (emb_fn, len(self.embeddings_list), self.embeddings_dim))

    def add_token_sequences(self, token_sequences):
        if not self.embeddings_loaded:
            raise ValueError('Embeddings were not loaded, please, load them first!')
        for tokens in token_sequences:
            for token in tokens:
                if self.caseless:
                    token = token.lower()
                if token not in self.tokens_list:
                    self.tokens_list.append(token)

    def add_tag_sequences(self, tag_sequences):
        for tags in tag_sequences:
            for tag in tags:
                if tag not in self.tags_list:
                    self.tags_list.append(tag)
                    self.tag2idx_dict[tag] = len(self.tag2idx_dict)

    def token2idx(self, token_sequences):
        idx_sequences = list()
        for tokens in token_sequences:
            curr_idx_seq = list()
            for token in tokens:
                if self.tok
                curr_idx_seq.append(self.token2idx_dict[token])
            idx_sequences.append(curr_idx_seq)
        return idx_sequences

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







