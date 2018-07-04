class Indexer():
    """
    Indexer creates dictionaries of integer indices for:
      1) tokens - uses dataset's input sequences and embeddings
      2) tags - uses dataset's outputs
    When dictionaries were created, sequences of tokens or tags can be converted to the sequences
     of integer indices and back.
    """

    def __init__(self):
        print('Hello Indexer!')
        self.embeddings_loaded = False
        self.embeddings_list = list()
        self.tokens_list = list()
        self.tags_dict = list()
        pass

    def load_embeddings(self, emb_fn, delimiter, caseless=True):
        if self.embeddings_loaded:
            raise ValueError('Embeddings are already loaded!')
        for line in open(emb_fn, 'r'):
            values = line.split(delimiter)
            token = values[0].lower() if caseless else values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.embeddings_list.append(emb_vector)
        self.embeddings_dim = len(emb_vector)

    def add_tokens_sequences(self, sequences):
        if not self.embeddings_loaded:
            raise ValueError('Embeddings were not loaded, please, load them first!')
        pass

    def add_tags_sequences(self, tags_sequences):
        if not self.embeddings_loaded:
            raise ValueError('Embeddings were not loaded, please, load them first!')
        pass




