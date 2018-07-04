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
        pass

    def load_embeddings(self):
        if self.embeddings_loaded:
            raise ValueError('Embeddings are already loaded!')
        pass

    def add_sequences(self):
        if not self.embeddings_loaded:
            raise ValueError('Embeddings were not loaded, please, load them first!')
        pass



