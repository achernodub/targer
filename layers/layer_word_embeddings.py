import torch.nn as nn
from classes.element_seq_indexer import ElementSeqIndexer

class LayerWordEmbeddings(nn.Module):
    def __init__(self, word_seq_indexer, freeze_word_embeddings=False):
        super(LayerWordEmbeddings, self).__init__()
        embeddings_tensor = word_seq_indexer.get_embeddings_tensor()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_word_embeddings)
        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = embeddings_tensor.shape[0]
        self.embeddings_dim = embeddings_tensor.shape[1]
        self.feature_dim = self.embeddings_dim

    def forward(self, word_sequences):
        input_tensor = self.word_seq_indexer.elements2tensor(word_sequences) # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x feature_dim
        return word_embeddings_feature