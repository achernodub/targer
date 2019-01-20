"""class implements word embeddings"""
import torch.nn as nn
from src.layers.layer_base import LayerBase


class LayerWordEmbeddings(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerWordEmbeddings, self).__init__(gpu)
        embeddings_tensor = word_seq_indexer.get_loaded_embeddings_tensor()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_word_embeddings)
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = embeddings_tensor.shape[0]
        self.embeddings_dim = embeddings_tensor.shape[1]
        self.output_dim = self.embeddings_dim

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        input_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(word_sequences)) # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature
