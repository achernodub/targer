"""converts list of lists of characters to list of lists of integer indices and back"""
from src.seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings


class SeqIndexerBaseChar(SeqIndexerBaseEmbeddings):
    """SeqIndexerBaseChar converts list of lists of characters to list of lists of integer indices and back."""
    def __init__(self, gpu):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=False, zero_digits=False, pad='<pad>',
                                          unk='<unk>', load_embeddings=False, embeddings_dim=0, verbose=True)

    def add_char(self, c):
        if not self.item_exists(c):
            self.add_item(c)

    def get_char_tensor(self, curr_char_seq, word_len):
        return SeqIndexerBaseEmbeddings.items2tensor(self, curr_char_seq, align='center', word_len=word_len)  # curr_seq_len x word_len
