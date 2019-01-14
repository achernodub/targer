"""base abstract class for sequence indexers"""
import numpy as np
import torch


class SeqIndexerBase():
    """
    SeqIndexerBase is a base abstract class for sequence indexers. It converts list of lists of string items
    to the list of lists of integer indices and back. Items could be either words, tags or characters.
    """
    def __init__(self, gpu=-1, check_for_lowercase=True, zero_digits=False, pad='<pad>', unk='<unk>',
                 load_embeddings=False, embeddings_dim=0, verbose=False):
        self.gpu = gpu
        self.check_for_lowercase = check_for_lowercase
        self.zero_digits = zero_digits
        self.pad = pad
        self.unk = unk
        self.load_embeddings = load_embeddings
        self.embeddings_dim = embeddings_dim
        self.verbose = verbose
        self.out_of_vocabulary_list = list()
        self.item2idx_dict = dict()
        self.idx2item_dict = dict()
        if load_embeddings:
            self.embeddings_loaded = False
            self.embedding_vectors_list = list()
        if pad is not None:
            self.pad_idx = self.add_item(pad)
            if load_embeddings:
                self.add_emb_vector(self.generate_zero_emb_vector())
        if unk is not None:
            self.unk_idx = self.add_item(unk)
            if load_embeddings:
                self.add_emb_vector(self.generate_random_emb_vector())

    def get_items_list(self):
        return list(self.item2idx_dict.keys())

    def get_items_count(self):
        return len(self.get_items_list())

    def item_exists(self, item):
        return item in self.item2idx_dict.keys()

    def add_item(self, item):
        idx = len(self.get_items_list())
        self.item2idx_dict[item] = idx
        self.idx2item_dict[idx] = item
        return idx

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return self.get_items_count() - 2
        if self.pad is not None or self.unk is not None:
            return self.get_items_count() - 1
        return self.get_items_count()

    def items2idx(self, item_sequences):
        idx_sequences = []
        for item_seq in item_sequences:
            idx_seq = list()
            for item in item_seq:
                if item in self.item2idx_dict:
                    idx_seq.append(self.item2idx_dict[item])
                else:
                    if self.unk is not None:
                        idx_seq.append(self.item2idx_dict[self.unk])
                    else:
                        idx_seq.append(self.item2idx_dict[self.pad])
            idx_sequences.append(idx_seq)
        return idx_sequences

    def idx2items(self, idx_sequences):
        item_sequences = []
        for idx_seq in idx_sequences:
            item_seq = [self.idx2item_dict[idx] for idx in idx_seq]
            item_sequences.append(item_seq)
        return item_sequences

    def items2tensor(self, item_sequences, align='left', word_len=-1):
        idx = self.items2idx(item_sequences)
        return self.idx2tensor(idx, align, word_len)

    def idx2tensor(self, idx_sequences, align='left', word_len=-1):
        batch_size = len(idx_sequences)
        if word_len == -1:
            word_len = max([len(idx_seq) for idx_seq in idx_sequences])
        # tensor = torch.zeros(batch_size, word_len, dtype=torch.long)
        if self.gpu >= 0:
            tensor = torch.cuda.LongTensor(batch_size, word_len).fill_(0)
        else:
            tensor = torch.LongTensor(batch_size, word_len).fill_(0)
        for k, idx_seq in enumerate(idx_sequences):
            curr_seq_len = len(idx_seq)
            if curr_seq_len > word_len:
                idx_seq = [idx_seq[i] for i in range(word_len)]
                curr_seq_len = word_len
            if align == 'left':
                tensor[k, :curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            elif align == 'center':
                start_idx = (word_len - curr_seq_len) // 2
                tensor[k, start_idx:start_idx+curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            else:
                raise ValueError('Unknown align string.')
        if self.gpu >= 0:
            tensor = tensor.cuda(device=self.gpu)
        return tensor
