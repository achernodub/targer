"""converts list of lists of words as strings to list of lists of integer indices and back + BERT tokenizer"""
import numpy
import torch
from pytorch_pretrained_bert import BertTokenizer


class SeqIndexerWordBert():
    """SeqIndexerWordBert converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self, gpu=-1, pad='<pad>', unk='<unk>', verbose=True):
        self.gpu = gpu
        self.pad = pad
        self.unk = unk
        self.verbose = verbose
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    def items2tensor(self, word_sequences):
        idx_sequences = list()
        for word_seq in word_sequences:
            idx_seq = list()
            #print('----------------')
            for w in word_seq:
                w_tokenized = [self.bert_tokenizer.tokenize(w)[0]]
                idx = self.bert_tokenizer.convert_tokens_to_ids(w_tokenized)
                idx_seq.extend(idx)
                #sprint('w =', w, 'w_tokenized =', w_tokenized, 'idx =', idx, 'idx_seq =', idx_seq)
            idx_sequences.append(idx_seq)
        return self.idx2tensor(idx_sequences)

    def idx2tensor(self, idx_sequences):
        batch_size = len(idx_sequences)
        max_seq_len = max([len(idx_seq) for idx_seq in idx_sequences])
        tensor = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for n, idx_seq in enumerate(idx_sequences):
            tensor[n, :len(idx_seq)] = torch.tensor(idx_seq)
        if self.gpu >= 0:
            tensor = tensor.cuda(device=self.gpu)
        return tensor

