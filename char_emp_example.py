import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
import string

"""
Since character embeddings are a bit weak in pytorch 3, this will hopefully help out
I think these should be trainable and also, invertable!
So you can actually recover output from the embeddings using Cos Similarity
"""

class CharacterEmbedding:
    def __init__(self, embedding_size):

        self.vocab = ['<pad>'] + list(string.printable) + ['<SOS>', '<EOS>']
        self.embed = nn.Embedding(len(self.vocab), embedding_size)
        self.is_cuda = False
        self.cos = nn.CosineSimilarity(dim=2)

    def flatten(self, l):
        return list(itertools.chain.from_iterable(l))

    def embedAndPack(self, seqs, batch_first=False):

        vectorized_seqs = [[self.vocab.index(tok) for tok in seq]for seq in seqs]

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_lengths = seq_lengths.cuda() if self.is_cuda else seq_lengths

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
        seq_tensor = seq_tensor.cuda() if self.is_cuda else seq_tensor

        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda() if self.is_cuda else torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
        # Otherwise, give (L,B,D) tensors
        if not batch_first:
            seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

        # embed your sequences
        seq_tensor = self.embed(seq_tensor)

        # pack them up nicely
        return pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

    def cuda(self):
        self.is_cuda = True
        self.embed = self.embed.cuda()
        return self

    def unpackToSequence(self, packed_output):
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        words = self.unembed(output)
        return words

    def unembed(self, embedded_sequence):
        weights = self.embed.state_dict()['weight']
        weights = weights.transpose(0,1).unsqueeze(0).unsqueeze(0)
        e_sequence = embedded_sequence.unsqueeze(3).data
        cosines = self.cos(e_sequence, weights)
        _, indexes = torch.topk(cosines, 1, dim=2)

        words = []
        for word in indexes:
            word_l = ''
            for char_index in word:
                word_l += self.vocab[char_index[0]]
            words.append(word_l)
        return words









if __name__ == '__main__':

    seqs = ['ghatmasala','nicela','c-pakodas']

    # make model

    embedding = CharacterEmbedding(embedding_size=5).cuda()

    lstm = nn.LSTM(5, 5, batch_first=True).cuda()

    packed_input = embedding.embedAndPack(seqs, batch_first=True)
    words = embedding.unpackToSequence(packed_input)

    print(words)

    # throw them through your LSTM (remember to give batch_first=True here if you packed with it)
    packed_output, (ht, ct) = lstm(packed_input)

    words = embedding.unpackToSequence(packed_output)


    print(words)


