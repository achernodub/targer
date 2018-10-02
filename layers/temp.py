import torch
import torch.nn as nn

# A list of sentences, each being a list of tokens.
sents = [[4, 545, 23, 1], [34, 84], [23, 6, 774]]
# Embedding for 10k words with d=128
emb = nn.Embedding(1000, 128)
# When packing a sequence it has to be sorted on length.
sents.sort(key=len, reverse=True)
packed = nn.utils.rnn.pack_sequence(
    [torch.tensor(s) for s in sents])
embedded = nn.utils.rnn.PackedSequence(
    emb(packed.data), packed.batch_sizes)
# An LSTM
lstm_layer = nn.LSTM(128, 128)
output = lstm_layer(embedded)
# Output is a PackedSequence too

