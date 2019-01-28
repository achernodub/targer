"""pure softmax + linear layer model for sequences tagging, for BERT only."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bert_word_embeddings import LayerBertWordEmbeddings
from src.layers.layer_bivanilla import LayerBiVanilla
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_bigru import LayerBiGRU


class TaggerSoftmax(TaggerBase):
    """TaggerSoftmax is a pure softmax + linear layer model for sequences tagging, for BERT only."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, emb_dim=100,
                 dropout_ratio=0.5, gpu=-1, emb_bert=True):
        super(TaggerSoftmax, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.emb_dim = emb_dim
        self.dropout_ratio = dropout_ratio
        self.gpu = gpu
        self.emb_bert = emb_bert
        if emb_bert:
            self.word_embeddings_layer = LayerBertWordEmbeddings(word_seq_indexer, gpu, emb_dim)
        else:
            raise NotImplementedError()
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.word_embeddings_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.emb_bert = emb_bert
        self.nll_loss = nn.NLLLoss(ignore_index=0) # "0" target values actually are zero-padded parts of sequences

    def forward(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        z_rnn_out = self.apply_mask(self.lin_layer(rnn_output_h), mask) # shape: batch_size x class_num + 1 x max_seq_len
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))
        return y

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss
