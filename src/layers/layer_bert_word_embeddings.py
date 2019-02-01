"""class implements BERT word embeddings"""
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase
from pytorch_pretrained_bert import BertModel


class LayerBertWordEmbeddings(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, output_dim, output_bert_num=4):
        super(LayerBertWordEmbeddings, self).__init__(gpu)
        self.word_seq_indexer = word_seq_indexer
        self.gpu = gpu
        self.output_bert_num = output_bert_num
        self.bert_dim = 768 * output_bert_num
        self.output_dim = output_dim
        self.lin_layer = nn.Linear(in_features=self.bert_dim, out_features=output_dim)
        self.feature_cache = dict()

    def is_cuda(self):
        return self.lin_layer.weight.is_cuda

    def get_bert_feature(self, word_sequences):
        #tokens_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(word_sequences)) # shape: batch_size x max_seq_len
        #segments_tensor = self.tensor_ensure_gpu(torch.zeros(tokens_tensor.shape, dtype=torch.long))
        #bert_model = BertModel.from_pretrained('bert-base-cased')
        #if self.is_cuda():
        #    bert_model.cuda()
        tokens_tensor = self.word_seq_indexer.items2tensor(word_sequences).cpu() # shape: batch_size x max_seq_len
        segments_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long).cpu()
        bert_model = BertModel.from_pretrained('bert-base-cased')
        bert_model.cpu()
        bert_model.eval()
        y, _ = bert_model(tokens_tensor, segments_tensor)  # y : batch_size x max_seq_len x dim
        if self.output_bert_num == 4:
            bert_features = torch.cat((y[8], y[9], y[10], y[11]), dim=2) # 2 x 7 x 3072
        elif self.output_bert_num == 3:
            bert_features = torch.cat((y[9], y[10], y[11]), dim=2) # 2 x 7 x 3072
        elif self.output_bert_num == 2:
            bert_features = torch.cat((y[10], y[11]), dim=2) # 2 x 7 x 3072
        elif self.output_bert_num == 1:
            bert_features = y[11]
        else:
            raise NotImplementedError()
        compressed_bert_features = self.lin_layer(bert_features) # 2 x 7 x 300
        return compressed_bert_features

    def forward(self, word_sequences):
        batch_size = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        bert_features = self.tensor_ensure_gpu(torch.zeros(batch_size, max_seq_len, self.output_dim))
        for n, word_seq in enumerate(word_sequences):
            word_seq_key = '-'.join(word_seq)
            if word_seq_key in self.feature_cache:
                feature = self.feature_cache[word_seq_key]
            else:
                feature = self.get_bert_feature([word_seq])
            #print('bert_features.shape', bert_features.shape)
            #print('feature.shape', feature.shape)
            bert_features[n, :feature.shape[1], :] = self.tensor_ensure_gpu(feature)
        return bert_features
