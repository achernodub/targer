import torch.nn as nn


class TaggerBase(nn.Module):
    """
    TaggerBase is a base class for tagger models. It implements the tagging functionality for different types of
     inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class SequencesIndexer is used
     for input and output data formats conversions. Abstract method `forward` is used in order to make these predictions,
     it have to be implemented in ancestors.
    """
    def __init__(self,  sequences_indexer, gpu):
        super(TaggerBase, self).__init__()
        self.sequences_indexer = sequences_indexer

    def make_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def clip_gradients(self, clip_grad):
        nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

    def predict_idx_from_tensor(self, inputs_tensor):# inputs_tensor: batch_size x max_seq_len
        self.eval()
        outputs_tensor = self.forward(inputs_tensor) # batch_size x num_class+1 x max_seq_len
        outputs_idx = list()
        batch_size, max_seq_len = inputs_tensor.size()
        for k in range(batch_size):
            list_idx = list()
            for l in range(max_seq_len):
                curr_input = inputs_tensor[k, l].item()
                if curr_input > 0: # ignore zero-padded inputs of sequence
                    curr_output = outputs_tensor[k, 1:, l] # ignore the first output, reason is the same
                    _, max_no = curr_output.max(0) # argmax
                    list_idx.append(max_no.item() + 1)
            outputs_idx.append(list_idx)
        return outputs_idx

    def predict_idx_from_idx(self, inputs_idx):
        inputs_tensor = self.sequences_indexer.idx2tensor(inputs_idx)
        return self.predict_idx_from_tensor(inputs_tensor)

    def predict_idx_from_tokens(self, token_sequences):
        inputs_idx = self.sequences_indexer.token2idx(token_sequences)
        return self.predict_idx_from_idx(inputs_idx)

    def predict_tags_from_tensor(self, inputs_tensor):
        outputs_idx = self.predict_idx_from_tensor(inputs_tensor)
        return self.sequences_indexer.idx2tag(outputs_idx)

    def predict_tags_from_idx(self, inputs_idx):
        inputs_tensor = self.sequences_indexer.idx2tensor(inputs_idx)
        return self.predict_tags_from_tensor(inputs_tensor)

    def predict_tags_from_tokens(self, token_sequences):
        inputs_idx = self.sequences_indexer.token2idx(token_sequences)
        return self.predict_tags_from_idx(inputs_idx)
