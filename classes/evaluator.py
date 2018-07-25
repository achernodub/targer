import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluator():
    def __init__(self, sequences_indexer=None):
        self.sequences_indexer = sequences_indexer

    def get_macro_scores_inputs_tensor_targets_idx(self, tagger, inputs_tensor, targets_idx):
        tags_list_idx = [i for i in range(1, self.sequences_indexer.get_tags_num() + 1)]
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        accuracy_sum, f1_sum, precision_sum, recall_sum = 0, 0, 0, 0
        for y_true, y_pred in zip(targets_idx, outputs_idx):
            accuracy_sum += accuracy_score(y_true, y_pred) * 100
            f1_sum += f1_score(y_true, y_pred, average='macro', labels=tags_list_idx) * 100
            precision_sum += precision_score(y_true, y_pred, average='macro', labels=tags_list_idx) * 100
            recall_sum += recall_score(y_true, y_pred, average='macro', labels=tags_list_idx) * 100
        data_num = len(targets_idx)
        return accuracy_sum / data_num, f1_sum / data_num, precision_sum / data_num, recall_sum / data_num

    def get_macro_f1_scores_details(self, tagger, token_sequences, tag_sequences):
        outputs_idx = tagger.predict_idx_from_tokens(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        data_num = len(token_sequences)
        tags_num =  self.sequences_indexer.get_tags_num()
        tags_list_idx = [i for i in range(1, tags_num + 1)]
        f1_scores_array = np.zeros([data_num, tags_num], dtype=float)
        #f1_mean = 0
        for k, (y_true, y_pred) in enumerate(zip(targets_idx, outputs_idx)):
            f1_scores_array[k, :] = f1_score(y_true, y_pred, average=None, labels=tags_list_idx) * 100
            #f1_mean += f1_score(y_true, y_pred, average='macro', labels=tags_list_idx) * 100
        f1_scores_mean = np.mean(f1_scores_array, axis=0)
        str = 'Tag             | F1\n----------------------\n'
        for n in range(1, tags_num + 1):
            tag = self.sequences_indexer.idx2tag_dict[n]  # minumum tag no. is "1"
            str += '%015s |  %1.2f\n' % (tag, f1_scores_mean[n-1])
        str += '-----------------\n%015s |  %1.2f\n' % ('F1', np.mean(f1_scores_mean))
        return str

    def is_tensor(self, X):
        return isinstance(X[0][0], torch.Tensor)

    def is_idx(self, X):
        return isinstance(X[0][0], int)

    def is_str(self, X):
        return isinstance(X[0][0], str)

    def get_macro_scores(self, tagger, inputs, targets):
        if self.is_tensor(inputs) and self.is_tensor(targets):
            return self.get_macro_scores_inputs_tensor_targets_tensor(tagger, inputs, targets)
        elif self.is_tensor(inputs) and self.is_idx(targets):
            return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs, targets)
        elif self.is_idx(inputs) and self.is_idx(targets):
            return self.get_macro_scores_inputs_idx_targets_idx(tagger, inputs, targets)
        elif self.is_str(inputs) and self.is_str(targets):
            return self.get_macro_scores_tokens_tags(tagger, inputs, targets)
        else:
            raise ValueError('Unknown combination of inputs and targets')

    def get_macro_scores_inputs_tensor_targets_tensor(self, tagger, inputs_tensor, targets_tensor):
        targets_idx = self.sequences_indexer.tensor2idx(targets_tensor)
        return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_macro_scores_inputs_idx_targets_idx(self, tagger, inputs_idx, targets_idx):
        inputs_tensor = self.sequences_indexer.idx2tensor(inputs_idx)
        return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_macro_scores_tokens_tags(self, tagger, token_sequences, tag_sequences):
        inputs_idx = self.sequences_indexer.token2idx(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        return self.get_macro_scores_inputs_idx_targets_idx(tagger, inputs_idx, targets_idx)

    def write_report(self, fn, args, tagger, token_sequences, tag_sequences):

        text_file = open(fn, mode='w')
        for hyper_param in str(args).replace('Namespace(', '').replace(')', '').split(', '):
            text_file.write('%s\n' % hyper_param)

        acc_test, f1_test, precision_test, recall_test = self.get_macro_scores(tagger=tagger,
                                                                               inputs=token_sequences,
                                                                               targets=tag_sequences)

        text_file.write('\nResults on TEST: Accuracy = %1.2f, MACRO F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n\n' % (
                                                             acc_test, f1_test, precision_test, recall_test))
        text_file.write(self.get_macro_f1_scores_details(tagger, token_sequences, tag_sequences))
        text_file.close()

    '''def get_macro_scores_inputs_tensor_targets_idx_flat(self, tagger, inputs_tensor, targets_idx):
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        accuracy = accuracy_score(y_true, y_pred)*100
        f1 = f1_score(y_true, y_pred, average='macro')*100
        precision = precision_score(y_true, y_pred, average='macro')*100
        recall = recall_score(y_true, y_pred, average='macro')*100
        return accuracy, f1, precision, recall'''

    '''def get_macro_f1_scores_details_flat(self, tagger, token_sequences, tag_sequences):
    outputs_idx = tagger.predict_idx_from_tokens(token_sequences)
    targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
    y_true = [i for sequence in targets_idx for i in sequence]
    y_pred = [i for sequence in outputs_idx for i in sequence]
    f1_scores = f1_score(y_true, y_pred, average=None)*100
    str = 'Tag    | MACRO-F1\n-----------------\n'
    for n in range(self.sequences_indexer.get_tags_num()):
        tag = self.sequences_indexer.idx2tag_dict[n+1]  # minumum tag no is "1"
        str += '%006s |  %1.2f\n' % (tag, f1_scores[n])
    str += '-----------------\n'
    str += '%006s |  %1.2f\n' % ('F1', np.mean(f1_scores))
    return str'''