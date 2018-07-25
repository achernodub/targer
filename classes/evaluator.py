import torch
import numpy as np
from sklearn.metrics import accuracy_score # f1_score, precision_score, recall_score


class Evaluator():
    @staticmethod
    def get_accuracy_token_level(targets_idx, outputs_idx):
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        return accuracy_score(y_true, y_pred) * 100

    @staticmethod
    def get_f1_idx(targets_idx, outputs_idx):
        return Evaluator.get_accuracy_token_level(targets_idx, outputs_idx)


    '''
    def get_f1_micro(self, targets_idx, outputs_idx):
        TP, FN, FP = 0, 0, 0
        for curr_targets, curr_outputs in zip(targets_idx, outputs_idx):
            for k, t in enumerate(curr_targets):
                if t == curr_outputs[k]:
                    TP += 1
                else:
                    FN += 1
            for k, o in enumerate(curr_outputs):
                if o != curr_targets[k]:
                    FP += 1

        print(TP, FN, FP)

        P = TP/max(TP+FP, 1)
        R = TP/max(TP+FN, 1)
        F1 = 2*TP/max(2*TP+FP+FN, 1)

        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        accuracy = accuracy_score(y_true, y_pred)*100

        print(F1, P, R, accuracy)
    '''


    @staticmethod
    def get_f1(targets_tag_components_sequences, outputs_tag_components_sequences):
        TP, FN, FP = 0, 0, 0
        for targets_tag_components, outputs_tag_components in zip(targets_tag_components_sequences, outputs_tag_components_sequences):
            for target_tc in targets_tag_components:
                found = False
                for output_tc in outputs_tag_components:
                    if output_tc.is_equal(target_tc):
                        found = True
                        break
                if found:
                    TP += 1
                else:
                    FN += 1
            for output_tc in outputs_tag_components:
                found = False
                for target_tc in targets_tag_components:
                    if target_tc.is_equal(output_tc):
                        found = True
                        break
                if not found:
                    FP += 1
        #P = TP / max(TP + FP, 1)
        #R = TP / max(TP + FN, 1)
        F1 = (2 * TP / max(2 * TP + FP + FN, 1))*100
        return F1

    @staticmethod
    def get_f1_scores_details(tagger, token_sequences, tag_sequences):
        str = 'get_f1_scores_details - blank'
        '''outputs_idx = tagger.predict_idx_from_tokens(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        data_num = len(token_sequences)
        tags_num =  self.sequences_indexer.get_tags_num()
        tags_list_idx = [i for i in range(1, tags_num + 1)]
        f1_scores_array = np.zeros([data_num, tags_num], dtype=float)
        #f1_mean = 0
        for k, (y_true, y_pred) in enumerate(zip(targets_idx, outputs_idx)):
            f1_scores_array[k, :] = f1_score(y_true, y_pred, average=None, labels=tags_list_idx) * 100
        f1_scores_mean = np.mean(f1_scores_array, axis=0)
        str = 'Tag             | F1\n----------------------\n'
        for n in range(1, tags_num + 1):
            tag = self.sequences_indexer.idx2tag_dict[n]  # minumum tag no. is "1"
            str += '%015s |  %1.2f\n' % (tag, f1_scores_mean[n-1])
        str += '-----------------\n%015s |  %1.2f\n' % ('F1', np.mean(f1_scores_mean))'''
        return str

    @staticmethod
    def write_report(fn, args, tagger, token_sequences, tag_sequences):
        pass
        '''
        text_file = open(fn, mode='w')
        for hyper_param in str(args).replace('Namespace(', '').replace(')', '').split(', '):
            text_file.write('%s\n' % hyper_param)

        acc_test, f1_test, precision_test, recall_test = self.get_scores(tagger=tagger,
                                                                         inputs=token_sequences,
                                                                         targets=tag_sequences)

        text_file.write('\nResults on TEST: Accuracy = %1.2f, F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n\n' % (
                                                             acc_test, f1_test, precision_test, recall_test))
        text_file.write(self.get_f1_scores_details(tagger, token_sequences, tag_sequences))
        text_file.close()'''




    '''
    def get_scores_inputs_tensor_targets_idx0(self, tagger, inputs_tensor, targets_idx, average='micro'):
        tags_list_idx = [i for i in range(1, self.sequences_indexer.get_tags_num() + 1)]
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        accuracy_sum, f1_sum, precision_sum, recall_sum = 0, 0, 0, 0
        for y_true, y_pred in zip(targets_idx, outputs_idx):
            accuracy_sum += accuracy_score(y_true, y_pred) * 100
            f1_sum += f1_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100
            precision_sum += precision_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100
            recall_sum += recall_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100
        data_num = len(targets_idx)
        return accuracy_sum / data_num, f1_sum / data_num, precision_sum / data_num, recall_sum / data_num

    def get_scores_inputs_tensor_targets_idx(self, tagger, inputs_tensor, targets_idx, average='micro'):
        tags_list_idx = [i for i in range(1, self.sequences_indexer.get_tags_num() + 1)]
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        accuracy = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100
        precision = precision_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100
        recall = recall_score(y_true, y_pred, average=average, labels=tags_list_idx) * 100

        for n in tags_list_idx:
            curr_f1 = f1_score(y_true, y_pred, labels = [n], average=average)
            print('%s %1.2f' % (self.sequences_indexer.idx2tag_dict[n], curr_f1*100))

        return accuracy, f1, precision, recall

    def get_f1_scores_details(self, tagger, token_sequences, tag_sequences):
        outputs_idx = tagger.predict_idx_from_tokens(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        data_num = len(token_sequences)
        tags_num =  self.sequences_indexer.get_tags_num()
        tags_list_idx = [i for i in range(1, tags_num + 1)]
        f1_scores_array = np.zeros([data_num, tags_num], dtype=float)
        #f1_mean = 0
        for k, (y_true, y_pred) in enumerate(zip(targets_idx, outputs_idx)):
            f1_scores_array[k, :] = f1_score(y_true, y_pred, average=None, labels=tags_list_idx) * 100
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

    def get_scores(self, tagger, inputs, targets):
        if self.is_tensor(inputs) and self.is_tensor(targets):
            return self.get_scores_inputs_tensor_targets_tensor(tagger, inputs, targets)
        elif self.is_tensor(inputs) and self.is_idx(targets):
            return self.get_scores_inputs_tensor_targets_idx(tagger, inputs, targets)
        elif self.is_idx(inputs) and self.is_idx(targets):
            return self.get_scores_inputs_idx_targets_idx(tagger, inputs, targets)
        elif self.is_str(inputs) and self.is_str(targets):
            return self.get_scores_tokens_tags(tagger, inputs, targets)
        else:
            raise ValueError('Unknown combination of inputs and targets')

    def get_scores_inputs_tensor_targets_tensor(self, tagger, inputs_tensor, targets_tensor):
        targets_idx = self.sequences_indexer.tensor2idx(targets_tensor)
        return self.get_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_scores_inputs_idx_targets_idx(self, tagger, inputs_idx, targets_idx):
        inputs_tensor = self.sequences_indexer.idx2tensor(inputs_idx)
        return self.get_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_scores_tokens_tags(self, tagger, token_sequences, tag_sequences):
        inputs_idx = self.sequences_indexer.token2idx(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        return self.get_scores_inputs_idx_targets_idx(tagger, inputs_idx, targets_idx)

    def write_report(self, fn, args, tagger, token_sequences, tag_sequences):

        text_file = open(fn, mode='w')
        for hyper_param in str(args).replace('Namespace(', '').replace(')', '').split(', '):
            text_file.write('%s\n' % hyper_param)

        acc_test, f1_test, precision_test, recall_test = self.get_scores(tagger=tagger,
                                                                         inputs=token_sequences,
                                                                         targets=tag_sequences)

        text_file.write('\nResults on TEST: Accuracy = %1.2f, F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n\n' % (
                                                             acc_test, f1_test, precision_test, recall_test))
        text_file.write(self.get_f1_scores_details(tagger, token_sequences, tag_sequences))
        text_file.close()
    '''

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