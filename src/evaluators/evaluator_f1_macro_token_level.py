"""macro-F1 scores evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorF1MacroTokenLevel(EvaluatorBase):
    def __init__(self):
        self.tag_list = None
        self.tag2idx = dict()

    def __init_tag_list(self, targets_tag_sequences):
        if self.tag_list is not None:
            return
        self.tag_list = list()
        for tag_seq in targets_tag_sequences:
            for t in tag_seq:
                if t not in self.tag_list:
                    self.tag_list.append(t)
                    self.tag2idx[t] = len(self.tag_list)
        self.tag_list.sort()

    def tag_seq_2_idx_list(self, tag_seq):
        return [self.tag2idx[t] for t in tag_seq]

    def __get_zeros_tag_dict(self):
        return {tag: 0 for tag in self.tag_list}

    def __add_dict(self, dict1, dict2):
        for tag in self.tag_list:
            dict1[tag] += dict2[tag]
        return dict1

    def __div_dict(self, dict, d):
        for tag in self.tag_list:
            dict[tag] /= d
        return dict

    def __get_M_F1_msg(self, F1):
        msg = '\nF1 scores\n'
        msg += '-' * 24 + '\n'
        sum_M_F1 = 0
        for tag in self.tag_list:
            sum_M_F1 += F1[tag]
            msg += '%15s = %1.2f\n' % (tag, F1[tag])
        M_F1 = sum_M_F1 / len(F1)
        msg += '-'*24 + '\n'
        msg += 'Macro-F1 = %1.3f' % M_F1
        return M_F1, msg

    def __add_to_dict(self, dict_in, tag, val):
        if tag in dict_in:
            dict_in[tag] += val
        else:
            dict_in[tag] = val
        return dict_in

    """EvaluatorF1MacroTagComponents is macro-F1 scores evaluator for each class of BOI-like tags."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        # Create list of tags
        self.__init_tag_list(targets_tag_sequences)
        # Init values
        TP = self.__get_zeros_tag_dict()
        FP = self.__get_zeros_tag_dict()
        FN = self.__get_zeros_tag_dict()
        F1 = self.__get_zeros_tag_dict()
        for targets_seq, outputs_tag_seq in zip(targets_tag_sequences, outputs_tag_sequences):
            for t, o in zip(targets_seq, outputs_tag_seq):
                if t == o:
                    TP = self.__add_to_dict(TP, t, 1)
                else:
                    FN = self.__add_to_dict(FN, t, 1)
                    FP = self.__add_to_dict(FP, o, 1)
        # Calculate F1 for each tag
        for tag in self.tag_list:
            F1[tag] = (2 * TP[tag] / max(2 * TP[tag] + FP[tag] + FN[tag], 1)) * 100
        # Calculate Macro-F1 score and prepare the message
        M_F1, msg = self.__get_M_F1_msg(F1)
        print(msg)
        #self.validate_M_F1_scikitlearn( targets_tag_sequences, outputs_tag_sequences)
        return M_F1, msg

    '''# for valid ation
    def validate_M_F1_scikitlearn(self, targets_tag_sequences, outputs_tag_sequences):
        from sklearn.metrics import f1_score
        targets_tag_sequences_flat = [t for targets_tag_seq in targets_tag_sequences for t in targets_tag_seq]
        outputs_tag_sequences_flat = [o for outputs_tag_seq in outputs_tag_sequences for o in outputs_tag_seq]
        y_true = self.tag_seq_2_idx_list(targets_tag_sequences_flat)
        y_pred = self.tag_seq_2_idx_list(outputs_tag_sequences_flat)
        M_F1_scikitlearn = f1_score(y_true=y_true, y_pred=y_pred, average='macro', sample_weight=None)*100
        print('Macro-F1_scikitlearn = %1.3f, for validation' % M_F1_scikitlearn)'''

