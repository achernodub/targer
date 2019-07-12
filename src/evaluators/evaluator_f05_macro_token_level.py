"""macro-F05 scores + Prec + Recall evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorF05MacroTokenLevel(EvaluatorBase):
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

    def __get_M_F05_msg(self, F05):
        msg = '\nF05 scores\n'
        msg += '-' * 24 + '\n'
        sum_M_F05 = 0
        for tag in self.tag_list:
            sum_M_F05 += F05[tag]
            msg += '%15s = %1.2f\n' % (tag, F05[tag])
        M_F05 = sum_M_F05 / len(F05)
        msg += '-'*24 + '\n'
        msg += 'Macro-F05 = %1.3f' % M_F05
        return M_F05, msg

    def __add_to_dict(self, dict_in, tag, val):
        if tag in dict_in:
            dict_in[tag] += val
        else:
            dict_in[tag] = val
        return dict_in

    def __get_f_beta(self, tp, fn, fp, beta=0.5):
        return (1 + beta*beta)*tp*100.0 / max((1 + beta*beta)*tp + (beta*beta)*fn + fp, 1)

    """EvaluatorF05MacroTagComponents is macro-F05 scores evaluator for each class of BOI-like tags."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        # Create list of tags
        self.__init_tag_list(targets_tag_sequences)
        # Init values
        TP = self.__get_zeros_tag_dict()
        FP = self.__get_zeros_tag_dict()
        FN = self.__get_zeros_tag_dict()
        F05 = self.__get_zeros_tag_dict()
        for targets_seq, outputs_tag_seq in zip(targets_tag_sequences, outputs_tag_sequences):
            for t, o in zip(targets_seq, outputs_tag_seq):
                if t == o:
                    TP = self.__add_to_dict(TP, t, 1)
                else:
                    FN = self.__add_to_dict(FN, t, 1)
                    FP = self.__add_to_dict(FP, o, 1)
        # Calculate F05 for each tag
        for tag in self.tag_list:
            #F05[tag] = (2 * TP[tag] / max(2 * TP[tag] + FP[tag] + FN[tag], 1)) * 100
            F05[tag] = self.__get_f_beta(TP[tag], FN[tag], FP[tag], beta=0.5)
        # Calculate Macro-F05 score and prepare the message
        M_F05, msg = self.__get_M_F05_msg(F05)
        print(msg)
        #self.validate_M_F05_scikitlearn( targets_tag_sequences, outputs_tag_sequences)
        return M_F05, msg
