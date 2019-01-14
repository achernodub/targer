"""macro-F1 scores evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorF1MacroTagComponents(EvaluatorBase):
    """EvaluatorF1MacroTagComponents is macro-F1 scores evaluator for each class of BOI-like tags."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        # Create list of tags
        tag_list = list()
        for targets_tag_seq in targets_tag_sequences:
            for t in targets_tag_seq:
                if t not in tag_list:
                    tag_list.append(t)
        # Init values
        TP = {tag: 0 for tag in tag_list}
        FP = {tag: 0 for tag in tag_list}
        FN = {tag: 0 for tag in tag_list}
        F1 = {tag: 0 for tag in tag_list}
        # Calculate TP/FN/FP
        for targets_seq, targets_tag_seq in zip(targets_tag_sequences, outputs_tag_sequences):
            for t, o in zip(targets_seq, targets_tag_seq):
                if t == o:
                    TP[t] += 1
                else:
                    FN[t] += 1
                    FP[o] += 1
        # Calculate F1 for each tag
        for tag in tag_list:
            F1[tag] = (2 * TP[tag] / max(2 * TP[tag] + FP[tag] + FN[tag], 1)) * 100
        # Calculate Macro-F1 score and prepare the message
        msg = '\nF1 scores\n'
        msg += '-' * 24 + '\n'
        sum_M_F1 = 0
        for tag in F1:
            sum_M_F1 += F1[tag]
            msg += '%15s = %1.2f\n' % (tag, F1[tag])
        M_F1 = sum_M_F1 / len(tag_list)
        msg += '-'*24 + '\n'
        msg += 'Macro-F1 = %1.2f' % M_F1
        return M_F1, msg
