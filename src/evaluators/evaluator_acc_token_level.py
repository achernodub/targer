"""token-level accuracy evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorAccuracyTokenLevel(EvaluatorBase):
    """EvaluatorAccuracyTokenLevel is token-level accuracy evaluator for each class of BOI-like tags."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        cnt = 0
        match = 0
        for target_seq, output_seq in zip(targets_tag_sequences, outputs_tag_sequences):
            for t, o in zip(target_seq, output_seq):
                cnt += 1
                if t == o:
                    match += 1
        acc = match*100.0/cnt
        msg = '*** Token-level accuracy: %1.2f%% ***' % acc
        return acc, msg
