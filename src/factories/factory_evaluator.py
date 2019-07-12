"""creates various evaluators"""
from src.evaluators.evaluator_f1_micro_spans_connl import EvaluatorF1MicroSpansConnl
from src.evaluators.evaluator_f1_micro_spans_alpha_match_10 import EvaluatorF1MicroSpansAlphaMatch10
from src.evaluators.evaluator_f1_micro_spans_alpha_match_05 import EvaluatorF1MicroSpansAlphaMatch05
from src.evaluators.evaluator_f1_macro_token_level import EvaluatorF1MacroTokenLevel
from src.evaluators.evaluator_f05_macro_token_level import EvaluatorF05MacroTokenLevel
from src.evaluators.evaluator_acc_token_level import EvaluatorAccuracyTokenLevel


class EvaluatorFactory():
    """EvaluatorFactory contains wrappers to create various evaluators."""
    @staticmethod
    def create(args):
        if args.evaluator == 'f1-connl':
            return EvaluatorF1MicroSpansConnl()
        elif args.evaluator == 'f1-alpha-match-10':
            return EvaluatorF1MicroSpansAlphaMatch10()
        elif args.evaluator == 'f1-alpha-match-05':
            return EvaluatorF1MicroSpansAlphaMatch05()
        elif args.evaluator == 'f1-macro':
            return EvaluatorF1MacroTokenLevel()
        elif args.evaluator == 'f05-macro':
            return EvaluatorF05MacroTokenLevel()
        elif args.evaluator == 'token-acc':
            return EvaluatorAccuracyTokenLevel()
        else:
            raise ValueError('Unknown evaluator %s.' % args.evaluator)
