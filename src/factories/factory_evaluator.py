"""creates various evaluators"""
from src.evaluators.evaluator_f1_connl import EvaluatorF1Connl
from src.evaluators.evaluator_f1_alpha_match_10 import EvaluatorF1AlphaMatch10
from src.evaluators.evaluator_f1_alpha_match_05 import EvaluatorF1AlphaMatch05
from src.evaluators.evaluator_f1_macro_tag_components import EvaluatorF1MacroTagComponents
from src.evaluators.evaluator_token_acc import EvaluatorTokenAcc


class EvaluatorFactory():
    """EvaluatorFactory contains wrappers to create various evaluators."""
    @staticmethod
    def create(args):
        if args.evaluator == 'f1-connl':
            return EvaluatorF1Connl()
        elif args.evaluator == 'f1-alpha-match-10':
            return EvaluatorF1AlphaMatch10()
        elif args.evaluator == 'f1-alpha-match-05':
            return EvaluatorF1AlphaMatch05()
        elif args.evaluator == 'f1-macro':
            return EvaluatorF1MacroTagComponents()
        elif args.evaluator == 'token-acc':
            return EvaluatorTokenAcc()
        else:
            raise ValueError('Unknown evaluator %s.' % args.evaluator)
