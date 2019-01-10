"""
.. module:: EvaluatorFactory
    :synopsis: EvaluatorFactory contains wrappers to create evaluators

.. moduleauthor:: Artem Chernodub
"""
from src.evaluators.evaluator_f1_connl import EvaluatorF1Connl
from src.evaluators.evaluator_f1_alpha_match_10 import EvaluatorF1AlphaMatch10
from src.evaluators.evaluator_f1_alpha_match_05 import EvaluatorF1AlphaMatch05
from src.evaluators.evaluator_token_acc import EvaluatorTokenAcc

class EvaluatorFactory():
    @staticmethod
    def create(args):
        if args.evaluator == 'f1_connl':
            return EvaluatorF1Connl()
        elif args.evaluator == 'f1_alpha_match_10':
            return EvaluatorF1AlphaMatch10()
        elif args.evaluator == 'f1_alpha_match_05':
            return EvaluatorF1AlphaMatch05()
        elif args.evaluator == 'token_acc':
            return EvaluatorTokenAcc()
        else:
            raise ValueError('Unknown evaluator, must be one of "f1_connl"/"token_acc".')
