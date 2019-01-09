"""
.. module:: EvaluatorFactory
    :synopsis: EvaluatorFactory contains wrappers to create evaluators

.. moduleauthor:: Artem Chernodub
"""

from src.evaluators.evaluator_f1_connl import EvaluatorF1Connl


class EvaluatorFactory():
    @staticmethod
    def create(args):
        if args.evaluator == 'f1_connl':
            return EvaluatorF1Connl()
        else:
            raise ValueError('Unknown evaluator, must be one of "f1_connl"/"token_acc".')
