"""
Isaac Persing and Vincent Ng. End-to-end argumentation mining in student essays. NAACL 2016.
http://www.aclweb.org/anthology/N16-1164.
"""
from src.evaluators.evaluator_f1_alpha_match_base import EvaluatorF1AlphaMatchBase


class EvaluatorF1AlphaMatch05(EvaluatorF1AlphaMatchBase):
    def __init__(self):
        super(EvaluatorF1AlphaMatch05, self).__init__(match_alpha_ratio=0.5)
