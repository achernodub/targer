"""f1-micro averaging evaluation for tag components, spans detection + classification, alpha = 0.5"""
from src.evaluators.evaluator_f1_micro_spans_alpha_match_base import EvaluatorF1MicroSpansAlphaMatchBase


class EvaluatorF1MicroSpansAlphaMatch05(EvaluatorF1MicroSpansAlphaMatchBase):
    """
    EvaluatorF1MicroSpansAlphaMatch05 is f1-micro averaging evaluator for tag components, alpha = 0.5 (fuzzy)
    Isaac Persing and Vincent Ng. End-to-end argumentation mining in student essays. NAACL 2016.
    http://www.aclweb.org/anthology/N16-1164.
    """
    def __init__(self):
        super(EvaluatorF1MicroSpansAlphaMatch05, self).__init__(match_alpha_ratio=0.5)
