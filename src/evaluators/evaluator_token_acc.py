import os
import random
import time
from src.classes.data_io import DataIO
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorTokenAcc(EvaluatorBase):
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences):
        pass
