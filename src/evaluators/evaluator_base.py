"""
.. module:: EvaluatorBase
    :synopsis: EvaluatorBase

.. moduleauthor:: Artem Chernodub
"""

import os
import os.path
import random
import time
from sklearn.metrics import accuracy_score # f1_score, precision_score, recall_score
from src.classes.data_io import DataIO
from src.evaluators.tag_component import TagComponent


class EvaluatorBase():
    @staticmethod
    def get_evaluation_score_train_dev_test(tagger, datasets_bank, batch_size=-1):
        if batch_size == -1:
            batch_size = tagger.batch_size
        score_train = EvaluatorBase.predict_evaluation_score(tagger=tagger,
                                                             word_sequences=datasets_bank.word_sequences_train,
                                                             targets_tag_sequences=datasets_bank.tag_sequences_train,
                                                             batch_size=batch_size)
        score_dev = EvaluatorBase.predict_evaluation_score(tagger=tagger,
                                                           word_sequences=datasets_bank.word_sequences_dev,
                                                           targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                           batch_size=batch_size)
        score_test = EvaluatorBase.predict_evaluation_score(tagger=tagger,
                                                            word_sequences=datasets_bank.word_sequences_test,
                                                            targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                            batch_size=batch_size)
        return score_train, score_dev, score_test

    @staticmethod
    def predict_evaluation_score(tagger, word_sequences, targets_tag_sequences, batch_size):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size)
        return EvaluatorBase.get_evaluation_score(targets_tag_sequences, outputs_tag_sequences, word_sequences)

    #@staticmethod
    #def get_evaluation_score(word_sequences, targets_tag_sequences, outputs_tag_sequences):
    #    pass
