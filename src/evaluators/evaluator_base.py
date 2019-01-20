"""abstract base class for all evaluators"""


class EvaluatorBase():
    """EvaluatorBase is abstract base class for all evaluators"""
    def get_evaluation_score_train_dev_test(self, tagger, datasets_bank, batch_size=-1):
        if batch_size == -1:
            batch_size = tagger.batch_size
        score_train, _ = self.predict_evaluation_score(tagger=tagger,
                                                       word_sequences=datasets_bank.word_sequences_train,
                                                       targets_tag_sequences=datasets_bank.tag_sequences_train,
                                                       batch_size=batch_size)
        score_dev, _ = self.predict_evaluation_score(tagger=tagger,
                                                     word_sequences=datasets_bank.word_sequences_dev,
                                                     targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                     batch_size=batch_size)
        score_test, msg_test = self.predict_evaluation_score(tagger=tagger,
                                                             word_sequences=datasets_bank.word_sequences_test,
                                                             targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                             batch_size=batch_size)
        return score_train, score_dev, score_test, msg_test

    def predict_evaluation_score(self, tagger, word_sequences, targets_tag_sequences, batch_size):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size)
        return self.get_evaluation_score(targets_tag_sequences, outputs_tag_sequences, word_sequences)
