import codecs
import os
import os.path
import random
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score # f1_score, precision_score, recall_score
from classes.data_io import DataIO
from classes.tag_component import TagComponent


class Evaluator():
    @staticmethod
    def __get_accuracy_from_sequences_token_level(targets_tag_sequences, outputs_tag_sequences, tag_seq_indexer):
        targets_idx = tag_seq_indexer.items2idx(targets_tag_sequences)
        outputs_idx = tag_seq_indexer.items2idx(outputs_tag_sequences)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        return accuracy_score(y_true, y_pred) * 100

    @staticmethod
    def get_acuracy_token_level(tagger, word_sequences, targets_tag_sequences):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences=word_sequences, batch_size=100)
        acc = Evaluator.__get_accuracy_from_sequences_token_level(targets_tag_sequences=targets_tag_sequences,
                                                                  outputs_tag_sequences=outputs_tag_sequences,
                                                                  tag_seq_indexer=tagger.tag_seq_indexer)
        return acc

    @staticmethod
    def get_accuracy_train_dev_test(tagger, datasets_bank):
        acc_train = Evaluator.get_acuracy_token_level(tagger=tagger,
                                                      word_sequences=datasets_bank.word_sequences_train,
                                                      targets_tag_sequences=datasets_bank.tag_sequences_train)
        acc_dev = Evaluator.get_acuracy_token_level(tagger=tagger,
                                                    word_sequences=datasets_bank.word_sequences_dev,
                                                    targets_tag_sequences=datasets_bank.tag_sequences_dev)
        acc_test = Evaluator.get_acuracy_token_level(tagger=tagger,
                                                     word_sequences=datasets_bank.word_sequences_test,
                                                     targets_tag_sequences=datasets_bank.tag_sequences_test)
        return acc_train, acc_dev, acc_test

    @staticmethod
    def get_f1_connl_script(tagger, word_sequences, targets_tag_sequences, outputs_tag_sequences=None, fn_out=None):
        if fn_out == None:
            fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        if outputs_tag_sequences is None:
            outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences)
        DataIO.write_CoNNL_2003_two_columns(fn_out, word_sequences, targets_tag_sequences, outputs_tag_sequences)
        cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        connl_str = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        connl_str += ''.join(os.popen(cmd).readlines())
        time.sleep(0.5)
        if fn_out.startswith('out_temp_') and os.path.exists(fn_out):
            os.remove(fn_out)
        f1 = float(connl_str.split('\n')[3].split(':')[-1].strip())
        return f1, connl_str

    @staticmethod
    def get_f1_connl_train_dev_test(tagger, datasets_bank):
        f1_train, _ = Evaluator.get_f1_connl_script(tagger=tagger,
                                                    word_sequences=datasets_bank.word_sequences_train,
                                                    targets_tag_sequences=datasets_bank.tag_sequences_train)
        f1_dev, _ = Evaluator.get_f1_connl_script(tagger=tagger,
                                                  word_sequences=datasets_bank.word_sequences_dev,
                                                  targets_tag_sequences=datasets_bank.tag_sequences_dev)
        f1_test, _ = Evaluator.get_f1_connl_script(tagger=tagger,
                                                   word_sequences=datasets_bank.word_sequences_test,
                                                   targets_tag_sequences=datasets_bank.tag_sequences_test)
        return f1_train, f1_dev, f1_test

    @staticmethod
    def get_f1_components_from_words(targets_tag_sequences, outputs_tag_sequences, match_alpha_ratio=0.999):
        targets_tag_components_sequences = TagComponent.extract_tag_components_sequences(targets_tag_sequences)
        outputs_tag_components_sequences = TagComponent.extract_tag_components_sequences(outputs_tag_sequences)
        return Evaluator.__get_f1_components_from_sequences(targets_tag_components_sequences,
                                                            outputs_tag_components_sequences,
                                                            match_alpha_ratio)

    @staticmethod
    def __get_f1_components_from_sequences(targets_tag_components_sequences, outputs_tag_components_sequences, match_alpha_ratio):
        TP, FN, FP = 0, 0, 0
        for targets_tag_components, outputs_tag_components in zip(targets_tag_components_sequences, outputs_tag_components_sequences):
            for target_tc in targets_tag_components:
                found = False
                for output_tc in outputs_tag_components:
                    if output_tc.is_equal(target_tc, match_alpha_ratio):
                        found = True
                        break
                if found:
                    TP += 1
                else:
                    FN += 1
            for output_tc in outputs_tag_components:
                found = False
                for target_tc in targets_tag_components:
                    if target_tc.is_equal(output_tc, match_alpha_ratio):
                        found = True
                        break
                if not found:
                    FP += 1
        Precision = (TP / max(TP + FP, 1))*100
        Recall = (TP / max(TP + FN, 1))*100
        F1 = (2 * TP / max(2 * TP + FP + FN, 1))*100
        return F1, Precision, Recall, (TP, FP, FN)
