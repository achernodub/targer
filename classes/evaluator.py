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
    def get_accuracy_token_level(targets_tag_sequences, outputs_tag_sequences, tag_seq_indexer):
        targets_idx = tag_seq_indexer.elements2idx(targets_tag_sequences)
        outputs_idx = tag_seq_indexer.elements2idx(outputs_tag_sequences)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        return accuracy_score(y_true, y_pred) * 100

    @staticmethod
    def get_f1_from_words(targets_tag_sequences, outputs_tag_sequences, match_alpha_ratio=0.999):
        targets_tag_components_sequences = TagComponent.extract_tag_components_sequences(targets_tag_sequences)
        outputs_tag_components_sequences = TagComponent.extract_tag_components_sequences(outputs_tag_sequences)
        return Evaluator.get_f1_from_components_sequences(targets_tag_components_sequences,
                                                          outputs_tag_components_sequences,
                                                          match_alpha_ratio)
    @staticmethod
    def get_f1_from_words_connl_script(word_sequences, targets_tag_sequences, outputs_tag_sequences):
        fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        DataIO.write_CoNNL_2003_two_columns(fn_out, word_sequences, targets_tag_sequences, outputs_tag_sequences)
        cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        str_out = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        str_out += ''.join(os.popen(cmd).readlines())
        time.sleep(1)
        if os.path.exists(fn_out):
            os.remove(fn_out)
        return str_out

    @staticmethod
    def get_f1_from_components_sequences(targets_tag_components_sequences, outputs_tag_components_sequences, match_alpha_ratio):
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

    @staticmethod
    def write_report_table(fn_out, tagger, epoch, word_sequences_train, tag_sequences_train, word_sequences_dev,
                           tag_sequences_dev, word_sequences_test, tag_sequences_test):
        outputs_tag_sequences_train = tagger.predict_tags_from_words(word_sequences_train, batch_size=100)
        outputs_tag_sequences_dev = tagger.predict_tags_from_words(word_sequences_dev, batch_size=100)
        outputs_tag_sequences_test = tagger.predict_tags_from_words(word_sequences_test, batch_size=100)

        acc_train = Evaluator.get_accuracy_token_level(tag_sequences_train, outputs_tag_sequences_train, tagger.tag_seq_indexer)
        acc_dev = Evaluator.get_accuracy_token_level(tag_sequences_dev, outputs_tag_sequences_dev, tagger.tag_seq_indexer)
        acc_test = Evaluator.get_accuracy_token_level(tag_sequences_test, outputs_tag_sequences_test, tagger.tag_seq_indexer)

        connl_report_train_str = Evaluator.get_f1_from_words_connl_script(word_sequences_train, tag_sequences_train,
                                                                          outputs_tag_sequences_train)
        connl_report_dev_str = Evaluator.get_f1_from_words_connl_script(word_sequences_dev, tag_sequences_dev,
                                                                        outputs_tag_sequences_dev)
        connl_report_test_str = Evaluator.get_f1_from_words_connl_script(word_sequences_test, tag_sequences_test,
                                                                         outputs_tag_sequences_test)
        f = open(fn_out, 'a')
        f.write('-' * 80 + '\n')
        f.write('%10s | %10s | %10s | %10s\n' % ('epoch', 'TRAIN acc.', 'DEV acc.', 'TEST acc.'))
        f.write('-' * 80 + '\n')
        f.write('%10s | %10s | %10s | %10s\n\n' % ('%d' % epoch, '%1.2f' % acc_train, '%1.2f' % acc_dev,
                                                   '%1.2f' % acc_test))
        f.write('TRAIN: %s\n' % connl_report_train_str)
        f.write('DEV: %s\n' % connl_report_dev_str)
        f.write('TEST: %s\n' % connl_report_test_str)
        f.close()
