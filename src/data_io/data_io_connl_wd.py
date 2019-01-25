"""input/output data wrapper for CoNNL file format used in Web Discourse dataset"""
import codecs
import json
from glob import glob
from os.path import join


class DataIOConnlWd():
    """DataIOConnlAbsPe is an input/output data wrapper for CoNNL format where each document is in separate text file.
    Habernal, Ivan, and Iryna Gurevych. "Argumentation Mining In User-generated Web Discourse."
    Computational Linguistics 43.1 (2017): 125-179.
    """
    def read_train_dev_test(self, args):
        word_sequences, tag_sequences = self.read_data(dir=args.train, verbose=args.verbose)
        cross_folds = self.get_cross_folds(word_sequences, tag_sequences, args.cross_folds_num)
        sequences = self.split_cross_folds(cross_folds, args.cross_folds_num, args.cross_fold_id)
        #with open('wd_test_cv_%d.txt' % args.cross_fold_id, 'w') as f:
        #    json.dump([sequences[4], sequences[5]], f)
        if args.verbose:
            print('*** Loading WD data from dir = %s' % args.train)
            print('*** train : dev : test = %d : %d : %d, cross-fold-id = %d' % (len(sequences[0]), len(sequences[2]),
                                                                             len(sequences[4]), args.cross_fold_id))
        return sequences[0], sequences[1], sequences[2], sequences[3], sequences[4], sequences[5]

    def get_cross_folds(self, word_sequences, tag_sequences, cross_folds_num):
        assert len(word_sequences) == len(tag_sequences)
        fold_len = len(word_sequences) // cross_folds_num
        folds = list()
        for k in range(cross_folds_num):
            i = k*fold_len
            j = (k + 1)*fold_len
            if k == cross_folds_num - 1:
                j = len(word_sequences)
            folds.append((word_sequences[i:j], tag_sequences[i:j]))
        return folds

    def split_cross_folds_v1(self, cross_folds, cross_folds_num, cross_fold_id):
        dev_cross_fold_id = cross_fold_id - 1
        test_cross_fold_id = cross_fold_id
        if cross_fold_id == cross_folds_num:
            test_cross_fold_id = 0
        word_sequences_train = list()
        tag_sequences_train = list()
        word_sequences_dev = list()
        tag_sequences_dev = list()
        word_sequences_test = list()
        tag_sequences_test = list()
        for n in range(cross_folds_num):
            if n == dev_cross_fold_id:
                word_sequences_dev.extend(cross_folds[n][0])
                tag_sequences_dev.extend(cross_folds[n][1])
            elif n == test_cross_fold_id:
                word_sequences_test.extend(cross_folds[n][0])
                tag_sequences_test.extend(cross_folds[n][1])
            else:
                word_sequences_train.extend(cross_folds[n][0])
                tag_sequences_train.extend(cross_folds[n][1])
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, \
               tag_sequences_test

    def __get_train_dev_test_ids(self, cross_folds_num, cross_fold_id):
        ids = list(range(cross_folds_num))
        for i in range(cross_fold_id):
            ids.insert(0, ids.pop())
        test_ids = [ids[0]]
        dev_ids = [ids[1], ids[2]]
        train_ids = [ids[i] for i in range(3, cross_folds_num)]
        return train_ids, dev_ids, test_ids

    def split_cross_folds(self, cross_folds, cross_folds_num, cross_fold_id):
        word_sequences_train = list()
        tag_sequences_train = list()
        word_sequences_dev = list()
        tag_sequences_dev = list()
        word_sequences_test = list()
        tag_sequences_test = list()
        _, dev_ids, test_ids = self.__get_train_dev_test_ids(cross_folds_num, cross_fold_id)
        for n in range(cross_folds_num):
            if n in dev_ids:
                word_sequences_dev.extend(cross_folds[n][0])
                tag_sequences_dev.extend(cross_folds[n][1])
            elif n in test_ids:
                word_sequences_test.extend(cross_folds[n][0])
                tag_sequences_test.extend(cross_folds[n][1])
            else:
                word_sequences_train.extend(cross_folds[n][0])
                tag_sequences_train.extend(cross_folds[n][1])
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, \
               tag_sequences_test

    def read_data(self, dir, verbose=True):
        file_list = glob(join(dir, '*.txt'))
        word_sequences, tag_sequences = list(), list()
        for fn in file_list:
            word_seq, tag_seq = self.read_single_file(fn, verbose)
            assert len(word_seq) == len(tag_seq)
            word_sequences.append(word_seq)
            tag_sequences.append(tag_seq)
        return word_sequences, tag_sequences

    def read_single_file(self, fn, verbose):
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        word_seq, tag_seq = list(), list()
        for k, line in enumerate(lines):
            elements = line.strip().split('\t')
            if len(elements) == 2:
                word_seq.append(elements[0])
                tag_seq.append(elements[1])
        return word_seq, tag_seq
