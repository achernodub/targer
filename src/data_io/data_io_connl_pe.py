"""input/output data wrapper for CoNNL file format used in Persuassive Essays dataset"""
import codecs
from src.classes.utils import get_words_num


class DataIOConnlPe():
    """DataIOConnlPe is an input/output data wrapper for CoNNL file format (".abs.dat" files).
    Eger, Steffen, Johannes Daxenberger, and Iryna Gurevych. "Neural end-to-end learning for computational argumentation
    mining." arXiv preprint arXiv:1704.06104 (2017).
    """
    def read_train_dev_test(self, args):
        word_sequences_train, tag_sequences_train = self.read_data(fn=args.train, verbose=args.verbose)
        word_sequences_dev, tag_sequences_dev = self.read_data(fn=args.dev, verbose=args.verbose)
        word_sequences_test, tag_sequences_test = self.read_data(fn=args.test, verbose=args.verbose)
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, \
               tag_sequences_test

    def read_data(self, fn, verbose=True, column_no=-1):
        word_sequences = list()
        tag_sequences = list()
        curr_words = list()
        curr_tags = list()
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        for k, line in enumerate(lines):
            elements = line.strip().split('\t')
            if len(elements) < 3: # end of the document
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
                continue
            word = elements[1]
            tag = elements[2].split(':')[0]
            curr_words.append(word)
            curr_tags.append(tag)
        if verbose:
            print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
        return word_sequences, tag_sequences

    def write(self, fn, word_sequences, tag_sequences):
        with open(fn, mode='w') as text_file:
            for words, tags in zip(word_sequences, tag_sequences):
                for i, (word, tag) in enumerate(zip(words, tags)):
                    text_file.write('%d\t%s\t%s\n' % (i+1, word, tag))
                text_file.write('\n')
