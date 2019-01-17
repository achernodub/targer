"""input/output data wrapper for CoNNL file format used in Web Discourse dataset"""
import codecs
from glob import glob
from os.path import join
from src.classes.utils import get_words_num


class DataIOConnlWd():
    """DataIOConnlAbsPe is an input/output data wrapper for CoNNL format where each document is in separate text file.
    Habernal, Ivan, and Iryna Gurevych. "Argumentation Mining In User-generated Web Discourse."
    Computational Linguistics 43.1 (2017): 125-179.
    """
    def read_train_dev_test(self, args):
        word_sequences, tag_sequences = self.read_dir(dir=args.train, verbose=args.verbose)
        i1 = 272
        i2 = 306
        word_sequences_train = word_sequences[:i1]
        tag_sequences_train = tag_sequences[:i1]
        word_sequences_dev = word_sequences[i1:i2]
        tag_sequences_dev = tag_sequences[i1:i2]
        word_sequences_test = word_sequences[i2:]
        tag_sequences_test = tag_sequences[i2:]
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test
        #if verbose:
        #    print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))

    def read_dir(self, dir, verbose=True):
        file_list = glob(join(dir, '*.txt'))
        word_sequences, tag_sequences = list(), list()
        for fn in file_list:
            word_seq, tag_seq = self.read_data(fn, verbose)
            assert len(word_seq) == len(tag_seq)
            word_sequences.append(word_seq)
            tag_sequences.append(tag_seq)
        return word_sequences, tag_sequences

    def read_data(self, fn, verbose):
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        word_seq, tag_seq = list(), list()
        for k, line in enumerate(lines):
            elements = line.strip().split('\t')
            if len(elements) == 2:
                word_seq.append(elements[0])
                tag_seq.append(elements[1])
        return word_seq, tag_seq

#    def get_train_batches(self, batch_size):
#        random_indices = np.random.permutation(np.arange(self.train_data_num))
#        for k in range(self.train_data_num // batch_size): # oh yes, we drop the last batch
#            batch_indices = random_indices[k:k + batch_size].tolist()
#            word_sequences_train_batch, tag_sequences_train_batch = self.__get_train_batch(batch_indices)
#            yield word_sequences_train_batch, tag_sequences_train_batch
