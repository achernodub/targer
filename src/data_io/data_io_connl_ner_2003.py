"""input/output data wrapper for CoNNL file format used in  NER-2003 Shared Task dataset"""
import codecs
from src.classes.utils import get_words_num


class DataIOConnlNer2003():
    """DataIONerConnl2003 is an input/output data wrapper for CoNNL-2003 Shared Task file format.
    Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent
    named entity recognition." Proceedings of the seventh conference on Natural language learning at HLT-NAACL
    2003-Volume 4. Association for Computational Linguistics, 2003.
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
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        curr_words = list()
        curr_tags = list()
        for k in range(len(lines)):
            line = lines[k].strip()
            if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
                if len(curr_words) > 0:
                    word_sequences.append(curr_words)
                    tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue
            strings = line.split(' ')
            word = strings[0]
            tag = strings[column_no] # be default, we take the last tag
            curr_words.append(word)
            curr_tags.append(tag)
            if k == len(lines) - 1:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
        if verbose:
            print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
        return word_sequences, tag_sequences

    def write_data(self, fn, word_sequences, tag_sequences_1, tag_sequences_2):
        text_file = open(fn, mode='w')
        for i, words in enumerate(word_sequences):
            tags_1 = tag_sequences_1[i]
            tags_2 = tag_sequences_2[i]
            for j, word in enumerate(words):
                tag_1 = tags_1[j]
                tag_2 = tags_2[j]
                text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
            text_file.write('\n')
        text_file.close()
