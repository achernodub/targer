import codecs
from classes.utils import is_number

class DataIO():
    @staticmethod
    def read_CoNNL_dat_abs(fn, column_no=-1):
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
        return word_sequences, tag_sequences

    @staticmethod
    def write_CoNNL_dat_abs(fn, word_sequences, tag_sequences):
        with open(fn, mode='w') as text_file:
            for words, tags in zip(word_sequences, tag_sequences):
                for i, (word, tag) in enumerate(zip(words, tags)):
                    text_file.write('%d\t%s\t%s\n' % (i+1, word, tag))
                text_file.write('\n')

    @staticmethod
    def read_CoNNL_2003(fn, column_no=-1):
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
        return word_sequences, tag_sequences

    @staticmethod
    def write_CoNNL_2003(fn, word_sequences, tag_sequences):
        text_file = open(fn, mode='w')
        for i, words in enumerate(word_sequences):
            text_file.write('-DOCSTART- -X- -X-\n')
            tags_1 = tag_sequences[i]
            for j, word in enumerate(words):
                tag_1 = tags_1[j]
                text_file.write('%s %s\n' % (word, tag_1))
        text_file.close()

    @staticmethod
    def __is_CoNNL_dat_abs(fn):
        with codecs.open(fn, 'r', 'utf-8') as f:
            c = f.readlines()[0][0]
            return is_number(c)

    @staticmethod
    def read_CoNNL_universal(fn,column_no=-1):
        if DataIO.__is_CoNNL_dat_abs(fn):
            return DataIO.read_CoNNL_dat_abs(fn, column_no)
        else:
            return DataIO.read_CoNNL_2003(fn, column_no)
