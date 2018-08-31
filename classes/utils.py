import codecs
import datetime
import itertools

def info(t, name=''):
    print(name, '|', t.type(), '|', t.shape)

def flatten(list_in):
    return [list(itertools.chain.from_iterable(list_item)) for list_item in list_in]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def write_textfile(fn, s):
    if fn is not None:
        with open(fn, mode='w') as text_file:
            text_file.write(s)

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

def extract_settings(args):
    return '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')]) + '\n'