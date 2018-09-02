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

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute)