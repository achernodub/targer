import codecs
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
