import codecs
import itertools

def info(name, t):
    print(name, '|', t.type(), '|', t.shape)

def flatten(list_in):
    return [list(itertools.chain.from_iterable(list_item)) for list_item in list_in]