"""several auxiliary functions"""

import argparse
import datetime
import itertools
import sys
import torch


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
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_sequences_by_indices(sequences, indices):
    return [sequences[i] for i in indices]


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def argsort_sequences_by_lens(list_in):
    data_num = len(list_in)
    sort_indices = argsort([-len(item) for item in list_in])
    reverse_sort_indices = [-1 for _ in range(data_num)]
    for i in range(data_num):
        reverse_sort_indices[sort_indices[i]] = i
    return sort_indices, reverse_sort_indices


def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_input_arguments():
    return 'python3 main.py ' + ' '.join([arg for arg in sys.argv[1:]])
