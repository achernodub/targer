"""generate predefined FastText embeddings for dataset"""
from __future__ import print_function
import argparse
import codecs
from numpy import mean, std
from os import listdir
from os.path import dirname, join, realpath


def get_score_from_report(fn):
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    return float(lines[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--dir', help='Directory with reports')
    args = parser.parse_args()
    dir_path = dirname(realpath(__file__))
    path = join(dir_path, args.dir)
    scores = list()
    for fn in listdir(path):
        if not fn.endswith('.txt'):
            continue
        scores.append(get_score_from_report(join(path, fn)))
    print('\ndir = %s' % args.dir)
    print('scores =', scores)
    print('mean = %1.3f, std = %1.3f' % (mean(scores), std(scores)))
