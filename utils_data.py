import argparse
import os
from sys import exit
from random import shuffle
from os import system
from os.path import isfile, join, exists
import codecs

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


def read_CoNNL(fn, column_no=-1):
    token_sequences = list()
    tag_sequences = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    curr_tokens = list()
    curr_tags = list()
    for k in range(len(lines)):
        line = lines[k].strip()
        if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
            if len(curr_tokens) > 0:
                token_sequences.append(curr_tokens)
                tag_sequences.append(curr_tags)
                curr_tokens = list()
                curr_tags = list()
            continue
        strings = line.split(' ')
        token = strings[0]
        tag = strings[column_no] # be default, we take the last tag
        curr_tokens.append(token)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            token_sequences.append(curr_tokens)
            tag_sequences.append(curr_tags)
    return token_sequences, tag_sequences

def info(name, t):
    print(name, '|', t.type(), '|', t.shape)