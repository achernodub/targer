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


def load_embeddings(emb_fn, delimiter, feature_str_unique_list, caseless, unk=None, shrink_to_train=False, show_not_found_tokens=False):
    feature_str_map_out = dict()
    embeddings_list = list()
    # 1) Add embeddings from the embeddings file
    for line in open(emb_fn, 'r'):
        values = line.split(delimiter)
        fs = values[0].lower() if caseless else values[0]
        emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
        if shrink_to_train and fs not in feature_str_unique_list:
            continue
        feature_str_map_out[fs] = len(feature_str_map_out)
        embeddings_list.append(emb_vector)
    emb_len = len(emb_vector)
    # 2) Generate random embeddings for words which were not found in the embeddings file
    num_not_found = 0
    for fs in feature_str_unique_list:
        if fs not in feature_str_map_out:
            feature_str_map_out[fs] = len(feature_str_map_out)
            embeddings_list.append(np.random.uniform(-np.sqrt(3.0 / emb_len), np.sqrt(3.0 / emb_len), emb_len).tolist())
            num_not_found += 1
            if show_not_found_tokens:
                print('Embedding token not found # %d, %s' % (num_not_found, fs))
    # 3) Generate embedding for 'unknown' token
    if unk is not None:
        feature_str_map_out[unk] = len(feature_str_map_out)
        embeddings_list.append(np.random.uniform(-np.sqrt(3.0 / emb_len), np.sqrt(3.0 / emb_len), emb_len).tolist())
    print('File \"%s\" with embeddings was found.\n%d vectors were loaded.\n%d/%d vectors were not found and were replaced with random uniform values.' % (emb_fn, len(embeddings_list), num_not_found, len(feature_str_unique_list)))
    embedding_tensor = torch.FloatTensor(np.asarray(embeddings_list))
    return embedding_tensor, feature_str_map_out


def seq_list_str2idx(seq_str_list, str_map, unk=None):
    if unk is not None:
        seq_idx_list = list(map(lambda t: list(map(lambda m: str_map.get_batch_seq_lists(m, unk), t)), seq_str_list))
    else:
        seq_idx_list = list(map(lambda t: list(map(lambda m: str_map[m], t)), seq_str_list))
    return seq_idx_list