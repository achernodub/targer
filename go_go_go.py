from __future__ import print_function

import argparse
import codecs
import copy
import datetime
import functools
import json
import itertools
import os
import random
import sys
import time

import numpy as np
from os import system
from os.path import isfile, join, exists
from random import randint, shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
from sys import exit
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils_data import *
from sequences_indexer import SequencesIndexer

seed_num = 42
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)

print('Hello, train/dev/test script!')

emb_fn = 'embeddings/glove.6B.100d.txt'
gpu = -1 # current version is for CPU only!

caseless = True
shrink_to_train = False
unk = None
delimiter = ' '
epoch_num = 50

hidden_layer_dim = 200
hidden_layer_num = 1
dropout_ratio = 0.5
clip_grad = 5.0
opt_method = 'sgd'

lr = 0.015
momentum = 0.9
batch_size = 10

debug_mode = False
verbose = True


# Select data

if (1 == 1):
    # Essays
    fn_train = 'data/argument_mining/persuasive_essays/es_paragraph_level_train.txt'
    fn_dev = 'data/argument_mining/persuasive_essays/es_paragraph_level_dev.txt'
    fn_test = 'data/argument_mining/persuasive_essays/es_paragraph_level_test.txt'
else:
    # CoNNL-2003 NER shared task
    fn_train = 'data/NER/CoNNL_2003_shared_task/train.txt'
    fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.txt'
    fn_test = 'data/NER/CoNNL_2003_shared_task/test.txt'


# Load CoNNL data as sequences of strings
token_sequences_train, tag_sequences_train = read_CoNNL(fn_train)
token_sequences_dev, tag_sequences_dev = read_CoNNL(fn_dev)
token_sequences_test, tag_sequences_test = read_CoNNL(fn_test)

# Indexer is a class to convert tokens and tags as strings to integer indices and back
sequences_indexer = SequencesIndexer(caseless=caseless, verbose=verbose)
sequences_indexer.load_embeddings(emb_fn=emb_fn, delimiter=delimiter)
sequences_indexer.add_token_sequences(token_sequences_train)
sequences_indexer.add_token_sequences(token_sequences_dev)
sequences_indexer.add_token_sequences(token_sequences_test)
sequences_indexer.add_tag_sequences(tag_sequences_train) # Surely, all necessarily tags exists in train data

inputs_train = sequences_indexer.token2idx(token_sequences_train)
outputs_train = sequences_indexer.tag2idx(tag_sequences_train)

print(sequences_indexer.token2idx_dict['<UNK>'])
print('------------------')
print(len(sequences_indexer.embeddings_list))
print(len(sequences_indexer.get_tags_list()))
print(len(sequences_indexer.get_token_list()))
print('====================')
print(len(sequences_indexer.token2idx_dict))
print(len(sequences_indexer.idx2token_dict))
print('/////////////////////')
print(len(sequences_indexer.tag2idx_dict))
print(len(sequences_indexer.idx2tag_dict))

embeddings_tensor = sequences_indexer.get_embeddings_tensor()

token_sequences_train2 = sequences_indexer.idx2token(inputs_train)



print('The end.')


exit()

#indexer.add_tokens_sequences()











sequences_all = sequences_train + sequences_dev + sequences_test

_, _, feature_str_unique_list, label_str_map = generate_corpus(sequences_all, caseless)

feature_train_str_seq_list, label_train_str_seq_list, _, _ = generate_corpus(sequences_train, caseless)
feature_dev_str_seq_list, label_dev_str_seq_list, _, _ = generate_corpus(sequences_dev, caseless)
feature_test_str_seq_list, label_test_str_seq_list, _, _ = generate_corpus(sequences_test, caseless)

# Load embeddings and create feature maps for them
embeddings, feature_str_map = load_embeddings(emb_fn, delimiter, feature_str_unique_list, caseless, unk, shrink_to_train,
                                              show_not_found_tokens=False)

# Prepare reverse feature and label string maps, i.e. indices to strings
feature_str_map_reverse = {v: k for k, v in feature_str_map.items()}
label_str_map_reverse = {v: k for k, v in label_str_map.items()}

# Convert lists of text sequences to lists of indices sequences
feature_train_idx_seq_list = seq_list_str2idx(feature_train_str_seq_list, feature_str_map, unk)
feature_dev_idx_seq_list = seq_list_str2idx(feature_dev_str_seq_list, feature_str_map, unk)
feature_test_idx_seq_list = seq_list_str2idx(feature_test_str_seq_list, feature_str_map, unk)
label_train_idx_seq_list = seq_list_str2idx(label_train_str_seq_list, label_str_map)
label_dev_idx_seq_list = seq_list_str2idx(label_dev_str_seq_list, label_str_map)
label_test_idx_seq_list = seq_list_str2idx(label_test_str_seq_list, label_str_map)



print('The end!')