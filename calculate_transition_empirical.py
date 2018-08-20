from __future__ import print_function

import torch
from classes.data_io import DataIO
from classes.element_seq_indexer import ElementSeqIndexer


print('Hello!')

def pretty_print_transition_matrix(transition_matrix, tags):
    template_str = '%15s' % ' '
    for j in range(len(tags)):
        template_str += '%15s ' % tags[j]
    print(template_str)
    for i in range(len(tags)+1):
        if i < len(tags):
            template_str = '%15s ' % tags[i]
        else:
            template_str = '%15s ' % '<start>'
        for j in range(len(tags)):
            if transition_matrix[i, j] != 0:
                s = '%1.4f' % transition_matrix[i, j]
            else:
                s = '%d' % 0
            template_str += '%15s ' % s
        print(template_str)

#fn_train = 'data/persuasive_essays/Essay_Level/train.dat.abs'
fn_train = 'data/persuasive_essays/Paragraph_Level/train.dat.abs'
gpu = -1

word_sequences_train, tag_sequences_train = DataIO.read_CoNNL_universal(fn_train)

# Converts lists of lists of tags to integer indices and back
tag_seq_indexer = ElementSeqIndexer(gpu=gpu, caseless=False, pad='<pad>', unk=None)
tag_seq_indexer.load_vocabulary_from_element_sequences(tag_sequences_train)
tags = tag_seq_indexer.elements_list

transition_matrix = torch.zeros(len(tags)+1, len(tags), dtype=torch.float)
cnt = 0
for tag_sequence in tag_sequences_train:
    start_tag = tag_sequence[0]
    start_tag_idx = tag_seq_indexer.element2idx_dict[start_tag]
    transition_matrix[len(tags), start_tag_idx] += 1
    cnt += 1
    for k in range(1, len(tag_sequence)):
        prev_tag = tag_sequence[k - 1]
        curr_tag = tag_sequence[k]
        prev_tag_idx = tag_seq_indexer.element2idx_dict[prev_tag]
        curr_tag_idx = tag_seq_indexer.element2idx_dict[curr_tag]
        transition_matrix[prev_tag_idx, curr_tag_idx] += 1
        cnt += 1
for i in range(len(tags)+1):
    for j in range(len(tags)):
        transition_matrix[i, j] = transition_matrix[i, j] / cnt

pretty_print_transition_matrix(transition_matrix, tags)

print('\nThe end.')

