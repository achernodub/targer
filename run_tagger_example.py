from __future__ import print_function

import os.path
import torch
from classes.utils import read_CoNNL_dat_abs, write_CoNNL_dat_abs
from classes.evaluator import Evaluator

print('Start!')

# Read data in CoNNL-2003 dat.abs format (Eger, 2017)
fn = 'data/persuasive_essays/Paragraph_Level/test.dat.abs'
token_sequences, tag_sequences = read_CoNNL_dat_abs(fn)

# Load tagger model
fn_checkpoint = 'es_tagger_par_200ep_LSTM.txt'
if os.path.isfile(fn_checkpoint):
    tagger = torch.load(fn_checkpoint)
else:
    raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty "--save_best_path" param to create it.' % fn_checkpoint)

# We take sequences_indexer from the tagger
sequences_indexer = tagger.sequences_indexer

# GPU device number, -1  means CPU
gpu = 0
if gpu >= 0:
    tagger = tagger.cuda(device=0)

# Get tags as sequences of strings
output_tag_sequences = tagger.predict_tags_from_tokens(token_sequences)

# Get scores
targets_idx = sequences_indexer.tag2idx(tag_sequences)
outputs_idx = sequences_indexer.tag2idx(output_tag_sequences)
acc = Evaluator.get_accuracy_token_level(targets_idx, outputs_idx)

f1, precision, recall, _, _, _ = Evaluator.get_f1_tokens_tags(tag_sequences, output_tag_sequences, match_alpha_ratio=0.999)
print('\nmatch_alpha_ratio = %1.1f | Accuracy = %1.2f, F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n' % (0.999, acc, f1, precision, recall))

f1, precision, recall, _, _, _ = Evaluator.get_f1_tokens_tags(tag_sequences, output_tag_sequences, match_alpha_ratio=0.5)
print('\nmatch_alpha_ratio = %1.1f | Accuracy = %1.2f, F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n' % (0.5, acc, f1, precision, recall))

# Macro-F1 for each class
#print(Evaluator.get_f1_scores_details(tagger, token_sequences, tag_sequences)) # TBD

# Write results to text file
write_CoNNL_dat_abs('out.dat.abs', token_sequences, output_tag_sequences)

print('\nThe end.')