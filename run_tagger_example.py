from __future__ import print_function

import os.path
import torch
from classes.data_io import DataIO
from classes.evaluator import Evaluator
from models.tagger_base import TaggerBase

print('Start run_tagger_example.py.')

# Read data in CoNNL-2003 dat.abs format (Eger, 2017)
fn = 'data/persuasive_essays/Paragraph_Level/test.dat.abs'
word_sequences, tag_sequences = DataIO.read_CoNNL_dat_abs(fn)

# Load tagger model
fn_checkpoint = 'tagger_model_temp.hdf5'
if os.path.isfile(fn_checkpoint):
    tagger = TaggerBase.load(fn_checkpoint)
else:
    raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty "--save_best_path" param to create it.' % fn_checkpoint)

# GPU device number, -1  means CPU
gpu = 0
if gpu >= 0:
    tagger = tagger.cuda(device=0)

# Get tags as sequences of strings
output_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size=10)

# Calculate scores
outputs_tag_sequences_test = tagger.predict_tags_from_words(word_sequences)
acc = Evaluator.get_accuracy_token_level(targets_tag_sequences=tag_sequences,
                                         outputs_tag_sequences=output_tag_sequences,
                                         tag_seq_indexer=tagger.tag_seq_indexer)
f1_100, precision_100, recall_100, _ = Evaluator.get_f1_from_words(targets_tag_sequences=tag_sequences,
                                                                   outputs_tag_sequences=output_tag_sequences,
                                                                   match_alpha_ratio=0.999)

f1_50, precision_50, recall_50, _ = Evaluator.get_f1_from_words(targets_tag_sequences=tag_sequences,
                                                                outputs_tag_sequences=output_tag_sequences,
                                                                match_alpha_ratio=0.5)

scores_report_str = '\nResults : Accuracy = %1.2f.' % acc
scores_report_str += '\nmatch_alpha_ratio = %1.1f | F1-100%% = %1.2f, Precision-100%% = %1.2f, Recall-100%% = %1.2f.' \
                    % (0.999, f1_100, precision_100, recall_100)
scores_report_str += '\nmatch_alpha_ratio = %1.1f | F1-50%% = %1.2f, Precision-50%% = %1.2f, Recall-50%% = %1.2f.' \
                     % (0.5, f1_50, precision_50, recall_50)
print(scores_report_str)

# Write results to text file
DataIO.write_CoNNL_dat_abs('out.dat.abs', word_sequences, output_tag_sequences)

print('\nThe end.')