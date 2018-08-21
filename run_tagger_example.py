from __future__ import print_function

import torch
from classes.data_io import DataIO
from classes.evaluator import Evaluator
from models.tagger_base import TaggerBase

print('Start run_tagger_example.py.')

# Read data in CoNNL-2003 dat.abs format (Eger, 2017)
#fn = 'data/persuasive_essays/Paragraph_Level/test.dat.abs'
fn = 'data/NER/CoNNL_2003_shared_task/test.txt' # NER task
gpu = 0 # GPU device number, -1  means CPU

word_sequences, targets_tag_sequences = DataIO.read_CoNNL_universal(fn)

# Load tagger model
fn_checkpoint = 'tagger_model_BiRNN_NER_10ep.hdf5'
tagger = TaggerBase.load(fn_checkpoint, gpu)

# Get tags as sequences of strings
outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size=10)

# Calculate scores
acc = Evaluator.get_accuracy_token_level(targets_tag_sequences=targets_tag_sequences,
                                         outputs_tag_sequences=outputs_tag_sequences,
                                         tag_seq_indexer=tagger.tag_seq_indexer)
f1_100, precision_100, recall_100, _ = Evaluator.get_f1_from_words(targets_tag_sequences=targets_tag_sequences,
                                                                   outputs_tag_sequences=outputs_tag_sequences,
                                                                   match_alpha_ratio=0.999)

f1_50, precision_50, recall_50, _ = Evaluator.get_f1_from_words(targets_tag_sequences=targets_tag_sequences,
                                                                outputs_tag_sequences=outputs_tag_sequences,
                                                                match_alpha_ratio=0.5)

connl_report_str = Evaluator.get_f1_from_words_connl_script(word_sequences=word_sequences,
                                                            targets_tag_sequences=targets_tag_sequences,
                                                            outputs_tag_sequences=outputs_tag_sequences)

# Prepare text report
scores_report_str = '\nResults : Accuracy = %1.2f.' % acc
scores_report_str += '\nmatch_alpha_ratio = %1.1f | F1-100%% = %1.2f, Precision-100%% = %1.2f, Recall-100%% = %1.2f.' \
                    % (0.999, f1_100, precision_100, recall_100)
scores_report_str += '\nmatch_alpha_ratio = %1.1f | F1-50%% = %1.2f, Precision-50%% = %1.2f, Recall-50%% = %1.2f.' \
                     % (0.5, f1_50, precision_50, recall_50)
scores_report_str += '\n' + connl_report_str
print(scores_report_str)

# Write results to text file
#DataIO.write_CoNNL_dat_abs('out_test.dat.abs', word_sequences, output_tag_sequences)

# Another format, true CoNNL
#DataIO.write_CoNNL_2003_two_columns('out_test.txt', word_sequences, targets_tag_sequences, outputs_tag_sequences)

print('\nThe end.')
