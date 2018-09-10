from __future__ import print_function

from classes.data_io import DataIO
from classes.evaluator import Evaluator
from models.tagger_base import TaggerBase

print('Start run_tagger_example.py.')

fn = 'data/NER/CoNNL_2003_shared_task/test.txt'
gpu = 0 # GPU device number, -1  means work on CPU

# Read data in CoNNL-2003 file format format
word_sequences_test, targets_tag_sequences_test = DataIO.read_CoNNL_universal(fn)

# Load tagger model
fn_checkpoint = 'tagger_NER.hdf5'
tagger = TaggerBase.load(fn_checkpoint, gpu)

# Get tags as sequences of strings
output_tag_sequences_test = tagger.predict_tags_from_words(word_sequences_test)
f1_test_final, test_connl_str = Evaluator.get_f1_connl_script(tagger=tagger,
                                                              word_sequences=word_sequences_test,
                                                              targets_tag_sequences=targets_tag_sequences_test,
                                                              outputs_tag_sequences=output_tag_sequences_test)
# Show the evaluation results
print('\nMicro f1 score = %1.2f' % f1_test_final)
print(test_connl_str)

# Write results to text file
DataIO.write_CoNNL_2003_two_columns(fn='out.txt',
                                    word_sequences=word_sequences_test,
                                    tag_sequences_1=targets_tag_sequences_test,
                                    tag_sequences_2=output_tag_sequences_test)
print('\nThe end.')