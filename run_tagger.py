from __future__ import print_function

import argparse

from classes.data_io import DataIO
from classes.evaluator import Evaluator
from models.tagger_io import TaggerIO

print('Start run_tagger.py.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trained tagger from the checkpoint file')
    parser.add_argument('--fn', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--checkpoint_fn', default='pretrained/tagger_NER_BiLSTMCNNCRF.hdf5', help='Path to load the trained model.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1  means CPU.')
    args = parser.parse_args()

    # Read data in CoNNL-2003 file format format
    word_sequences_test, targets_tag_sequences_test = DataIO.read_CoNNL_universal(args.fn)

    # Load tagger model
    tagger = TaggerIO.load_tagger(args.checkpoint_fn, args.gpu)

    # Get tags as sequences of strings
    output_tag_sequences_test = tagger.predict_tags_from_words(word_sequences_test, batch_size=100)
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