from __future__ import print_function

import argparse
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_tagger import TaggerFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trained tagger from the checkpoint file')
    parser.add_argument('--fn', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('-d', '--data-io', choices=['connl-abs', 'connl-2003'], default='connl-2003',
                        help='Data read/write file format.')
    parser.add_argument('--checkpoint_fn', default='pretrained/tagger_NER_BiLSTMCNNCRF.hdf5',
                        help='Path to load the trained model.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1  means CPU.')
    parser.add_argument('-v', '--evaluator', choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'token-acc'],
                        default='f1-connl', help='Evaluation method.')
    print('Start run_tagger.py.')
    args = parser.parse_args()
    # Create DataIO object
    data_io = DataIOFactory.create(args)
    # Read data in CoNNL-2003 file format format
    word_sequences_test, targets_tag_sequences_test = data_io.read(args.fn)
    # Load tagger model
    tagger = TaggerFactory.load(args.checkpoint_fn, args.gpu)
    # Create evaluator
    evaluator = EvaluatorFactory.create(args)
    # Get tags as sequences of strings
    output_tag_sequences_test = tagger.predict_tags_from_words(word_sequences_test, batch_size=100)
    test_score, test_msg = evaluator.get_evaluation_score(targets_tag_sequences=targets_tag_sequences_test,
                                                          outputs_tag_sequences=output_tag_sequences_test,
                                                          word_sequences=word_sequences_test)
    # Show the evaluation results
    print('\n\n%s = %1.2f' % (args.evaluator, test_score))
    print(test_msg)
    # Write results to text file
    data_io.write(fn='out.txt', word_sequences=word_sequences_test, tag_sequences_1=targets_tag_sequences_test,
                  tag_sequences_2=output_tag_sequences_test)
    print('\nThe end.')
