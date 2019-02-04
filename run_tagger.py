from __future__ import print_function
import argparse
import json
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_tagger import TaggerFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trained tagger from the checkpoint file')
    parser.add_argument('load', help='Path to load from the trained model.',
                        default='pretrained/tagger_NER_BiLSTMCNNCRF.hdf5')
    parser.add_argument('-d', '--data-io', choices=['connl-ner-2003', 'connl-pe', 'connl-wd'],
                        default='connl-wd', help='Data read/write file format.')
    parser.add_argument('--fn', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Test data.')
    parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
                        choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1 means CPU.')
    print('Start run_tagger.py.')
    args = parser.parse_args()
    # Create DataIO object
    data_io = DataIOFactory.create(args)
    # Read data in CoNNL-2003 file format format
    word_sequences, targets_tag_sequences_test = data_io.read_data(args.fn)
    # Load tagger model
    tagger = TaggerFactory.load(args.load, args.gpu)
    # Create evaluator
    evaluator = EvaluatorFactory.create(args)
    # Get tags as sequences of strings
    output_tag_sequences_test = tagger.predict_tags_from_words(word_sequences, batch_size=100)
    test_score, test_msg = evaluator.get_evaluation_score(targets_tag_sequences=targets_tag_sequences_test,
                                                          outputs_tag_sequences=output_tag_sequences_test,
                                                          word_sequences=word_sequences)
    # Show the evaluation results
    print('\n\n%s = %1.2f' % (args.evaluator, test_score))
    print(test_msg)
    # Write results to text file
    with open('out.txt', 'w') as f:
        json.dump(output_tag_sequences_test, f)
    print('\nThe end.')
