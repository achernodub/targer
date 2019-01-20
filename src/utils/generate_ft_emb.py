"""generate predefined FastText embeddings for dataset"""
from __future__ import print_function
import argparse
import fastText as ft
from src.data_io.data_io_connl_ner_2003 import DataIOConnlNer2003
from src.classes.datasets_bank import DatasetsBank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--train', default='data/NER/CoNNL_2003_shared_task/train.txt',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--dev', default='data/NER/CoNNL_2003_shared_task/dev.txt',
                        help='Dev data in CoNNL-2003 format, it is used to find best model during the training.')
    parser.add_argument('--test', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Test data in CoNNL-2003 format, it is used to obtain the final accuracy/F1 score.')
    parser.add_argument('--fn-fasttext-emb-bin', default='embeddings/wiki.en.bin', help='Fasttext binary model.')
    parser.add_argument('--fn_out', default='out.txt', help='Output file.')
    args = parser.parse_args()
    # Load CoNNL data as sequences of strings of words and corresponding tags
    data_io = DataIOConnlNer2003()
    word_sequences_train, tag_sequences_train = data_io.read(args.train, verbose=False)
    word_sequences_dev, tag_sequences_dev = data_io.read(args.dev, verbose=False)
    word_sequences_test, tag_sequences_test = data_io.read(args.test, verbose=True)
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches from them
    datasets_bank = DatasetsBank(verbose=True)
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)
    # Load FastText binary model
    ft_bin_model = ft.load_model(args.fn_fasttext_emb_bin)
    out_file = open(args.fn_out, 'w')
    for n, word in enumerate(datasets_bank.unique_words_list):
        emb_vector = ft_bin_model.get_word_vector(word)
        emb_vector_str = word
        for v in emb_vector:
            emb_vector_str += ' %1.5f' % v
        emb_vector_str += '\n'
        out_file.write(emb_vector_str)
        if n % 100 == 0:
            print('\r-- process word %d/%d.' % (n, len(datasets_bank.unique_words_list)), end='', flush=True)
    print('\n')
    out_file.close()
