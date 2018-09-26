from __future__ import print_function

import argparse
from math import ceil
from os.path import isfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from classes.data_io import DataIO
from classes.datasets_bank import DatasetsBankSorted
from classes.evaluator import Evaluator
from classes.report import Report
from classes.utils import *
from seq_indexers.seq_indexer_word import SeqIndexerWord
from seq_indexers.seq_indexer_tag import SeqIndexerTag
from models.tagger_io import TaggerIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--model', default='BiRNNCNNCRF', help='Tagger model: "BiRNN", "BiRNNCNN", "BiRNNCRF", '
                                                               '"BiRNNCNNCRF".')
    parser.add_argument('--fn_train', default='data/NER/CoNNL_2003_shared_task/train.txt',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--fn_dev', default='data/NER/CoNNL_2003_shared_task/dev.txt',
                        help='Dev data in CoNNL-2003 format, it is used to find best model during the training.')
    parser.add_argument('--fn_test', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Test data in CoNNL-2003 format, it is used to obtain the final accuracy/F1 score.')
    parser.add_argument('--emb_fn', default='embeddings/glove.6B.100d.txt', help='Path to word embeddings file.')
    parser.add_argument('--emb_dim', type=int, default=100, help='Dimension of word embeddings file.')
    parser.add_argument('--emb_delimiter', default=' ', help='Delimiter for word embeddings file.')
    parser.add_argument('--freeze_word_embeddings', type=bool, default=False, help='False to continue training the \                                                                                    word embeddings.')
    parser.add_argument('--freeze_char_embeddings', type=bool, default=False,
                        help='False to continue training the char embeddings.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1  means CPU.')
    parser.add_argument('--check_for_lowercase', type=bool, default=True, help='Read characters caseless.')
    parser.add_argument('--epoch_num', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--min_epoch_num', type=int, default=50, help='Minimum number of epochs.')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='Number hidden units in the recurrent layer.')
    parser.add_argument('--rnn_type', default='GRU', help='RNN cell units type: "Vanilla", "LSTM", "GRU".')
    parser.add_argument('--char_embeddings_dim', type=int, default=25, help='Char embeddings dim, only for char CNNs.')
    parser.add_argument('--word_len', type=int, default=20, help='Max length of words in characters for char CNNs.')
    parser.add_argument('--char_cnn_filter_num', type=int, default=30, help='Number of filters in Char CNN.')
    parser.add_argument('--char_window_size', type=int, default=3, help='Convolution1D size.')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--clip_grad', type=float, default=5, help='Clipping gradients maximum L2 norm.')
    parser.add_argument('--opt_method', default='sgd', help='Optimization method: "sgd", "adam".')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--lr', type=float, default=0.015, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='Learning decay rate.') # 0.05
    parser.add_argument('--momentum', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--verbose', type=bool, default=True, help='Show additional information.')
    parser.add_argument('--seed_num', type=int, default=42, help='Random seed number, but 42 is the best forever!')
    parser.add_argument('--load_checkpoint_fn', default=None, help='Path to load from the trained model.')
    parser.add_argument('--save_checkpoint_fn', default=None, help='Path to save the trained model.')
    parser.add_argument('--match_alpha_ratio', type=float, default='0.999',
                        help='Alpha ratio from non-strict matching, options: 0.999 or 0.5')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping.')
    parser.add_argument('--word_seq_indexer_path', type=str, default=None, help='Load word_seq_indexer object from hdf5\ '
                                                                                'file.')
    parser.add_argument('--save_best', type=bool, default=True, help = 'Save best on dev model as a final.')

    args = parser.parse_args()
    # Custom params
    args.word_seq_indexer_path = 'wsi_NER.hdf5'
    args.save_checkpoint_fn = 'tagger_LSTM_NER.hdf5'
    args.rnn_type = 'LSTM'

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    # Load CoNNL data as sequences of strings of words and corresponding tags
    word_sequences_train, tag_sequences_train = DataIO.read_CoNNL_universal(args.fn_train, verbose=True)
    word_sequences_dev, tag_sequences_dev = DataIO.read_CoNNL_universal(args.fn_dev, verbose=True)
    word_sequences_test, tag_sequences_test = DataIO.read_CoNNL_universal(args.fn_test, verbose=True)

    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches from them
    datasets_bank = DatasetsBankSorted(verbose=True)
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)

    # Word_seq_indexer converts lists of lists of words to lists of lists of integer indices and back
    if args.word_seq_indexer_path is not None and isfile(args.word_seq_indexer_path):
        word_seq_indexer = torch.load(args.word_seq_indexer_path)
    else:
        word_seq_indexer = SeqIndexerWord(gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
                                          embeddings_dim=args.emb_dim, verbose=True)
        word_seq_indexer.load_items_from_embeddings_file_and_unique_words_list(emb_fn=args.emb_fn,
                                                                      emb_delimiter=args.emb_delimiter,
                                                                      unique_words_list=datasets_bank.unique_words_list)
    if args.word_seq_indexer_path is not None and not isfile(args.word_seq_indexer_path):
        torch.save(word_seq_indexer, args.word_seq_indexer_path)

    # Tag_seq_indexer converts lists of lists of tags to lists of lists of integer indices and back
    tag_seq_indexer = SeqIndexerTag(gpu=args.gpu)
    tag_seq_indexer.load_items_from_tag_sequences(tag_sequences_train)

    # Create or load pre-trained tagger
    if args.load_checkpoint_fn is None:
        tagger = TaggerIO.create_tagger(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train)
    else:
        tagger = TaggerIO.load_tagger(args.load_checkpoint_fn, args.gpu)

    # Create optimizer
    if args.opt_method == 'sgd':
        optimizer = optim.SGD(list(tagger.parameters()), lr=args.lr, momentum=args.momentum)
    elif args.opt_method == 'adam':
        optimizer = optim.Adam(list(tagger.parameters()), lr=args.lr, betas=(0.9, 0.999))
    else:
        raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + args.lr_decay*epoch))
    iterations_num = int(datasets_bank.train_data_num / args.batch_size)
    best_f1_dev = -1
    patience_counter = 0
    report_fn = 'report_%s_%s_batch%d_%dep.txt' % (get_datetime_str(), args.model, args.batch_size, args.epoch_num)
    report = Report(report_fn, args, score_names=('train loss', 'f1-train', 'f1-dev', 'f1-test', 'acc. train', 'acc. dev',
                                                  'acc. test'))
    print('\nStart training...\n')
    for epoch in range(0, args.epoch_num + 1):
        time_start = time.time()
        loss_sum = 0
        if epoch > 0:
            tagger.train()
            if args.lr_decay > 0:
                scheduler.step()
            for i, (word_sequences_train_batch, tag_sequences_train_batch) in enumerate(datasets_bank.get_train_batches(args.batch_size)):
                tagger.train()
                tagger.zero_grad()
                loss = tagger.get_loss(word_sequences_train_batch, tag_sequences_train_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(tagger.parameters(), args.clip_grad)
                optimizer.step()
                loss_sum += loss.item()
                if i % 1 == 0:
                    print('\r-- train epoch %d/%d, batch %d/%d (%1.2f%%), loss = %1.2f.' % (epoch, args.epoch_num, i + 1,
                                                                                            iterations_num,
                                                                                            ceil(i*100.0/iterations_num),
                                                                                            loss_sum*100 / iterations_num),
                                                                                            end='', flush=True)
        # Evaluate tagger
        f1_train, f1_dev, f1_test, acc_train, acc_dev, acc_test = Evaluator.get_evaluation_train_dev_test(tagger,
                                                                                                          datasets_bank,
                                                                                                          batch_size=100)
        print('\n== eval epoch %d/%d train / dev / test | micro-f1: %1.2f / %1.2f / %1.2f, acc: %1.2f%% / %1.2f%% / %1.2f%%.'
              %(epoch, args.epoch_num, f1_train, f1_dev, f1_test, acc_train, acc_dev, acc_test))
        report.write_epoch_scores(epoch, (loss_sum*100 / iterations_num, f1_train, f1_dev, f1_test, acc_train, acc_dev,
                                          acc_test))
        # Save curr tagger
        # tagger.save('tagger_NER_epoch_%03d.hdf5' % epoch)

        # Early stopping
        if f1_dev > best_f1_dev:
            best_f1_dev = f1_dev
            patience_counter = 0
            if args.save_checkpoint_fn is not None and args.save_best:
                tagger.save_tagger(args.save_checkpoint_fn)
            print('## [BEST epoch], %d seconds.\n' % (time.time() - time_start))
        else:
            patience_counter += 1
            print('## [no improvement micro-f1 on DEV during the last %d epochs (f1_dev=%1.2f), %d seconds].\n' %
                                                                                                 (patience_counter,
                                                                                                 f1_dev,
                                                                                                 (time.time()-time_start)))
        if patience_counter > args.patience and epoch > args.min_epoch_num:
            break

    # Save final trained tagger to disk
    if args.save_checkpoint_fn is not None and not args.save_best:
        tagger.save_tagger(args.save_checkpoint_fn)

    # Make final evaluation of trained tagger
    output_tag_sequences_test = tagger.predict_tags_from_words(datasets_bank.word_sequences_test, batch_size=100)
    f1_test_final, test_connl_str = Evaluator.get_f1_connl_script(tagger=tagger,
                                                     word_sequences=datasets_bank.word_sequences_test,
                                                     targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                     outputs_tag_sequences=output_tag_sequences_test)
    report.write_final_score(f1_test_final)
    print(report.text)
    print(test_connl_str)
