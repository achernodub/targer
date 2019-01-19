from __future__ import print_function
from math import ceil, floor
from os.path import isfile
import time
import numpy as np
import torch.nn as nn
from src.classes.report import Report
from src.classes.utils import *
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_optimizer import OptimizerFactory
from src.factories.factory_tagger import TaggerFactory
from src.seq_indexers.seq_indexer_tag import SeqIndexerTag
from src.seq_indexers.seq_indexer_word import SeqIndexerWord


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
    parser.add_argument('--train', default='data/NER/CoNNL_2003_shared_task/train.txt',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--dev', default='data/NER/CoNNL_2003_shared_task/dev.txt',
                        help='Development data in format defined by --data-io param.')
    parser.add_argument('--test', default='data/NER/CoNNL_2003_shared_task/test.txt',
                        help='Test data in format defined by --data-io param.')
    parser.add_argument('-d', '--data-io', choices=['connl-ner-2003', 'connl-pe', 'connl-wd'],
                        default='connl-ner-2003', help='Data read/write file format.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')
    parser.add_argument('--model', help='Tagger model.', choices=['BiRNN', 'BiRNNCNN', 'BiRNNCRF', 'BiRNNCNNCRF'],
                        default='BiRNNCNNCRF')
    parser.add_argument('--load', '-l', default=None, help='Path to load from the trained model.')
    parser.add_argument('--save', '-s', default='%s_tagger.hdf5' % get_datetime_str(),
                        help='Path to save the trained model.')
    parser.add_argument('--word-seq-indexer', '-w', type=str, default=None,
                        help='Load word_seq_indexer object from hdf5 file.')
    parser.add_argument('--epoch-num', '-e',  type=int, default=100, help='Number of epochs.')
    parser.add_argument('--min-epoch-num', '-n', type=int, default=50, help='Minimum number of epochs.')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--evaluator', '-v', default='f1-connl', help='Evaluation method.',
                        choices=['f1-connl', 'f1-alpha-match-10', 'f1-alpha-match-05', 'f1-macro', 'token-acc'])
    parser.add_argument('--save-best', type=str2bool, default=False, help = 'Save best on dev model as a final model.',
                        nargs='?', choices=['yes', True, 'no (default)', False])
    parser.add_argument('--dropout-ratio', '-r', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--opt-method', '-o', help='Optimization method.', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lr-decay', type=float, default=0.05, help='Learning decay rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--clip-grad', type=float, default=5, help='Clipping gradients maximum L2 norm.')
    parser.add_argument('--rnn-type', help='RNN cell units type.', choices=['Vanilla', 'LSTM', 'GRU'], default='LSTM')
    parser.add_argument('--rnn-hidden-dim', type=int, default=100, help='Number hidden units in the recurrent layer.')
    parser.add_argument('--emb-fn', default='embeddings/glove.6B.100d.txt', help='Path to word embeddings file.')
    parser.add_argument('--emb-dim', type=int, default=100, help='Dimension of word embeddings file.')
    parser.add_argument('--emb-delimiter', default=' ', help='Delimiter for word embeddings file.')
    parser.add_argument('--emb-load-all', type=str2bool, default=False, help='Load all embeddings to model.', nargs='?',
                        choices = ['yes', True, 'no (default)', False])
    parser.add_argument('--freeze-word-embeddings', type=str2bool, default=False,
                        help='False to continue training the word embeddings.', nargs='?',
                        choices=['yes', True, 'no (default)', False])
    parser.add_argument('--check-for-lowercase', type=str2bool, default=True, help='Read characters caseless.',
                        nargs='?', choices=['yes (default)', True, 'no', False])
    parser.add_argument('--char-embeddings-dim', type=int, default=25, help='Char embeddings dim, only for char CNNs.')
    parser.add_argument('--char-cnn_filter-num', type=int, default=30, help='Number of filters in Char CNN.')
    parser.add_argument('--char-window-size', type=int, default=3, help='Convolution1D size.')
    parser.add_argument('--freeze-char-embeddings', type=str2bool, default=False,
                        choices=['yes', True, 'no (default)', False], nargs='?',
                        help='False to continue training the char embeddings.')
    parser.add_argument('--word-len', type=int, default=20, help='Max length of words in characters for char CNNs.')
    parser.add_argument('--dataset-sort', type=str2bool, default=False, help='Sort sequences by length for training.',
                        nargs='?', choices=['yes', True, 'no (default)', False])
    parser.add_argument('--seed-num', type=int, default=42, help='Random seed number, note that 42 is the answer.')
    parser.add_argument('--report-fn', type=str, default='%s_report.txt' % get_datetime_str(), help='Report filename.')
    parser.add_argument('--cross-folds-num', type=int, default=-1,
                        help='Number of folds for cross-validation (optional, for some datasets).')
    parser.add_argument('--cross-fold-id', type=int, default=-1,
                        help='Current cross-fold, 1<=cross-fold-id<=cross-folds-num (optional, for some datasets).')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                        choices=['yes (default)', True, 'no', False])
    args = parser.parse_args()
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)
    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test = data_io.read_train_dev_test(args)
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
    datasets_bank = DatasetsBankFactory.create(args)
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)
    # Word_seq_indexer converts lists of lists of words to lists of lists of integer indices and back
    if args.word_seq_indexer is not None and isfile(args.word_seq_indexer):
        word_seq_indexer = torch.load(args.word_seq_indexer)
    else:
        word_seq_indexer = SeqIndexerWord(gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
                                          embeddings_dim=args.emb_dim, verbose=True)
        word_seq_indexer.load_items_from_embeddings_file_and_unique_words_list(emb_fn=args.emb_fn,
                                                                               emb_delimiter=args.emb_delimiter,
                                                                               emb_load_all=args.emb_load_all,
                                                                               unique_words_list=datasets_bank.unique_words_list)
    if args.word_seq_indexer is not None and not isfile(args.word_seq_indexer):
        torch.save(word_seq_indexer, args.word_seq_indexer)
    # Tag_seq_indexer converts lists of lists of tags to lists of lists of integer indices and back
    tag_seq_indexer = SeqIndexerTag(gpu=args.gpu)
    tag_seq_indexer.load_items_from_tag_sequences(tag_sequences_train)
    # Create or load pre-trained tagger
    if args.load is None:
        tagger = TaggerFactory.create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train)
    else:
        tagger = TaggerFactory.load(args.load, args.gpu)
    # Create evaluator
    evaluator = EvaluatorFactory.create(args)
    # Create optimizer
    optimizer, scheduler = OptimizerFactory.create(args, tagger)
    # Prepare report and temporary variables for "save best" strategy
    report = Report(args.report_fn, args, score_names=('train loss', '%s-train' % args.evaluator,
                                                       '%s-dev' % args.evaluator, '%s-test' % args.evaluator))
    # Initialize training variables
    iterations_num = floor(datasets_bank.train_data_num / args.batch_size)
    best_dev_score = -1
    best_epoch = -1
    best_test_score = -1
    best_test_msg = 'N\A'
    patience_counter = 0
    print('\nStart training...\n')
    for epoch in range(0, args.epoch_num + 1):
        time_start = time.time()
        loss_sum = 0
        if epoch > 0:
            tagger.train()
            if args.lr_decay > 0:
                scheduler.step()
            for i, (word_sequences_train_batch, tag_sequences_train_batch) in \
                    enumerate(datasets_bank.get_train_batches(args.batch_size)):
                tagger.train()
                tagger.zero_grad()
                loss = tagger.get_loss(word_sequences_train_batch, tag_sequences_train_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(tagger.parameters(), args.clip_grad)
                optimizer.step()
                loss_sum += loss.item()
                if i % 1 == 0:
                    print('\r-- train epoch %d/%d, batch %d/%d (%1.2f%%), loss = %1.2f.' % (epoch, args.epoch_num,
                                                                                         i + 1, iterations_num,
                                                                                         ceil(i*100.0/iterations_num),
                                                                                         loss_sum*100 / iterations_num),
                                                                                         end='', flush=True)
        # Evaluate tagger
        train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                     datasets_bank,
                                                                                                     batch_size=100)
        print('\n== eval epoch %d/%d "%s" train / dev / test | %1.2f / %1.2f / %1.2f.' % (epoch, args.epoch_num,
                                                                                        args.evaluator, train_score,
                                                                                        dev_score, test_score))
        report.write_epoch_scores(epoch, (loss_sum*100 / iterations_num, train_score, dev_score, test_score))
        # Save curr tagger if required
        # tagger.save('tagger_NER_epoch_%03d.hdf5' % epoch)
        # Early stopping
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_test_score = test_score
            best_epoch = epoch
            best_test_msg = test_msg
            patience_counter = 0
            if args.save is not None and args.save_best:
                tagger.save_tagger(args.save)
            print('## [BEST epoch], %d seconds.\n' % (time.time() - time_start))
        else:
            patience_counter += 1
            print('## [no improvement micro-f1 on DEV during the last %d epochs (best_f1_dev=%1.2f), %d seconds].\n' %
                                                                                            (patience_counter,
                                                                                             best_dev_score,
                                                                                             (time.time()-time_start)))
        if patience_counter > args.patience and epoch > args.min_epoch_num:
            break
    # Save final trained tagger to disk, if it is not already saved according to "save best"
    if args.save is not None and not args.save_best:
        tagger.save_tagger(args.save)
    # Show and save the final scores
    if args.save_best:
        report.write_final_score('Final eval on test, "save best", best epoch on dev %d, %s, test = %1.2f)' %
                                 (best_epoch, args.evaluator, best_test_score))
        report.write_msg(best_test_msg)
    else:
        report.write_final_score('Final eval on test, %s test = %1.2f)' % (args.evaluator, test_score))
        report.write_msg(test_msg)
        print(test_msg)
    report.write_input_arguments()
    print('Input arguments:\n' + get_input_arguments())
