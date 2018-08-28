from __future__ import print_function

import argparse
import datetime
import time
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from classes.data_io import DataIO
from classes.datasets_bank import DatasetsBank
from classes.element_seq_indexer import ElementSeqIndexer
from classes.evaluator import Evaluator
from classes.utils import write_textfile

from models.tagger_birnn import TaggerBiRNN
from models.tagger_birnn_cnn import TaggerBiRNNCNN
from models.tagger_birnn_cnn_crf import TaggerBiRNNCNNCRF

def tok_num(word_sequences):
    n = 0
    for word_seq in word_sequences:
        for token in word_seq:
            n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--model', default='BiRNN', help='Tagger model: "BiRNN" or "BiRNNCNN".')
    parser.add_argument('--fn_train', default='data/persuasive_essays/Paragraph_Level/train.dat.abs',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--fn_dev', default='data/persuasive_essays/Paragraph_Level/dev.dat.abs',
                        help='Dev data in CoNNL-2003 format, it is used to find best model during the training.')
    parser.add_argument('--fn_test', default='data/persuasive_essays/Paragraph_Level/test.dat.abs',
                        help='Test data in CoNNL-2003 format, it is used to obtain the final accuracy/F1 score.')
    parser.add_argument('--emb_fn', default='embeddings/glove.6B.100d.txt', help='Path to embeddings file.')
    parser.add_argument('--emb_delimiter', default=' ', help='Delimiter for embeddings file.')
    parser.add_argument('--freeze_word_embeddings', type=bool, default=False, help='False to continue training the \
                                                                                    word embeddings.')
    parser.add_argument('--freeze_char_embeddings', type=bool, default=False,
                        help='False to continue training the char embeddings.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1  means CPU.')
    parser.add_argument('--caseless', type=bool, default=True, help='Read characters caseless.')
    parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='Number hidden units in the recurrent layer.')
    parser.add_argument('--rnn_type', default='GRU', help='RNN cell units type: "Vanilla", "LSTM", "GRU".')
    parser.add_argument('--char_embeddings_dim', type=int, default=25, help='Char embeddings dim, only for char CNNs.')
    parser.add_argument('--word_len', type=int, default=20, help='Max length of words in characters for char CNNs.')
    parser.add_argument('--char_cnn_filter_num', type=int, default=30, help='Number of filters in Char CNN.')
    parser.add_argument('--char_window_size', type=int, default=3, help='Convolution1D size.')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Clipping gradients maximum L2 norm.')
    parser.add_argument('--opt_method', default='sgd', help='Optimization method: "sgd", "adam".')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0, help='Learning decay rate.') # 0.05
    parser.add_argument('--momentum', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--verbose', type=bool, default=True, help='Show additional information.')
    parser.add_argument('--seed_num', type=int, default=43, help='Random seed number, but 42 is the best forever!')
    parser.add_argument('--checkpoint_fn', default=None, help='Path to save the trained model.')
    parser.add_argument('--report_fn', default='report_%d_%02d_%02d_%02d_%02d.txt' % (datetime.datetime.now().year,
                                                                                      datetime.datetime.now().month,
                                                                                      datetime.datetime.now().day,
                                                                                      datetime.datetime.now().hour,
                                                                                      datetime.datetime.now().minute),
                                                                                      help='Path to report.')
    parser.add_argument('--match_alpha_ratio', type=float, default='0.999',
                        help='Alpha ratio from non-strict matching, options: 0.999 or 0.5')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--load_word_seq_indexer', type=str, default=None, help='Load word_seq_indexer object from hdf5 file.')


    args = parser.parse_args()

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    # Custom params here to replace the defaults
    args.fn_train = 'data/NER/CoNNL_2003_shared_task/train.txt'
    args.fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.txt'
    args.fn_test = 'data/NER/CoNNL_2003_shared_task/test.txt'

    #from utils_nlp import convert_conll_from_bio_to_bioes
    #convert_conll_from_bio_to_bioes(args.fn_train, 'data/NER/CoNNL_2003_shared_task/train.bioes.txt')
    #convert_conll_from_bio_to_bioes(args.fn_dev, 'data/NER/CoNNL_2003_shared_task/dev.bioes.txt')
    #convert_conll_from_bio_to_bioes(args.fn_test, 'data/NER/CoNNL_2003_shared_task/test.bioes.txt')

    args.fn_train = 'data/NER/CoNNL_2003_shared_task/train.bioes.txt'
    args.fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.bioes.txt'
    args.fn_test = 'data/NER/CoNNL_2003_shared_task/test.bioes.txt'
    #args.load_word_seq_indexer = 'word_seq_indexer_CoNNL_2003.hdf5'

    #args.fn_train = 'data/persuasive_essays/Essay_Level/train.dat.abs'
    #args.fn_dev = 'data/persuasive_essays/Essay_Level/dev.dat.abs'
    #args.fn_test = 'data/persuasive_essays/Essay_Level/test.dat.abs'

    args.model = 'BiRNN'
    #args.model = 'BiRNNCNN'
    #args.model = 'BiRNNCNNCRF'
    args.epoch_num = 50
    args.rnn_hidden_dim = 100
    #args.batch_size = 10
    #args.gpu = -1
    args.lr_decay = 0.05
    args.rnn_type = 'LSTM'
    #args.checkpoint_fn = 'tagger_model_BiRNNCNN_NER_nosb.hdf5'
    #args.report_fn = 'report_model_BiRNN_NER_300_LSTM.txt'
    #args.report_fn = 'report_model_BiRNNCNNCRF7_NER.txt'
    #args.checkpoint_fn = 'tagger_model_BiRNNCNNCRF7_NER.hdf5'

    #from layers.layer_bilstm import LayerBiLSTM
    #birnn_layer = LayerBiLSTM(input_dim=10, hidden_dim=5, gpu=0)
    #exit()

    # Load CoNNL data as sequences of strings of words and corresponding tags
    word_sequences_train, tag_sequences_train = DataIO.read_CoNNL_universal(args.fn_train)
    word_sequences_dev, tag_sequences_dev = DataIO.read_CoNNL_universal(args.fn_dev)
    word_sequences_test, tag_sequences_test = DataIO.read_CoNNL_universal(args.fn_test)

    print(tok_num(word_sequences_train))
    print(tok_num(word_sequences_dev))
    print(tok_num(word_sequences_test))

    # Converts lists of lists of tags to integer indices and back
    tag_seq_indexer = ElementSeqIndexer(gpu=args.gpu, caseless=False, verbose=args.verbose, pad='<pad>', unk=None,
                                        zero_digits=False)
    tag_seq_indexer.load_vocabulary_from_element_sequences(tag_sequences_train)

    # Word_seq_indexer converts lists of lists of words to integer indices and back; optionally contains embeddings
    if args.load_word_seq_indexer is not None and os.path.isfile(args.load_word_seq_indexer):
        word_seq_indexer = torch.load(args.load_word_seq_indexer)
    else:
        word_seq_indexer = ElementSeqIndexer(gpu=args.gpu, caseless=args.caseless, load_embeddings=True,
                                                 verbose=args.verbose, pad='<pad>', unk='<unk>', zero_digits=True)
        word_seq_indexer.load_vocabulary_from_embeddings_file(emb_fn=args.emb_fn, emb_delimiter=args.emb_delimiter)
        #word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_train)
        #word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_dev)
        #word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_test, verbose=True)
        if args.load_word_seq_indexer is not None:
            torch.save(word_seq_indexer, args.load_word_seq_indexer)

    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches from them
    datasets_bank = DatasetsBank()
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)

    if args.model == 'BiRNN':
        tagger = TaggerBiRNN(word_seq_indexer=word_seq_indexer,
                             tag_seq_indexer=tag_seq_indexer,
                             class_num=tag_seq_indexer.get_class_num(),
                             rnn_hidden_dim=args.rnn_hidden_dim,
                             freeze_word_embeddings=args.freeze_word_embeddings,
                             dropout_ratio=args.dropout_ratio,
                             rnn_type=args.rnn_type,
                             gpu=args.gpu)
    elif args.model == 'BiRNNCNN':
        tagger = TaggerBiRNNCNN(word_seq_indexer=word_seq_indexer,
                                tag_seq_indexer=tag_seq_indexer,
                                class_num=tag_seq_indexer.get_class_num(),
                                rnn_hidden_dim=args.rnn_hidden_dim,
                                freeze_word_embeddings=args.freeze_word_embeddings,
                                dropout_ratio=args.dropout_ratio,
                                rnn_type=args.rnn_type,
                                gpu=args.gpu,
                                freeze_char_embeddings=args.freeze_char_embeddings,
                                char_embeddings_dim=args.char_embeddings_dim,
                                word_len=args.word_len,
                                char_cnn_filter_num=args.char_cnn_filter_num,
                                char_window_size=args.char_window_size)
    elif args.model == 'BiRNNCNNCRF':
        tagger = TaggerBiRNNCNNCRF(word_seq_indexer=word_seq_indexer,
                                   tag_seq_indexer=tag_seq_indexer,
                                   class_num=tag_seq_indexer.get_class_num(),
                                   rnn_hidden_dim=args.rnn_hidden_dim,
                                   freeze_word_embeddings=args.freeze_word_embeddings,
                                   dropout_ratio=args.dropout_ratio,
                                   rnn_type=args.rnn_type,
                                   gpu=args.gpu,
                                   freeze_char_embeddings=args.freeze_char_embeddings,
                                   char_embeddings_dim=args.char_embeddings_dim,
                                   word_len=args.word_len,
                                   char_cnn_filter_num=args.char_cnn_filter_num,
                                   char_window_size=args.char_window_size)

    else:
        raise ValueError('Unknown tagger model, must be one of "BiRNN"/"BiRNNCNN"/"BiRNNCNNCRF".')

    optimizer = optim.SGD(list(tagger.parameters()), lr=args.lr, momentum=args.momentum)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + args.lr_decay*epoch))
    iterations_num = int(datasets_bank.train_data_num / args.batch_size)
    best_f1_dev = -1
    patience_counter = 0
    report_str = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])+'\n' # hyperparams
    for epoch in range(1, args.epoch_num + 1):
        tagger.train()
        if args.lr_decay > 0:
            scheduler.step()
        time_start = time.time()
        best_epoch_msg = ''
        word_sequences_train_batch_list, tag_sequences_train_batch_list = datasets_bank.get_train_batches(args.batch_size)
        loss_sum = 0
        for i, (word_sequences_train_batch, tag_sequences_train_batch) in enumerate(zip(word_sequences_train_batch_list,
                                                                                        tag_sequences_train_batch_list)):
            tagger.zero_grad()
            loss = tagger.get_loss(word_sequences_train_batch, tag_sequences_train_batch)
            loss.backward()
            tagger.clip_gradients(args.clip_grad)
            optimizer.step()
            if i % 10 == 0:
                print('\r-- epoch %d, i = %d/%d (%1.2f%%), loss = %1.4f' % (epoch, i, iterations_num,
                                                                          i*100.0/iterations_num, loss.item()),
                                                                          end='', flush=True)
            loss_sum += loss.item()

        time_finish = time.time()
        outputs_tag_sequences_dev = tagger.predict_tags_from_words(datasets_bank.word_sequences_dev, batch_size=100)
        acc_dev = Evaluator.get_accuracy_token_level(targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                     outputs_tag_sequences=outputs_tag_sequences_dev,
                                                     tag_seq_indexer=tag_seq_indexer)

        f1_dev, _, _, _ = Evaluator.get_f1_from_words(targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                      outputs_tag_sequences=outputs_tag_sequences_dev,
                                                      match_alpha_ratio=0.999)

        connl_report_dev_str = Evaluator.get_f1_from_words_connl_script(word_sequences=datasets_bank.word_sequences_dev,
                                                                        targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                                        outputs_tag_sequences=outputs_tag_sequences_dev)

        if f1_dev > best_f1_dev:
            best_epoch_msg = '[BEST] '
            best_f1_dev = f1_dev
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_report = '\nDEV dataset' + connl_report_dev_str

        epoch_report += '\n%sEPOCH %d/%d, DEV dataset: loss = %1.2f, accuracy = %1.2f, F1-100%% = %1.2f  | %d sec.\n' % \
                                                                                              (best_epoch_msg,
                                                                                               epoch,
                                                                                               args.epoch_num,
                                                                                               loss_sum,
                                                                                               acc_dev,
                                                                                               f1_dev,
                                                                                               time.time() - time_start)
        report_str += epoch_report
        write_textfile(args.report_fn, report_str)
        print(epoch_report)
        print('No improvement on DEV during the last %d epochs.\n' % patience_counter)
        if patience_counter > args.patience:
            break

    outputs_tag_sequences_test = tagger.predict_tags_from_words(datasets_bank.word_sequences_test, batch_size=100)
    acc_test = Evaluator.get_accuracy_token_level(targets_tag_sequences=datasets_bank.tag_sequences_test,






                                                     outputs_tag_sequences=outputs_tag_sequences_test,
                                                  tag_seq_indexer=tag_seq_indexer)
    f1_100, precision_100, recall_100, _ = Evaluator.get_f1_from_words(targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                                       outputs_tag_sequences=outputs_tag_sequences_test,
                                                                       match_alpha_ratio=0.999)
    f1_50, precision_50, recall_50, _ = Evaluator.get_f1_from_words(targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                                    outputs_tag_sequences=outputs_tag_sequences_test,
                                                                    match_alpha_ratio=0.5)

    connl_report_test_str = Evaluator.get_f1_from_words_connl_script(word_sequences=datasets_bank.word_sequences_test,
                                                                     targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                                     outputs_tag_sequences=outputs_tag_sequences_test)

    # Prepare text report
    report_str += '\nResults on TEST (epoch = %d): Accuracy = %1.2f.' % (epoch, acc_test)
    report_str += '\nmatch_alpha_ratio = %1.1f | F1-100%% = %1.2f, Precision-100%% = %1.2f, Recall-100%% = %1.2f.' \
                  % (0.999, f1_100, precision_100, recall_100)
    report_str += '\nmatch_alpha_ratio = %1.1f | F1-50%% = %1.2f, Precision-50%% = %1.2f, Recall-50%% = %1.2f.' \
                  % (0.5, f1_50, precision_50, recall_50)
    report_str += '\n' + connl_report_test_str
    print(report_str)

    # Write final text report
    write_textfile(args.report_fn, report_str)

    # Save best tagger to disk
    if args.checkpoint_fn is not None:
        tagger.save(args.checkpoint_fn)

    print('The end!')
