from __future__ import print_function

import argparse
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from classes.evaluator import Evaluator
from classes.datasets_bank import DatasetsBank
from classes.element_seq_indexer import ElementSeqIndexer
from classes.utils import *

from models.tagger_birnn import TaggerBiRNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--fn_train', default='data/persuasive_essays/Paragraph_Level/train.dat.abs',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--fn_dev', default='data/persuasive_essays/Paragraph_Level/dev.dat.abs',
                        help='Dev data in CoNNL-2003 format, it is used to find best model during the training.')
    parser.add_argument('--fn_test', default='data/persuasive_essays/Paragraph_Level/test.dat.abs',
                        help='Test data in CoNNL-2003 format, it is used to obtain the final accuracy/F1 score.')
    parser.add_argument('--emb_fn', default='embeddings/glove.6B.100d.txt', help='Path to embeddings file.')
    parser.add_argument('--emb_delimiter', default=' ', help='Delimiter for embeddings file.')
    parser.add_argument('--freeze_embeddings', type=bool, default=False, help='False to continue training the embeddings.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, 0 by default, -1  means CPU.')
    parser.add_argument('--caseless', type=bool, default=True, help='Read characters caseless.')
    parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='Number hidden units in the recurrent layer.')
    parser.add_argument('--rnn_type', default='GRU', help='RNN cell units type: "Vanilla", "LSTM", "GRU".')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Clipping gradients maximum L2 norm.')
    parser.add_argument('--opt_method', default='sgd', help='Optimization method: "sgd", "adam".')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='Learning decay rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--verbose', type=bool, default=True, help='Show additional information.')
    parser.add_argument('--seed_num', type=int, default=42, help='Random seed number, but 42 is the best forever!')
    parser.add_argument('--checkpoint_fn', default=None, help='Path to save the trained model (best on DEV).')
    parser.add_argument('--report_fn', default='report_%s_%s.txt' % (datetime.datetime.now().hour,
                                                                     datetime.datetime.now().minute),
                                                                     help='Path to report.')
    parser.add_argument('--match_alpha_ratio', type=float, default='0.999',
                        help='Alpha ratio from non-strict matching, options: 0.999 or 0.5')
    args = parser.parse_args()

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    # Custom params here to replace the defaults
    #args.fn_train = 'data/NER/CoNNL_2003_shared_task/train.txt'
    #args.fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.txt'
    #args.fn_test = 'data/NER/CoNNL_2003_shared_task/test.txt'

    #args.fn_train = 'data/persuasive_essays/Essay_Level/train.dat.abs'
    #args.fn_dev = 'data/persuasive_essays/Essay_Level/dev.dat.abs'
    #args.fn_test = 'data/persuasive_essays/Essay_Level/test.dat.abs'

    #args.epoch_num = 2
    #args.lr_decay = 0.05
    #args.rnn_type = 'LSTM'
    #args.checkpoint_fn = 'tagger_model_es_par_GRU.bin'
    #args.report_fn = 'report_model_es_par_GRU.txt'

    # Load CoNNL data as sequences of strings of words and corresponding tags
    word_sequences_train, tag_sequences_train = read_CoNNL_dat_abs(args.fn_train)
    word_sequences_dev, tag_sequences_dev = read_CoNNL_dat_abs(args.fn_dev)
    word_sequences_test, tag_sequences_test = read_CoNNL_dat_abs(args.fn_test)

    # Converts lists of lists of words to integer indices and back
    word_seq_indexer = ElementSeqIndexer(gpu=args.gpu, caseless=args.caseless, load_embeddings=True, verbose=args.verbose)
    word_seq_indexer.load_vocabulary_from_embeddings_file(emb_fn=args.emb_fn, emb_delimiter=args.emb_delimiter)
    word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_train)
    word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_dev)
    word_seq_indexer.load_vocabulary_from_element_sequences(word_sequences_test, verbose=True)

    # Converts lists of lists of tags to integer indices and back
    tag_seq_indexer = ElementSeqIndexer(gpu=args.gpu, caseless=False, verbose=args.verbose)
    tag_seq_indexer.load_vocabulary_from_element_sequences(tag_sequences_train)

    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches from them
    datasets_bank = DatasetsBank()
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)

    tagger = TaggerBiRNN(word_seq_indexer=word_seq_indexer,
                         tag_seq_indexer=tag_seq_indexer,
                         class_num=tag_seq_indexer.get_elements_num(),
                         rnn_hidden_dim=args.rnn_hidden_dim,
                         freeze_embeddings=args.freeze_embeddings,
                         dropout_ratio=args.dropout_ratio,
                         rnn_type=args.rnn_type,
                         gpu=args.gpu)

    nll_loss = nn.NLLLoss(ignore_index=0) # "0" target values actually are zero-padded parts of sequences
    optimizer = optim.SGD(list(tagger.parameters()), lr=args.lr, momentum=args.momentum)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + args.lr_decay*epoch))
    iterations_num = int(datasets_bank.train_data_num / args.batch_size)
    best_f1_dev = -1
    for epoch in range(1, args.epoch_num + 1):
        tagger.train()
        if args.lr_decay > 0:
            scheduler.step()
        time_start = time.time()
        best_epoch_msg = ''
        word_sequences_train_batch_list, tag_sequences_train_batch_list = datasets_bank.get_train_batches(args.batch_size)
        for i, (word_sequences_train_batch, tag_sequences_train_batch) in enumerate(zip(word_sequences_train_batch_list, tag_sequences_train_batch_list)):
            tagger.zero_grad()
            outputs_tensor_train_batch_one_hot = tagger(word_sequences_train_batch)
            targets_tensor_train_batch = tag_seq_indexer.elements2tensor(tag_sequences_train_batch)
            loss = nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
            loss.backward()
            tagger.clip_gradients(args.clip_grad)
            optimizer.step()
            if i % 50 == 0 and args.verbose:
                print('-- epoch %d, i = %d/%d, loss = %1.4f' % (epoch, i, iterations_num, loss.item()))

        time_finish = time.time()
        outputs_tag_sequences_dev = tagger.predict_tags_from_words(datasets_bank.word_sequences_dev)
        acc_dev = Evaluator.get_accuracy_token_level(targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                     outputs_tag_sequences=outputs_tag_sequences_dev,
                                                     tag_seq_indexer=tag_seq_indexer)

        f1_dev, prec_dev, recall_dev, _ = Evaluator.get_f1_from_words(targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                                   outputs_tag_sequences=outputs_tag_sequences_dev,
                                                                   match_alpha_ratio=0.999)
        if f1_dev > best_f1_dev:
            best_epoch_msg = '[BEST] '
            best_epoch = epoch
            best_f1_dev = f1_dev
            best_tagger = tagger
        print('\n%sEPOCH %d/%d, DEV dataset: Accuracy = %1.2f, Precision = %1.2f, Recall = %1.2f, F1-100%% = %1.2f, %d sec.\n' %
                                                                                                (best_epoch_msg,
                                                                                                 epoch,
                                                                                                 args.epoch_num,
                                                                                                 acc_dev,
                                                                                                 f1_dev,
                                                                                                 prec_dev,
                                                                                                 recall_dev,
                                                                                                 time.time() - time_start))

    # After all epochs, perform the final evaluation on test dataset
    outputs_tag_sequences_test = tagger.predict_tags_from_words(datasets_bank.word_sequences_test)
    acc_test = Evaluator.get_accuracy_token_level(targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                  outputs_tag_sequences=outputs_tag_sequences_test,
                                                  tag_seq_indexer=tag_seq_indexer)
    f1_test, _, _, _ = Evaluator.get_f1_from_words(targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                outputs_tag_sequences=outputs_tag_sequences_test,
                                                match_alpha_ratio=0.999)

    print('Results on TEST (best on DEV, best epoch = %d): Accuracy = %1.2f, F1-100%% = %1.2f.\n' %  (best_epoch,
                                                                                                      acc_test,
                                                                                                      f1_test))

    print(Evaluator.get_f1_scores_extended(best_tagger, word_sequences_test, tag_sequences_test))

    # Write report
    if args.report_fn is not None:
        Evaluator.write_report(args.report_fn, args, best_tagger, word_sequences_test, tag_sequences_test)

    # Please, sequences_indexer objects are stored in the model
    if args.checkpoint_fn is not None:
        torch.save(best_tagger.cpu(), args.checkpoint_fn)

    print('The end!')
