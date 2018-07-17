from __future__ import print_function

import argparse
import time
import torch
from torch.optim.lr_scheduler import LambdaLR

from classes.datasets_bank import DatasetsBank
from classes.sequences_indexer import SequencesIndexer
from classes.utils import *
from classes.evaluator import Evaluator
from models.tagger_birnn import TaggerBiRNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagging problem using neural networks')
    parser.add_argument('--fn_train', default='data/argument_mining/persuasive_essays/es_paragraph_level_train.txt',
                        help='Train data in CoNNL-2003 format.')
    parser.add_argument('--fn_dev', default='data/argument_mining/persuasive_essays/es_paragraph_level_dev.txt',
                        help='Dev data in CoNNL-2003 format, it is used to find best model during the training.')
    parser.add_argument('--fn_test', default='data/argument_mining/persuasive_essays/es_paragraph_level_test.txt',
                        help='Test data in CoNNL-2003 format, it is used to obtain the final accuracy/F1 score.')
    parser.add_argument('--emb_fn', default='embeddings/glove.6B.100d.txt', help='Path to embeddings file.')
    parser.add_argument('--emb_delimiter', default=' ', help='Delimiter for embeddings file.')
    parser.add_argument('--freeze_embeddings', type=bool, default=False, help='To continue training the embedding or not.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device number, -1 by default (CPU).')
    parser.add_argument('--caseless', type=bool, default=True, help='Read tokens caseless.')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--rnn_hidden_size', type=int, default=101, help='Number hidden units in the recurrent layer.')
    parser.add_argument('--rnn_type', default='GRU', help='RNN cell units type: "Vanilla", "LSTM", "GRU".')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Clipping gradients maximum L2 norm.')
    parser.add_argument('--opt_method', default='sgd', help='Optimization method: "sgd", "adam".')
    parser.add_argument('--lr', type=float, default=0.015, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.0, help='Learning decay rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--verbose', type=bool, default=True, help='Show additional information.')
    parser.add_argument('--seed_num', type=int, default=42, help='Random seed number, but 42 is the best forever!')
    parser.add_argument('--save_best_path', default=None, help='Path to save the trained model (best on DEV).')

    args = parser.parse_args()

    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    # Add custom params here to replace the defaults, if you want
    #args.fn_train = 'data/NER/CoNNL_2003_shared_task/train.txt'
    #args.fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.txt'
    #args.fn_test = 'data/NER/CoNNL_2003_shared_task/test.txt'
    args.gpu = 0
    args.epoch_num = 4
    args.lr_decay = 0
    #args.rnn_type = 'LSTM'
    args.save_best_path = 'tagger_model.txt'

    # Load CoNNL data as sequences of strings of tokens and corresponding tags
    token_sequences_train, tag_sequences_train = read_CoNNL(args.fn_train)
    token_sequences_dev, tag_sequences_dev = read_CoNNL(args.fn_dev)
    token_sequences_test, tag_sequences_test = read_CoNNL(args.fn_test)

    # SequenceIndexer is a class to convert tokens and tags as strings to integer indices and back
    sequences_indexer = SequencesIndexer(caseless=args.caseless, verbose=args.verbose, gpu=args.gpu)
    sequences_indexer.load_embeddings(emb_fn=args.emb_fn, emb_delimiter=args.emb_delimiter)
    sequences_indexer.add_token_sequences(token_sequences_train, verbose=False)
    sequences_indexer.add_token_sequences(token_sequences_dev, verbose=False)
    sequences_indexer.add_token_sequences(token_sequences_test, verbose=True)
    sequences_indexer.add_tag_sequences(tag_sequences_train) # Surely, all necessarily tags must be into train data

    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches from them
    datasets_bank = DatasetsBank(sequences_indexer)
    datasets_bank.add_train_sequences(token_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(token_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(token_sequences_test, tag_sequences_test)

    evaluator = Evaluator(sequences_indexer)

    tagger = TaggerBiRNN(sequences_indexer=sequences_indexer,
                         class_num=sequences_indexer.get_tags_num(),
                         rnn_hidden_size=args.rnn_hidden_size,
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
        if args.lr_decay > 0:
            scheduler.step()
        time_start = time.time()
        best_epoch_msg = ''
        for i in range(iterations_num + 1):
            tagger.train()
            tagger.zero_grad()
            inputs_tensor_train_batch, targets_tensor_train_batch = datasets_bank.get_train_batch(args.batch_size)
            outputs_train_batch = tagger(inputs_tensor_train_batch)
            loss = nll_loss(outputs_train_batch, targets_tensor_train_batch)
            loss.backward()
            tagger.clip_gradients(args.clip_grad)
            optimizer.step()
            if i % 100 == 0 and args.verbose:
                print('-- epoch %d, i = %d/%d, loss = %1.4f' % (epoch, i, iterations_num, loss.item()))
        time_finish = time.time()
        f1_dev, precision_dev, recall_dev = evaluator.get_macro_scores(tagger=tagger,
                                                                       inputs=datasets_bank.inputs_tensor_dev,
                                                                       targets=datasets_bank.targets_tensor_dev)
        if f1_dev > best_f1_dev:
            best_epoch_msg = '[BEST] '
            best_epoch = epoch
            best_f1_dev = f1_dev
            best_tagger = tagger
        print('\n%sEPOCH %d/%d, DEV: F1 = %1.3f, Precision = %1.3f, Recall = %1.3f, %d sec.\n' % (best_epoch_msg,
                                                                                                  epoch,
                                                                                                  args.epoch_num,
                                                                                                  f1_dev,
                                                                                                  precision_dev,
                                                                                                  recall_dev,
                                                                                                  time.time() - time_start))


    f1_test, precision_test, recall_test = evaluator.get_macro_scores(tagger=best_tagger,
                                                                      inputs=datasets_bank.inputs_tensor_test,
                                                                      targets=datasets_bank.targets_tensor_test)

    print('Results on TEST (for best on DEV tagger, best epoch = %d): F1 = %1.3f, Precision = %1.3f, Recall = %1.3f.\n' % (best_epoch,
                                                                                                                f1_test,
                                                                                                                precision_test,
                                                                                                                recall_test))
    # Please, note that sequences indexer is stored in the "sequences_indexer" field of best_tagger object
    if args.save_best_path is not None:
        torch.save(best_tagger.cpu(), args.save_best_path)

    print('The end!')
