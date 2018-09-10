# BiLSTM-CNN-CRF-tagger

BiLSTM-CNN-CRF-tagger is a Pytorch-based implementation of "mainstream" neural tagging scheme from [Lample, 
et. al., 2016](https://arxiv.org/pdf/1603.01360.pdf). The evaluation of f1 score is provided using  standard Perl 
script for CoNNL-2003 shared task by Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26. 

The results on Named Enitity Recognition CoNNL-2003 shared task with default settings: 

|         Model       |     f1-score  |
| ------------------- | ------------- |
| Bi-GRU + CNN + CRF  |      90.20    |

## Requirements

- python 3.6
- [pytorch 0.4.1](http://pytorch.org/)
- numpy 1.15.1
- scipy 1.1.0
- scikit-learn 0.19.2

## Usage

### Train/test

```
usage: main.py [-h] [--model MODEL] [--fn_train FN_TRAIN] [--fn_dev FN_DEV]
               [--fn_test FN_TEST] [--emb_fn EMB_FN] [--emb_dim EMB_DIM]
               [--emb_delimiter EMB_DELIMITER]
               [--freeze_word_embeddings FREEZE_WORD_EMBEDDINGS]
               [--freeze_char_embeddings FREEZE_CHAR_EMBEDDINGS] [--gpu GPU]
               [--check_for_lowercase CHECK_FOR_LOWERCASE]
               [--epoch_num EPOCH_NUM] [--min_epoch_num MIN_EPOCH_NUM]
               [--rnn_hidden_dim RNN_HIDDEN_DIM] [--rnn_type RNN_TYPE]
               [--char_embeddings_dim CHAR_EMBEDDINGS_DIM]
               [--word_len WORD_LEN]
               [--char_cnn_filter_num CHAR_CNN_FILTER_NUM]
               [--char_window_size CHAR_WINDOW_SIZE]
               [--dropout_ratio DROPOUT_RATIO] [--clip_grad CLIP_GRAD]
               [--opt_method OPT_METHOD] [--lr LR] [--lr_decay LR_DECAY]
               [--momentum MOMENTUM] [--batch_size BATCH_SIZE]
               [--verbose VERBOSE] [--seed_num SEED_NUM]
               [--checkpoint_fn CHECKPOINT_FN]
               [--match_alpha_ratio MATCH_ALPHA_RATIO] [--patience PATIENCE]
               [--word_seq_indexer_path WORD_SEQ_INDEXER_PATH]

Learning tagging problem using neural networks

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Tagger model: "BiRNN", "BiRNNCNN", "BiRNNCRF",
                        "BiRNNCNNCRF".
  --fn_train FN_TRAIN   Train data in CoNNL-2003 format.
  --fn_dev FN_DEV       Dev data in CoNNL-2003 format, it is used to find best
                        model during the training.
  --fn_test FN_TEST     Test data in CoNNL-2003 format, it is used to obtain
                        the final accuracy/F1 score.
  --emb_fn EMB_FN       Path to word embeddings file.
  --emb_dim EMB_DIM     Dimension of word embeddings file.
  --emb_delimiter EMB_DELIMITER
                        Delimiter for word embeddings file.
  --freeze_word_embeddings FREEZE_WORD_EMBEDDINGS
                        False to continue training the \ word embeddings.
  --freeze_char_embeddings FREEZE_CHAR_EMBEDDINGS
                        False to continue training the char embeddings.
  --gpu GPU             GPU device number, 0 by default, -1 means CPU.
  --check_for_lowercase CHECK_FOR_LOWERCASE
                        Read characters caseless.
  --epoch_num EPOCH_NUM
                        Number of epochs.
  --min_epoch_num MIN_EPOCH_NUM
                        Minimum number of epochs.
  --rnn_hidden_dim RNN_HIDDEN_DIM
                        Number hidden units in the recurrent layer.
  --rnn_type RNN_TYPE   RNN cell units type: "Vanilla", "LSTM", "GRU".
  --char_embeddings_dim CHAR_EMBEDDINGS_DIM
                        Char embeddings dim, only for char CNNs.
  --word_len WORD_LEN   Max length of words in characters for char CNNs.
  --char_cnn_filter_num CHAR_CNN_FILTER_NUM
                        Number of filters in Char CNN.
  --char_window_size CHAR_WINDOW_SIZE
                        Convolution1D size.
  --dropout_ratio DROPOUT_RATIO
                        Dropout ratio.
  --clip_grad CLIP_GRAD
                        Clipping gradients maximum L2 norm.
  --opt_method OPT_METHOD
                        Optimization method: "sgd", "adam".
  --lr LR               Learning rate.
  --lr_decay LR_DECAY   Learning decay rate.
  --momentum MOMENTUM   Learning momentum rate.
  --batch_size BATCH_SIZE
                        Batch size, samples.
  --verbose VERBOSE     Show additional information.
  --seed_num SEED_NUM   Random seed number, but 42 is the best forever!
  --checkpoint_fn CHECKPOINT_FN
                        Path to save the trained model.
  --match_alpha_ratio MATCH_ALPHA_RATIO
                        Alpha ratio from non-strict matching, options: 0.999
                        or 0.5
  --patience PATIENCE   Patience for early stopping.
  --word_seq_indexer_path WORD_SEQ_INDEXER_PATH
                        Load word_seq_indexer object from hdf5 file.```
```

### Run pretrained model

```
usage: run_tagger_example.py [-h] [--fn FN] [--checkpoint_fn CHECKPOINT_FN]
                             [--gpu GPU]

Run trained tagger from the checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --fn FN               Train data in CoNNL-2003 format.
  --checkpoint_fn CHECKPOINT_FN
                        Path to load the trained model.
  --gpu GPU             GPU device number, 0 by default, -1 means CPU.
```