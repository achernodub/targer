# BiLSTM-CNN-CRF tagger

BiLSTM-CNN-CRF tagger is a PyTorch implementation of "mainstream" neural tagging scheme based on works of [Lample, 
et. al., 2016](https://arxiv.org/pdf/1603.01360.pdf) and [Ma et. al., 2016](https://arxiv.org/pdf/1603.01354.pdf). 

## Requirements

- python 3.6
- [pytorch 0.4.1](http://pytorch.org/)
- numpy 1.15.1
- scipy 1.1.0
- scikit-learn 0.19.2

 
## Project structure

```
|__ articles/ --> collection of papers related to the tagging, argument mining, etc. 
|__ classes/
        |__ data_io.py --> class for reading/writing data in different CoNNL file formats
        |__ datasets_bank.py --> class for storing the train/dev/test data subsets and sampling batches 
                                 from the train dataset 
        dataset
        |__ evaluator.py --> class for evaluation of F1 scores and token-level accuracies
        |__ report.py --> class for storing the evaluation results as text files
        |__ tag_components.py --> class for extracting tag components from BOI encodings 
        |__ utils.py --> several auxiliary utils and functions
|__ data/
        |__ NER/ --> Datasets for Named Entity Recognition 
            |__ CoNNL_2003_shared_task/ --> data for NER CoNLL-2003 shared task (English) in BOI-2 
                                            CoNNL format, from E.F. Tjong Kim Sang and F. De Meulder, 
                                            Introduction to the CoNLL-2003 Shared Task:  
                                            Language-Independent Named Entity Recognition, 2003. 
        |__ AM/ --> Datasets for Argument Mining
            |__ persuasive_essays/ --> data for persuasive essays in BOI-2-like CoNNL format, from: 
                                       Steffen Eger, Johannes Daxenberger, Iryna Gurevych. Neural 
                                       End-to-End  Learning for Computational Argumentation Mining, 2017
|__ embeddings/
        |__ get_glove_embeddings.sh --> script for downloading GloVe6B 100-dimensional word embeddings
|__ layers/
        |__ layer_base.py --> abstract base class for all types of layers
        |__ layer_birnn_base.py --> abstract base class for all bidirectional recurrent layers
        |__ layer_word_embeddings.py --> class implements word embeddings
        |__ layer_char_embeddings.py --> class implements character-level embeddings
        |__ layer_char_cnn.py --> class implements character-level convolutional 1D operation
        |__ layer_bilstm.py --> class implements bidirectional LSTM recurrent layer
        |__ layer_bigru.py --> class implements bidirectional GRU recurrent layer
        |__ layer_crf.py --> class implements conditional random field (CRF) 
|__ models/
        |__ tagger_base.py --> abstract base class for all types of taggers
        |__ tagger_io.py --> contains wrappers to create and load tagger models
        |__ tagger_birnn.py --> vanilla BiLSTM/BiGRU tagger model
        |__ tagger_birnn_crf.py --> BiLSTM/BiGRU + CRF tagger model
        |__ tagger_birnn_cnn.py --> BiLSTM/BiGRU + char-level CNN tagger model
        |__ tagger_birnn_cnn_crf.py --> BiLSTM/BiGRU + char-level CNN  + CRF tagger model
|__ pretrained/
        |__ tagger_NER.hdf5 --> tagger for NER, BiGRU+CNN+CRF trained on NER-2003 shared task, English 
|__ seq_indexers/
        |__ seq_indexer_base.py --> abstract class for sequence indexers, they converts list of lists 
                                    of string items
    to the list of lists of integer indices and back
        |__ seq_indexer_base_embeddings.py --> abstract sequence indexer class that implements work 
                                               with embeddings 
        |__ seq_indexer_word.py --> converts list of lists of words as strings to list of lists of 
                                    integer indices and back, has built-in embeddings
        |__ seq_indexer_char.py --> converts list of lists of characters to list of lists of integer 
                                    indices and back, has built-in embeddings 
        |__ seq_indexer_tag.py --> converts list of lists of string tags to list of lists of integer 
                                    indices and back, doesn't have built-in embeddings 
|__ main.py --> main script for training/evaluation/saving tagger models
|__ run_tagger.py --> run the trained tagger model from the checkpoint file
|__ conlleval --> "official" Perl script from NER 2003 shared task for evaluating the f1 scores, 
                   author: Erik Tjong Kim Sang, version: 2004-01-26
|__ requirements.txt --> file for managing packages requirements    
```

## Evaluation

Results of training the models with the default settings: 

|         tagger model       |     dataset           | micro-f1 on test        |
| ------------------- | --------------------- | ----------------------- |
| BiLSTM + CNN + CRF [Lample et. al., 2016](https://arxiv.org/pdf/1603.01360.pdf) | NER-2003 shared task (English)  | 90.94 |
| BiLSTM + CNN + CRF [Ma et al., 2016](https://arxiv.org/pdf/1603.01354.pdf)      | NER-2003 shared task (English)  | 91.21 |
| BiLSTM + CNN + CRF  (our)   | NER-2003 shared task (English)                     | 90.86  |          |
||||           
| STag_BLCC, [Eger et. al., 2017](https://arxiv.org/pdf/1704.06104.pdf)   | AM Persuasive Essays, Paragraph Level                     | 66.69  |          |
| LSTM-ER, [Eger et. al., 2017](https://arxiv.org/pdf/1704.06104.pdf)   | AM Persuasive Essays, Paragraph Level                     | 70.83  |          |
| BiGRU + CNN + CRF  (our)   | AM Persuasive Essays, Paragraph Level                     | 64.31  |          |

In order to ensure the consistency of the experiments, for evaluation purposes we use "official" Perl script from NER 2003 shared task, author: Erik Tjong Kim Sang, version: 2004-01-26, example of it's output:

```
processed 46435 tokens with 5648 phrases; found: 5679 phrases; correct: 5146.
accuracy:  97.92%; precision:  90.61%; recall:  91.11%; FB1:  90.86
              LOC: precision:  91.35%; recall:  93.65%; FB1:  92.48  1710
             MISC: precision:  78.20%; recall:  82.76%; FB1:  80.42  743
              ORG: precision:  90.25%; recall:  88.02%; FB1:  89.12  1620
              PER: precision:  95.95%; recall:  95.30%; FB1:  95.63  1606
``` 

## Usage

### Train/test

To train/evaluate/save trained tagger model, please run the `main.py` script.

```
usage: main.py [-h] [--seed_num SEED_NUM] [--model MODEL]
               [--fn_train FN_TRAIN] [--fn_dev FN_DEV] [--fn_test FN_TEST]
               [--load LOAD] [--save SAVE] [--wsi WSI] [--emb_fn EMB_FN]
               [--emb_dim EMB_DIM] [--emb_delimiter EMB_DELIMITER]
               [--freeze_word_embeddings FREEZE_WORD_EMBEDDINGS]
               [--freeze_char_embeddings FREEZE_CHAR_EMBEDDINGS] [--gpu GPU]
               [--check_for_lowercase CHECK_FOR_LOWERCASE]
               [--epoch_num EPOCH_NUM] [--min_epoch_num MIN_EPOCH_NUM]
               [--patience PATIENCE] [--rnn_type RNN_TYPE]
               [--rnn_hidden_dim RNN_HIDDEN_DIM]
               [--char_embeddings_dim CHAR_EMBEDDINGS_DIM]
               [--word_len WORD_LEN]
               [--char_cnn_filter_num CHAR_CNN_FILTER_NUM]
               [--char_window_size CHAR_WINDOW_SIZE]
               [--dropout_ratio DROPOUT_RATIO] [--dataset_sort DATASET_SORT]
               [--clip_grad CLIP_GRAD] [--opt_method OPT_METHOD]
               [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY]
               [--momentum MOMENTUM] [--verbose VERBOSE]
               [--match_alpha_ratio MATCH_ALPHA_RATIO] [--save_best SAVE_BEST]
               [--report_fn REPORT_FN]

Learning tagging problem using neural networks

optional arguments:
  -h, --help            show this help message and exit
  --seed_num SEED_NUM   Random seed number, you may use any but 42 is the
                        answer.
  --model MODEL         Tagger model: "BiRNN", "BiRNNCNN", "BiRNNCRF",
                        "BiRNNCNNCRF".
  --fn_train FN_TRAIN   Train data in CoNNL-2003 format.
  --fn_dev FN_DEV       Dev data in CoNNL-2003 format, it is used to find best
                        model during the training.
  --fn_test FN_TEST     Test data in CoNNL-2003 format, it is used to obtain
                        the final accuracy/F1 score.
  --load LOAD           Path to load from the trained model.
  --save SAVE           Path to save the trained model.
  --wsi WSI             Load word_seq_indexer object from hdf5 file.
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
  --patience PATIENCE   Patience for early stopping.
  --rnn_type RNN_TYPE   RNN cell units type: "Vanilla", "LSTM", "GRU".
  --rnn_hidden_dim RNN_HIDDEN_DIM
                        Number hidden units in the recurrent layer.
  --char_embeddings_dim CHAR_EMBEDDINGS_DIM
                        Char embeddings dim, only for char CNNs.
  --word_len WORD_LEN   Max length of words in characters for char CNNs.
  --char_cnn_filter_num CHAR_CNN_FILTER_NUM
                        Number of filters in Char CNN.
  --char_window_size CHAR_WINDOW_SIZE
                        Convolution1D size.
  --dropout_ratio DROPOUT_RATIO
                        Dropout ratio.
  --dataset_sort DATASET_SORT
                        Sort sequences by length for training.
  --clip_grad CLIP_GRAD
                        Clipping gradients maximum L2 norm.
  --opt_method OPT_METHOD
                        Optimization method: "sgd", "adam".
  --batch_size BATCH_SIZE
                        Batch size, samples.
  --lr LR               Learning rate.
  --lr_decay LR_DECAY   Learning decay rate.
  --momentum MOMENTUM   Learning momentum rate.
  --verbose VERBOSE     Show additional information.
  --match_alpha_ratio MATCH_ALPHA_RATIO
                        Alpha ratio from non-strict matching, options: 0.999
                        or 0.5
  --save_best SAVE_BEST
                        Save best on dev model as a final model.
  --report_fn REPORT_FN
                        Report filename.
```

### Run trained model

```
usage: run_tagger.py [-h] [--fn FN] [--checkpoint_fn CHECKPOINT_FN]
                             [--gpu GPU]

Run trained tagger from the checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --fn FN               Train data in CoNNL-2003 format.
  --checkpoint_fn CHECKPOINT_FN
                        Path to load the trained model.
  --gpu GPU             GPU device number, 0 by default, -1 means CPU.
```

### Example of output report

```
     epoch  | train loss |   f1-train |     f1-dev |    f1-test | acc. train |   acc. dev |  acc. test 
---------------------------------------------------------------------------------------------------------
          0 |       0.00 |       1.33 |       0.94 |       1.48 |      56.07 |      54.38 |      50.89 
          1 |     294.13 |      86.57 |      85.17 |      83.04 |      97.27 |      97.00 |      96.37 
          2 |     126.18 |      91.03 |      89.76 |      86.06 |      98.11 |      97.67 |      96.76 
          3 |      92.71 |      92.20 |      90.59 |      87.39 |      98.49 |      98.13 |      97.30 
          4 |      77.60 |      94.20 |      91.69 |      88.98 |      98.83 |      98.27 |      97.54 
          5 |      65.07 |      94.21 |      91.62 |      88.55 |      98.81 |      98.21 |      97.44 
          6 |      58.39 |      95.56 |      92.79 |      89.74 |      99.15 |      98.57 |      97.76 
          7 |      52.52 |      95.49 |      91.72 |      88.60 |      99.13 |      98.31 |      97.49 
          8 |      47.91 |      96.27 |      92.16 |      89.19 |      99.26 |      98.41 |      97.58 
          9 |      43.49 |      96.53 |      92.53 |      89.08 |      99.34 |      98.51 |      97.53 
         10 |      40.86 |      96.78 |      92.12 |      88.43 |      99.40 |      98.38 |      97.39 
         11 |      38.78 |      97.16 |      92.80 |      88.70 |      99.47 |      98.56 |      97.62 
         12 |      34.99 |      97.47 |      93.40 |      89.75 |      99.54 |      98.70 |      97.78 
         13 |      34.16 |      97.64 |      93.83 |      90.02 |      99.56 |      98.78 |      97.82 
         14 |      31.93 |      97.42 |      92.69 |      89.04 |      99.52 |      98.51 |      97.57 
         15 |      31.01 |      97.91 |      93.67 |      90.05 |      99.62 |      98.74 |      97.78 
         16 |      28.60 |      98.15 |      93.57 |      90.26 |      99.68 |      98.69 |      97.82 
         17 |      26.91 |      98.27 |      93.89 |      90.02 |      99.69 |      98.78 |      97.84 
         18 |      25.46 |      98.45 |      93.88 |      90.23 |      99.73 |      98.77 |      97.87 
         19 |      24.99 |      98.47 |      93.74 |      90.04 |      99.73 |      98.73 |      97.82 
         20 |      23.91 |      98.53 |      93.92 |      89.70 |      99.74 |      98.76 |      97.75 
         21 |      22.34 |      98.72 |      94.14 |      89.74 |      99.78 |      98.83 |      97.77 
         22 |      21.12 |      98.63 |      93.43 |      89.71 |      99.77 |      98.67 |      97.72 
         23 |      21.58 |      98.78 |      93.41 |      89.89 |      99.79 |      98.71 |      97.70 
         24 |      20.16 |      98.95 |      93.85 |      90.57 |      99.83 |      98.79 |      97.94 
         25 |      20.03 |      98.78 |      93.45 |      89.54 |      99.78 |      98.68 |      97.58 
         26 |      19.08 |      99.03 |      94.48 |      90.33 |      99.84 |      98.90 |      97.89 
         27 |      18.28 |      99.12 |      93.77 |      89.83 |      99.85 |      98.77 |      97.73 
         28 |      17.80 |      99.03 |      94.10 |      90.23 |      99.84 |      98.83 |      97.87 
         29 |      17.22 |      99.08 |      94.03 |      90.33 |      99.84 |      98.79 |      97.89 
         30 |      16.06 |      99.07 |      93.95 |      90.16 |      99.85 |      98.79 |      97.83 
         31 |      16.27 |      99.11 |      93.71 |      90.14 |      99.85 |      98.70 |      97.75 
         32 |      15.60 |      99.15 |      93.27 |      89.35 |      99.85 |      98.65 |      97.61 
         33 |      15.21 |      99.30 |      94.23 |      90.26 |      99.89 |      98.84 |      97.84 
         34 |      15.21 |      99.25 |      93.70 |      90.00 |      99.88 |      98.75 |      97.86 
         35 |      14.39 |      99.37 |      94.13 |      90.20 |      99.90 |      98.83 |      97.86 
         36 |      13.72 |      99.32 |      93.99 |      90.78 |      99.89 |      98.79 |      98.00 
         37 |      13.44 |      99.34 |      93.85 |      90.32 |      99.89 |      98.79 |      97.82 
         38 |      13.74 |      99.38 |      94.15 |      90.52 |      99.91 |      98.80 |      97.95 
         39 |      13.35 |      99.43 |      93.95 |      90.27 |      99.91 |      98.78 |      97.87 
         40 |      13.13 |      99.52 |      94.27 |      90.77 |      99.93 |      98.84 |      98.00 
         41 |      12.08 |      99.51 |      94.30 |      90.65 |      99.93 |      98.84 |      97.98 
         42 |      12.63 |      99.54 |      94.33 |      90.43 |      99.93 |      98.86 |      97.90 
         43 |      11.72 |      99.52 |      94.17 |      90.16 |      99.92 |      98.84 |      97.83 
         44 |      11.75 |      99.54 |      94.26 |      90.58 |      99.93 |      98.81 |      97.94 
         45 |      10.90 |      99.63 |      94.40 |      90.59 |      99.94 |      98.86 |      97.90 
         46 |      11.29 |      99.58 |      94.18 |      90.38 |      99.93 |      98.80 |      97.91 
         47 |      10.79 |      99.52 |      94.09 |      90.22 |      99.93 |      98.80 |      97.86 
         48 |      10.89 |      99.70 |      94.60 |      90.88 |      99.96 |      98.87 |      97.99 
         49 |      10.85 |      99.64 |      94.19 |      90.66 |      99.94 |      98.80 |      97.92 
         50 |       9.91 |      99.63 |      93.99 |      90.14 |      99.94 |      98.77 |      97.81 
         51 |       9.65 |      99.67 |      94.21 |      90.34 |      99.95 |      98.82 |      97.87 
         52 |      10.12 |      99.64 |      94.11 |      90.39 |      99.95 |      98.81 |      97.90 
         53 |       9.48 |      99.63 |      94.00 |      89.97 |      99.94 |      98.78 |      97.77 
         54 |       9.90 |      99.64 |      94.24 |      90.53 |      99.95 |      98.83 |      97.94 
         55 |       9.35 |      99.60 |      93.67 |      89.71 |      99.94 |      98.72 |      97.67 
         56 |       9.32 |      99.70 |      93.84 |      89.48 |      99.95 |      98.75 |      97.64 
         57 |       9.37 |      99.71 |      94.07 |      90.15 |      99.96 |      98.81 |      97.84 
         58 |       9.42 |      99.67 |      94.19 |      90.02 |      99.95 |      98.82 |      97.77 
         59 |       9.20 |      99.78 |      94.27 |      90.00 |      99.97 |      98.82 |      97.80 
         60 |       9.37 |      99.71 |      94.21 |      90.24 |      99.96 |      98.83 |      97.85 
         61 |       8.95 |      99.73 |      94.34 |      90.29 |      99.96 |      98.85 |      97.89 
         62 |       8.51 |      99.73 |      94.35 |      90.13 |      99.96 |      98.83 |      97.85 
         63 |       8.74 |      99.70 |      94.09 |      89.93 |      99.96 |      98.79 |      97.77 
         64 |       8.19 |      99.69 |      94.00 |      89.98 |      99.95 |      98.79 |      97.80 
         65 |       8.20 |      99.72 |      93.73 |      89.54 |      99.96 |      98.71 |      97.68 
         66 |       8.02 |      99.73 |      93.88 |      89.64 |      99.96 |      98.75 |      97.73 
         67 |       8.04 |      99.70 |      93.95 |      89.91 |      99.96 |      98.78 |      97.78 
         68 |       8.67 |      99.77 |      94.19 |      89.96 |      99.97 |      98.81 |      97.81 
         69 |       8.97 |      99.79 |      94.16 |      89.94 |      99.97 |      98.81 |      97.75 
         70 |       7.35 |      99.78 |      94.20 |      90.08 |      99.97 |      98.82 |      97.79 
         71 |       7.64 |      99.74 |      93.97 |      89.96 |      99.96 |      98.80 |      97.71 
         72 |       7.72 |      99.76 |      93.93 |      89.68 |      99.97 |      98.77 |      97.67 
         73 |       7.09 |      99.75 |      93.76 |      89.60 |      99.96 |      98.75 |      97.66 
         74 |       7.15 |      99.81 |      93.69 |      89.92 |      99.97 |      98.75 |      97.78 
         75 |       7.83 |      99.79 |      93.83 |      89.80 |      99.97 |      98.75 |      97.72 
         76 |       7.15 |      99.80 |      93.66 |      89.79 |      99.97 |      98.74 |      97.77 
         77 |       7.11 |      99.81 |      93.63 |      89.77 |      99.97 |      98.74 |      97.77 
         78 |       7.34 |      99.79 |      93.78 |      89.96 |      99.97 |      98.75 |      97.78 
         79 |       7.34 |      99.80 |      93.95 |      90.25 |      99.97 |      98.79 |      97.88 
---------------------------------------------------------------------------------------------------------
Final eval on test, "save best", best epoch on dev 48, micro-f1 test = 90.88
```

### Alternative neural taggers

- NeuroNER (Tensorflow) [https://github.com/Franck-Dernoncourt/NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER)
- LM-LSTM-CRF (Pytorch) [https://github.com/LiyuanLucasLiu/LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)
- LD-Net (Pytorch) [https://github.com/LiyuanLucasLiu/LD-Net](https://github.com/LiyuanLucasLiu/LD-Net)
- UKPLab/emnlp2017-bilstm-cnn-crf (Tensorflow & Keras)
[https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf)
- UKPLab/elmo-bilstm-cnn-crf (Tensorflow & Keras)
[https://github.com/UKPLab/elmo-bilstm-cnn-crf](https://github.com/UKPLab/elmo-bilstm-cnn-crf)
