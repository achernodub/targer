# BiLSTM-CNN-CRF tagger

BiLSTM-CNN-CRF tagger is a PyTorch implementation of "mainstream" neural tagging scheme based on works of [Lample, 
et. al., 2016](https://arxiv.org/pdf/1603.01360.pdf) and [Ma et. al., 2016](https://arxiv.org/pdf/1603.01354.pdf).

<p align="center"><img width="100%" src="docs/scheme.png"/></p> 

## Requirements

- numpy 1.15.1
- scipy 1.1.0
- python 3.5.2 or higher
- [pytorch 0.4.1](http://pytorch.org/)

## Benefits

- native PyTorch implementation;
- vectorized code for training on batches;
- easy adding new classes for custom data file formats and evaluation metrics;
- trustworthy evaluation of f1-score.
 
## Project structure

```
|__ articles/ --> collection of papers related to the tagging, argument mining, etc.
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
|__ docs/ --> documentation
|__ embeddings
        |__ get_glove_embeddings.sh --> script for downloading GloVe6B 100-dimensional word embeddings
        |__ get_fasttext_embeddings.sh --> script for downloading Fasttext word embeddings
|__ pretrained/
        |__ tagger_NER.hdf5 --> tagger for NER, BiLSTM+CNN+CRF trained on NER-2003 shared task, English
src/
|__utils/
   |__generate_tree_description.py --> import os
   |__generate_ft_emb.py --> generate predefined FastText embeddings for dataset
|__models/
   |__tagger_base.py --> abstract base class for all types of taggers
   |__tagger_birnn.py --> Vanilla recurrent network model for sequences tagging.
   |__tagger_birnn_crf.py --> BiLSTM/BiGRU + CRF tagger model
   |__tagger_birnn_cnn.py --> BiLSTM/BiGRU + char-level CNN tagger model
   |__tagger_birnn_cnn_crf.py --> BiLSTM/BiGRU + char-level CNN  + CRF tagger model   
|__data_io/   
   |__data_io_connl_ner_2003.py --> input/output data wrapper for CoNNL file format used in  NER-2003 Shared Task dataset
   |__data_io_connl_pe.py --> input/output data wrapper for CoNNL file format used in Persuassive Essays dataset
   |__data_io_connl_wd.py --> input/output data wrapper for CoNNL file format used in Web Discourse dataset
|__factories/
   |__factory_datasets_bank.py --> creates various datasets banks
   |__factory_data_io.py --> creates various data readers/writers
   |__factory_evaluator.py --> creates various evaluators
   |__factory_optimizer.py --> creates various optimizers
   |__factory_tagger.py --> creates various tagger models   
|__layers/
   |__layer_base.py --> abstract base class for all type of layers   
   |__layer_word_embeddings.py --> class implements word embeddings
   |__layer_char_cnn.py --> class implements character-level convolutional 1D layer
   |__layer_char_embeddings.py --> class implements character-level embeddings
   |__layer_birnn_base.py --> abstract base class for all bidirectional recurrent layers
   |__layer_bivanilla.py --> class implements standard bidirectional Vanilla recurrent layer      
   |__layer_bilstm.py --> class implements standard bidirectional LSTM recurrent layer
   |__layer_bigru.py --> class implements standard bidirectional GRU recurrent layer
   |__layer_crf.py --> class implements Conditional Random Fields (CRF)   
|__evaluators/
   |__evaluator_base.py --> abstract base class for all evaluators
   |__evaluator_acc_token_level.py --> token-level accuracy evaluator for each class of BOI-like tags
   |__evaluator_f1_macro_token_level.py --> macro-F1 scores evaluator for each class of BOI-like tags
   |__evaluator_f1_micro_spans_connl.py --> f1-micro averaging evaluator for tag components, spans detection + classification, uses standard CoNNL perl script
   |__evaluator_f1_micro_spans_alpha_match_base.py --> abstract base class for f1-micro averaging evaluation for tag components, spans detection + classification
   |__evaluator_f1_micro_spans_alpha_match_05.py --> f1-micro averaging evaluation for tag components, spans detection + classification, alpha = 0.5   
   |__evaluator_f1_micro_spans_alpha_match_10.py --> f1-micro averaging evaluation for tag components, spans detection + classification, alpha = 1.0 (strict)   
|__seq_indexers/
   |__seq_indexer_base.py --> base abstract class for sequence indexers
   |__seq_indexer_base_embeddings.py --> abstract sequence indexer class that implements work  with embeddings
   |__seq_indexer_word.py --> converts list of lists of words as strings to list of lists of integer indices and back
   |__seq_indexer_char.py --> converts list of lists of characters to list of lists of integer indices and back
   |__seq_indexer_tag.py --> converts list of lists of string tags to list of lists of integer indices and back
|__classes/
   |__datasets_bank.py --> provides storing the train/dev/test data subsets and sampling batches from the train dataset
   |__report.py --> stores evaluation results during the training process as text files
   |__utils.py --> several auxiliary functions
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
| BiLSTM + CNN + CRF  (our)   | NER-2003 shared task (English)                     | 90.42  |          |
||||           
| STag_BLCC, [Eger et. al., 2017](https://arxiv.org/pdf/1704.06104.pdf)   | AM Persuasive Essays, Paragraph Level                     | 64.74 +/- 1.97  |          |
| BiGRU + CNN + CRF  (our)   | AM Persuasive Essays, Paragraph Level                     | 62.82  |          |

In order to ensure the consistency of the experiments, for evaluation purposes we use "official" Perl script from NER 2003 shared task, author: Erik Tjong Kim Sang, version: 2004-01-26, example of it's output:

```
Standard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):
processed 46435 tokens with 5648 phrases; found: 5622 phrases; correct: 5105.
accuracy:  97.92%; precision:  90.80%; recall:  90.39%; FB1:  90.59
              LOC: precision:  93.06%; recall:  91.67%; FB1:  92.36  1643
             MISC: precision:  78.75%; recall:  80.77%; FB1:  79.75  720
              ORG: precision:  88.57%; recall:  88.62%; FB1:  88.59  1662
              PER: precision:  96.24%; recall:  95.05%; FB1:  95.64  1597
``` 

## Usage

### Train/test

To train/evaluate/save trained tagger model, please run the `main.py` script.

```
usage: main.py [-h] [--train TRAIN] [--dev DEV] [--test TEST]
               [-d {connl-ner-2003,connl-pe,connl-wd}] [--gpu GPU]
               [--model {BiRNN,BiRNNCNN,BiRNNCRF,BiRNNCNNCRF}] [--load LOAD]
               [--save SAVE] [--word-seq-indexer WORD_SEQ_INDEXER]
               [--epoch-num EPOCH_NUM] [--min-epoch-num MIN_EPOCH_NUM]
               [--patience PATIENCE]
               [--evaluator {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}]
               [--save-best [{yes,True,no default),False}]]
               [--dropout-ratio DROPOUT_RATIO] [--batch-size BATCH_SIZE]
               [--opt {sgd,adam}] [--lr LR] [--lr-decay LR_DECAY]
               [--momentum MOMENTUM] [--clip-grad CLIP_GRAD]
               [--rnn-type {Vanilla,LSTM,GRU}]
               [--rnn-hidden-dim RNN_HIDDEN_DIM] [--emb-fn EMB_FN]
               [--emb-dim EMB_DIM] [--emb-delimiter EMB_DELIMITER]
               [--emb-load-all [{yes,True,no (default),False}]]
               [--freeze-word-embeddings [{yes,True,no (default),False}]]
               [--check-for-lowercase [{yes (default),True,no,False}]]
               [--char-embeddings-dim CHAR_EMBEDDINGS_DIM]
               [--char-cnn_filter-num CHAR_CNN_FILTER_NUM]
               [--char-window-size CHAR_WINDOW_SIZE]
               [--freeze-char-embeddings [{yes,True,no (default),False}]]
               [--word-len WORD_LEN]
               [--dataset-sort [{yes,True,no (default),False}]]
               [--seed-num SEED_NUM] [--report-fn REPORT_FN]
               [--cross-folds-num CROSS_FOLDS_NUM]
               [--cross-fold-id CROSS_FOLD_ID]
               [--verbose [{yes (default,True,no,False}]]

Learning tagger using neural networks

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Train data in format defined by --data-io param.
  --dev DEV             Development data in format defined by --data-io param.
  --test TEST           Test data in format defined by --data-io param.
  -d {connl-ner-2003,connl-pe,connl-wd}, --data-io {connl-ner-2003,connl-pe,connl-wd}
                        Data read/write file format.
  --gpu GPU             GPU device number, -1 means CPU.
  --model {BiRNN,BiRNNCNN,BiRNNCRF,BiRNNCNNCRF}
                        Tagger model.
  --load LOAD, -l LOAD  Path to load from the trained model.
  --save SAVE, -s SAVE  Path to save the trained model.
  --word-seq-indexer WORD_SEQ_INDEXER, -w WORD_SEQ_INDEXER
                        Load word_seq_indexer object from hdf5 file.
  --epoch-num EPOCH_NUM, -e EPOCH_NUM
                        Number of epochs.
  --min-epoch-num MIN_EPOCH_NUM, -n MIN_EPOCH_NUM
                        Minimum number of epochs.
  --patience PATIENCE, -p PATIENCE
                        Patience for early stopping.
  --evaluator {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}, -v {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}
                        Evaluation method.
  --save-best [{yes,True,no (default),False}]
                        Save best on dev model as a final model.
  --dropout-ratio DROPOUT_RATIO, -r DROPOUT_RATIO
                        Dropout ratio.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size, samples.
  --opt {sgd,adam}, -o {sgd,adam}
                        Optimization method.
  --lr LR               Learning rate.
  --lr-decay LR_DECAY   Learning decay rate.
  --momentum MOMENTUM, -m MOMENTUM
                        Learning momentum rate.
  --clip-grad CLIP_GRAD
                        Clipping gradients maximum L2 norm.
  --rnn-type {Vanilla,LSTM,GRU}
                        RNN cell units type.
  --rnn-hidden-dim RNN_HIDDEN_DIM
                        Number hidden units in the recurrent layer.
  --emb-fn EMB_FN       Path to word embeddings file.
  --emb-dim EMB_DIM     Dimension of word embeddings file.
  --emb-delimiter EMB_DELIMITER
                        Delimiter for word embeddings file.
  --emb-load-all [{yes,True,no (default),False}]
                        Load all embeddings to model.
  --freeze-word-embeddings [{yes,True,no (default),False}]
                        False to continue training the word embeddings.
  --check-for-lowercase [{yes (default),True,no,False}]
                        Read characters caseless.
  --char-embeddings-dim CHAR_EMBEDDINGS_DIM
                        Char embeddings dim, only for char CNNs.
  --char-cnn_filter-num CHAR_CNN_FILTER_NUM
                        Number of filters in Char CNN.
  --char-window-size CHAR_WINDOW_SIZE
                        Convolution1D size.
  --freeze-char-embeddings [{yes,True,no (default),False}]
                        False to continue training the char embeddings.
  --word-len WORD_LEN   Max length of words in characters for char CNNs.
  --dataset-sort [{yes,True,no (default),False}]
                        Sort sequences by length for training.
  --seed-num SEED_NUM   Random seed number, note that 42 is the answer.
  --report-fn REPORT_FN
                        Report filename.
  --cross-folds-num CROSS_FOLDS_NUM
                        Number of folds for cross-validation (optional, for
                        some datasets).
  --cross-fold-id CROSS_FOLD_ID
                        Current cross-fold, 1<=cross-fold-id<=cross-folds-num
                        (optional, for some datasets).
  --verbose [{yes (default),True,no,False}]
                        Show additional information.
```

### Run trained model

```
usage: run_tagger.py [-h] [--fn FN] [-d {connl-ner-2003,connl-pe,connl-wd}]
                     [--evaluator {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}]
                     [--gpu GPU]
                     load

Run trained tagger from the checkpoint file

positional arguments:
  load                  Path to load from the trained model.

optional arguments:
  -h, --help            show this help message and exit
  --fn FN               Train data in CoNNL-2003 format.
  -d {connl-ner-2003,connl-pe,connl-wd}, --data-io {connl-ner-2003,connl-pe,connl-wd}
                        Data read/write file format.
  --evaluator {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}, -v {f1-connl,f1-alpha-match-10,f1-alpha-match-05,f1-macro,token-acc}
                        Evaluation method.
  --gpu GPU             GPU device number, 0 by default, -1 means CPU.
```

### Example of output report

```
Evaluation

batch_size=10
char_cnn_filter_num=30
char_embeddings_dim=25
char_window_size=3
check_for_lowercase=True
clip_grad=5
cross_fold_id=-1
cross_folds_num=-1
data_io='connl-ner-2003'
dataset_sort=False
dev='data/NER/CoNNL_2003_shared_task/dev.txt'
dropout_ratio=0.5
emb_delimiter=' '
emb_dim=100
emb_fn='embeddings/glove.6B.100d.txt'
emb_load_all=False
epoch_num=100
evaluator='f1-connl'
freeze_char_embeddings=False
freeze_word_embeddings=False
gpu=0
load=None
lr=0.01
lr_decay=0.05
min_epoch_num=50
model='BiRNNCNNCRF'
momentum=0.9
opt='sgd'
patience=15
report_fn='2019_01_21_18-59_27_report.txt'
rnn_hidden_dim=100
rnn_type='LSTM'
save='2019_01_21_18-59_27_tagger.hdf5'
save_best=True
seed_num=42
test='data/NER/CoNNL_2003_shared_task/test.txt'
train='data/NER/CoNNL_2003_shared_task/train.txt'
verbose=True
word_len=20
word_seq_indexer=None

         epoch  |     train loss | f1-connl-train |   f1-connl-dev |  f1-connl-test 
--------------------------------------------------------------------------------------
              0 |           0.00 |           2.80 |           2.13 |           5.32 
              1 |         289.14 |          78.75 |          78.50 |          77.45 
              2 |         152.54 |          85.58 |          86.30 |          82.26 
              3 |         113.81 |          89.69 |          89.21 |          86.46 
              4 |          97.00 |          90.32 |          89.08 |          85.63 
              5 |          79.96 |          91.38 |          90.07 |          86.23 
              6 |          75.38 |          90.08 |          88.59 |          84.31 
              7 |          73.16 |          92.88 |          90.97 |          87.07 
              8 |          65.99 |          93.12 |          91.11 |          88.10 
              9 |          63.29 |          92.85 |          91.31 |          87.24 
             10 |          59.74 |          94.26 |          91.80 |          88.62 
             11 |          57.86 |          92.43 |          90.07 |          85.36 
             12 |          59.29 |          94.45 |          91.56 |          88.43 
             13 |          49.88 |          94.78 |          91.59 |          88.79 
             14 |          43.62 |          95.25 |          91.94 |          88.91 
             15 |          49.21 |          95.52 |          92.51 |          88.80 
             16 |          48.19 |          95.64 |          92.58 |          89.09 
             17 |          41.08 |          95.74 |          92.37 |          88.86 
             18 |          42.94 |          95.98 |          92.35 |          89.23 
             19 |          40.35 |          96.25 |          92.59 |          88.93 
             20 |          37.66 |          95.83 |          92.49 |          88.31 
             21 |          34.63 |          96.51 |          92.78 |          89.22 
             22 |          34.85 |          96.28 |          92.68 |          89.17 
             23 |          33.07 |          96.61 |          92.99 |          89.34 
             24 |          36.03 |          96.31 |          92.78 |          89.11 
             25 |          32.11 |          96.70 |          92.87 |          89.20 
             26 |          29.25 |          96.95 |          93.72 |          89.87 
             27 |          33.87 |          97.16 |          93.38 |          89.59 
             28 |          34.77 |          97.30 |          93.82 |          89.95 
             29 |          28.70 |          96.97 |          93.20 |          89.54 
             30 |          30.73 |          96.92 |          93.10 |          89.13 
             31 |          29.28 |          97.34 |          93.51 |          89.93 
             32 |          28.76 |          97.10 |          92.93 |          88.81 
             33 |          28.58 |          97.56 |          93.55 |          90.09 
             34 |          24.82 |          97.18 |          93.58 |          89.82 
             35 |          28.73 |          97.81 |          94.01 |          90.21 
             36 |          24.40 |          97.72 |          93.59 |          89.82 
             37 |          24.64 |          97.85 |          93.79 |          89.87 
             38 |          29.51 |          97.96 |          93.82 |          90.08 
             39 |          27.63 |          98.05 |          94.30 |          90.05 
             40 |          25.07 |          98.02 |          94.15 |          90.08 
             41 |          22.76 |          98.16 |          94.10 |          90.20 
             42 |          22.73 |          97.96 |          93.86 |          89.92 
             43 |          22.78 |          98.37 |          94.41 |          90.59 
             44 |          22.38 |          98.24 |          94.05 |          90.25 
             45 |          19.20 |          98.33 |          94.10 |          89.90 
             46 |          21.02 |          98.30 |          93.97 |          90.13 
             47 |          22.39 |          98.25 |          94.17 |          90.53 
             48 |          19.91 |          98.30 |          94.15 |          90.09 
             49 |          20.66 |          98.17 |          93.61 |          90.15 
             50 |          18.91 |          98.34 |          93.84 |          89.81 
             51 |          19.94 |          98.25 |          93.82 |          90.00 
             52 |          18.10 |          98.44 |          94.10 |          90.09 
             53 |          17.20 |          98.52 |          94.30 |          90.29 
             54 |          20.11 |          98.38 |          93.92 |          90.36 
             55 |          17.46 |          98.51 |          94.34 |          90.29 
             56 |          19.65 |          98.38 |          94.05 |          90.07 
             57 |          18.48 |          98.47 |          93.98 |          90.11 
             58 |          19.20 |          98.38 |          94.01 |          90.10 
             59 |          16.50 |          98.68 |          94.06 |          90.09 
--------------------------------------------------------------------------------------
Final eval on test, "save best", best epoch on dev 43, f1-connl, test = 90.59)
--------------------------------------------------------------------------------------
Standard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):
processed 46435 tokens with 5648 phrases; found: 5622 phrases; correct: 5105.
accuracy:  97.92%; precision:  90.80%; recall:  90.39%; FB1:  90.59
              LOC: precision:  93.06%; recall:  91.67%; FB1:  92.36  1643
             MISC: precision:  78.75%; recall:  80.77%; FB1:  79.75  720
              ORG: precision:  88.57%; recall:  88.62%; FB1:  88.59  1662
              PER: precision:  96.24%; recall:  95.05%; FB1:  95.64  1597

Input arguments:
python3 main.py 
```

### Training on various datasets

Training on NER-2003 Shared dataset:
```
python3 main.py
```

Training on Peruassive Essays dataset:
```
python3 main.py --train data/AM/persuasive_essays/Essay_Level/train.dat.abs --dev data/AM/persuasive_essays/Essay_Level/dev.dat.abs --test data/AM/persuasive_essays/Essay_Level/test.dat.abs --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes
```

Training on Web Discourse dataset (cross-validation):
```
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 1;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 2;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 3;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 4;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 5;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 6;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 7;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 8;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 9;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --rnn-hidden-dim 150 --cross-folds-num 10 --cross-fold-id 10;
```

### Alternative neural taggers

- NeuroNER (Tensorflow) [https://github.com/Franck-Dernoncourt/NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER)
- LM-LSTM-CRF (Pytorch) [https://github.com/LiyuanLucasLiu/LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)
- LD-Net (Pytorch) [https://github.com/LiyuanLucasLiu/LD-Net](https://github.com/LiyuanLucasLiu/LD-Net)
- LSTM-CRF in PyTorch (Pytorch) [https://github.com/threelittlemonkeys/lstm-crf-pytorch](https://github.com/threelittlemonkeys/lstm-crf-pytorch) 
- UKPLab/emnlp2017-bilstm-cnn-crf (Tensorflow & Keras)
[https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf)
- UKPLab/elmo-bilstm-cnn-crf (Tensorflow & Keras)
[https://github.com/UKPLab/elmo-bilstm-cnn-crf](https://github.com/UKPLab/elmo-bilstm-cnn-crf)
