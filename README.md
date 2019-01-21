# BiLSTM-CNN-CRF tagger

BiLSTM-CNN-CRF tagger is a PyTorch implementation of "mainstream" neural tagging scheme based on works of [Lample, 
et. al., 2016](https://arxiv.org/pdf/1603.01360.pdf) and [Ma et. al., 2016](https://arxiv.org/pdf/1603.01354.pdf).

<p align="center"><img width="100%" src="docs/scheme.png"/></p> 

## Requirements

- numpy 1.15.1
- scipy 1.1.0
- python 3.5.2 or higher
- [pytorch 1.0.0](http://pytorch.org/)

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
processed 46435 tokens with 5648 phrases; found: 5650 phrases; correct: 5108.
accuracy:  97.96%; precision:  90.41%; recall:  90.44%; FB1:  90.42
              LOC: precision:  91.99%; recall:  92.99%; FB1:  92.49  1686
             MISC: precision:  79.29%; recall:  79.06%; FB1:  79.17  700
              ORG: precision:  88.38%; recall:  87.96%; FB1:  88.17  1653
              PER: precision:  95.65%; recall:  95.30%; FB1:  95.48  1611
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
patience=10
report_fn='2019_01_19_13-35_01_report.txt'
rnn_hidden_dim=100
rnn_type='LSTM'
save='2019_01_19_13-35_01_tagger.hdf5'
save_best=False
seed_num=42
test='data/NER/CoNNL_2003_shared_task/test.txt'
train='data/NER/CoNNL_2003_shared_task/train.txt'
verbose=True
word_len=20
word_seq_indexer=None

         epoch  |     train loss | f1-connl-train |   f1-connl-dev |  f1-connl-test 
--------------------------------------------------------------------------------------
              0 |           0.00 |           2.05 |           1.87 |           2.35 
              1 |         395.90 |          69.37 |          70.53 |          69.87 
              2 |         188.05 |          80.81 |          81.56 |          78.61 
              3 |         137.46 |          84.62 |          84.43 |          81.60 
              4 |         115.03 |          87.07 |          86.12 |          83.29 
              5 |          97.34 |          88.00 |          87.19 |          83.56 
              6 |          90.09 |          86.53 |          85.23 |          81.55 
              7 |          89.18 |          90.29 |          89.42 |          86.14 
              8 |          74.99 |          91.02 |          89.67 |          86.51 
              9 |          75.06 |          91.27 |          89.88 |          87.13 
             10 |          69.05 |          92.28 |          90.87 |          87.42 
             11 |          67.44 |          87.80 |          86.43 |          82.78 
             12 |          66.52 |          92.57 |          90.62 |          87.29 
             13 |          54.99 |          93.19 |          90.47 |          87.29 
             14 |          48.47 |          92.83 |          89.92 |          86.25 
             15 |          57.26 |          91.94 |          90.03 |          86.02 
             16 |          55.66 |          93.60 |          91.21 |          87.03 
             17 |          47.48 |          94.15 |          91.41 |          87.28 
             18 |          51.09 |          93.38 |          90.19 |          87.16 
             19 |          47.80 |          94.70 |          91.71 |          88.20 
             20 |          43.85 |          93.75 |          90.68 |          86.19 
             21 |          41.36 |          95.02 |          91.83 |          87.67 
             22 |          39.41 |          95.09 |          91.86 |          88.00 
             23 |          39.71 |          95.41 |          91.99 |          88.28 
             24 |          39.95 |          95.31 |          92.09 |          88.62 
             25 |          35.52 |          95.37 |          92.15 |          88.33 
             26 |          35.37 |          96.07 |          93.07 |          89.01 
             27 |          38.43 |          95.17 |          92.16 |          87.82 
             28 |          40.02 |          95.81 |          92.69 |          88.43 
             29 |          33.04 |          95.54 |          92.40 |          88.45 
             30 |          33.25 |          95.55 |          92.32 |          88.10 
             31 |          32.77 |          96.62 |          93.19 |          89.59 
             32 |          34.59 |          94.90 |          91.32 |          87.04 
             33 |          32.12 |          96.53 |          93.29 |          89.59 
             34 |          28.98 |          95.60 |          92.34 |          88.37 
             35 |          31.30 |          96.77 |          93.21 |          89.44 
             36 |          28.02 |          96.35 |          93.01 |          88.64 
             37 |          26.80 |          96.78 |          93.25 |          89.43 
             38 |          33.68 |          96.98 |          93.46 |          89.19 
             39 |          31.00 |          96.86 |          93.35 |          89.22 
             40 |          29.37 |          97.32 |          93.45 |          89.54 
             41 |          25.93 |          97.31 |          93.78 |          89.72 
             42 |          26.08 |          97.13 |          93.36 |          89.51 
             43 |          27.41 |          97.54 |          93.71 |          89.78 
             44 |          27.16 |          97.29 |          93.57 |          89.77 
             45 |          22.28 |          97.51 |          93.83 |          89.58 
             46 |          22.99 |          97.40 |          93.31 |          89.36 
             47 |          23.20 |          97.63 |          93.92 |          90.12 
             48 |          23.77 |          97.53 |          93.68 |          89.42 
             49 |          24.57 |          97.56 |          93.59 |          89.68 
             50 |          22.84 |          97.74 |          93.34 |          89.53 
             51 |          23.20 |          97.33 |          93.00 |          89.17 
             52 |          21.69 |          97.88 |          94.05 |          89.81 
             53 |          20.09 |          97.70 |          93.66 |          89.41 
             54 |          23.27 |          97.59 |          93.66 |          89.34 
             55 |          19.02 |          97.72 |          93.76 |          89.59 
             56 |          23.45 |          97.84 |          93.91 |          89.39 
             57 |          20.97 |          97.89 |          93.88 |          89.75 
             58 |          21.34 |          97.87 |          94.02 |          89.70 
             59 |          19.90 |          97.94 |          93.52 |          89.50 
             60 |          23.31 |          97.80 |          93.68 |          89.88 
             61 |          21.26 |          97.99 |          93.82 |          89.97 
             62 |          17.45 |          97.56 |          93.25 |          89.23 
             63 |          20.85 |          98.34 |          94.03 |          90.42 
--------------------------------------------------------------------------------------
Final eval on test, f1-connl test = 90.42)
--------------------------------------------------------------------------------------
Standard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):
processed 46435 tokens with 5648 phrases; found: 5650 phrases; correct: 5108.
accuracy:  97.96%; precision:  90.41%; recall:  90.44%; FB1:  90.42
              LOC: precision:  91.99%; recall:  92.99%; FB1:  92.49  1686
             MISC: precision:  79.29%; recall:  79.06%; FB1:  79.17  700
              ORG: precision:  88.38%; recall:  87.96%; FB1:  88.17  1653
              PER: precision:  95.65%; recall:  95.30%; FB1:  95.48  1611


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
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 1;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 2;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 3;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 4;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 5;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 6;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 7;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 8;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 9;
python3 main.py --train data/AM/web_discourse --evaluator f1-macro --data-io connl-wd --opt adam --lr 0.001 --save-best yes -w w_wd --cross-folds-num 10 --cross-fold-id 10;
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
