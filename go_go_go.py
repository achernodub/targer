from __future__ import print_function

import random

from sequences_indexer import SequencesIndexer
from evaluator import Evaluator
from models.tagger_birnn import TaggerBiRNN
from utils import *

print('Hello, train/dev/test script!')

emb_fn = 'embeddings/glove.6B.100d.txt'
gpu = -1 # current version is for CPU only!

caseless = True
shrink_to_train = False
unk = None
delimiter = ' '
epoch_num = 50

#hidden_layer_dim = 200
#hidden_layer_num = 1
rnn_hidden_size = 101
dropout_ratio = 0.5
clip_grad = 5.0
opt_method = 'sgd'

lr = 0.015
momentum = 0.9
batch_size = 5

debug_mode = False
verbose = True

seed_num = 42
np.random.seed(seed_num)
torch.manual_seed(seed_num)

freeze_embeddings = False

if gpu >= 0:
    torch.cuda.manual_seed(seed_num)

# Select data
if (1 == 1):
    # Essays
    fn_train = 'data/argument_mining/persuasive_essays/es_paragraph_level_train.txt'
    fn_dev = 'data/argument_mining/persuasive_essays/es_paragraph_level_dev.txt'
    fn_test = 'data/argument_mining/persuasive_essays/es_paragraph_level_test.txt'
else:
    # CoNNL-2003 NER shared task
    fn_train = 'data/NER/CoNNL_2003_shared_task/train.txt'
    fn_dev = 'data/NER/CoNNL_2003_shared_task/dev.txt'
    fn_test = 'data/NER/CoNNL_2003_shared_task/test.txt'


# Load CoNNL data as sequences of strings
token_sequences_train, tag_sequences_train = read_CoNNL(fn_train)
token_sequences_dev, tag_sequences_dev = read_CoNNL(fn_dev)
token_sequences_test, tag_sequences_test = read_CoNNL(fn_test)

# SequenceIndexer is a class to convert tokens and tags as strings to integer indices and back
sequences_indexer = SequencesIndexer(caseless=caseless, verbose=verbose)
sequences_indexer.load_embeddings(emb_fn=emb_fn, delimiter=delimiter)
sequences_indexer.add_token_sequences(token_sequences_train)
sequences_indexer.add_token_sequences(token_sequences_dev)
sequences_indexer.add_token_sequences(token_sequences_test)
sequences_indexer.add_tag_sequences(tag_sequences_train) # Surely, all necessarily tags must be into train data

inputs_idx_train = sequences_indexer.token2idx(token_sequences_train)
outputs_idx_train = sequences_indexer.tag2idx(tag_sequences_train)

batch_indices = random.sample(range(0, len(inputs_idx_train)), batch_size)
inputs_idx_train_batch = [inputs_idx_train[k] for k in batch_indices]
targets_idx_train_batch = [outputs_idx_train[k] for k in batch_indices]

inputs_tensor_train_batch = sequences_indexer.idx2tensor(inputs_idx_train_batch)
targets_tensor_train_batch = sequences_indexer.idx2tensor(targets_idx_train_batch)


print('Start...\n\n')

evaluator = Evaluator()

tagger = TaggerBiRNN(embeddings_tensor=sequences_indexer.get_embeddings_tensor(),
                     class_num=sequences_indexer.get_tags_num(),
                     rnn_hidden_size=rnn_hidden_size,
                     freeze_embeddings=freeze_embeddings,
                     dropout_ratio=dropout_ratio,
                     rnn_type='GRU')

nll_loss = nn.NLLLoss(ignore_index=0) # we suppose that target values "0" are zero-padded parts of sequences and
                                      # don't include them for calculating the derivatives for the loss function
optimizer = optim.SGD(list(tagger.parameters()), lr=lr, momentum=momentum)

for i in range(200):
    tagger.train()
    tagger.zero_grad()
    outputs_train_batch = tagger(inputs_tensor_train_batch)
    loss = nll_loss(outputs_train_batch, targets_tensor_train_batch)
    loss.backward()
    optimizer.step()
    f1, precision, recall = evaluator.get_macro_scores_from_inputs_tensor(tagger=tagger,
                                                                          inputs_tensor=inputs_tensor_train_batch,
                                                                          targets_idx=targets_idx_train_batch)
    print('i = %d, loss = %1.4f, F1 = %1.3f, Precision = %1.3f, Recall = %1.3f' % (i, loss.item(), f1, precision, recall))

curr_target_tags = sequences_indexer.idx2tag(targets_idx_train_batch)
curr_output_tags = tagger.predict_tags_from_tensor(inputs_tensor_train_batch, sequences_indexer)

print('curr_target_tags', curr_target_tags)
print('curr_output_tags', curr_output_tags)

print('The end!')
