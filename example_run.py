from __future__ import print_function

import time

from sequences_indexer import SequencesIndexer
from datasets_bank import DatasetsBank
from evaluator import Evaluator
from models.tagger_birnn import TaggerBiRNN
from utils import *

print('Start!')

gpu = 0

# Load tagger model
tagger = torch.load('tagger_model.txt')

if gpu >= 0:
    tagger = tagger.cuda(device=0)

# We take sequences_indexer from the tagger
sequences_indexer = tagger.sequences_indexer

evaluator = Evaluator(sequences_indexer)

fn_test = 'data/argument_mining/persuasive_essays/es_paragraph_level_test.txt'

token_sequences_test, tag_sequences_test = read_CoNNL(fn_test)

output_tag_sequences_test = tagger.predict_tags_from_tokens(token_sequences_test)


f1, precision, recall = evaluator.get_macro_scores_tokens_tags(tagger, token_sequences_test, tag_sequences_test)

print(f1, precision, recall)


print(len(tag_sequences_test))
print(len(output_tag_sequences_test))

for i in range(len(tag_sequences_test)):
    t = tag_sequences_test[i]
    y = output_tag_sequences_test[i]
    for j in range(len(t)):
        print(t[j], y[j])







print('The end.')