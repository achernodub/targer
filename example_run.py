from __future__ import print_function

import time

from sequences_indexer import SequencesIndexer
from datasets_bank import DatasetsBank
from evaluator import Evaluator
from models.tagger_birnn import TaggerBiRNN
from utils import *

print('Hello!')

tagger = torch.load('tagger_model.txt')



print('The end.')