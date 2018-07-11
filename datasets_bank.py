import random
import numpy as np
import torch

from sequences_indexer import SequencesIndexer

class DatasetsBank():
    """
    DatasetsBank provides 1) storing the different dataset subsets (train/dev/test)
                          2) sampling batches from them
    """

    def __init__(self, sequences_indexer):
        self.sequences_indexer = sequences_indexer

    def add_train_sequences(self, token_sequences_train, tag_sequences_train):
        self.token_sequences_train = token_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.inputs_idx_train = self.sequences_indexer.token2idx(token_sequences_train)
        self.targets_idx_train = self.sequences_indexer.tag2idx(tag_sequences_train)
        self.inputs_tensor_train = self.sequences_indexer.idx2tensor(self.inputs_idx_train)
        self.targets_tensor_train = self.sequences_indexer.idx2tensor(self.targets_idx_train)

    def add_dev_sequences(self, token_sequences_dev, tag_sequences_dev):
        self.token_sequences_dev = token_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.inputs_idx_dev = self.sequences_indexer.token2idx(token_sequences_dev)
        self.targets_idx_dev = self.sequences_indexer.tag2idx(tag_sequences_dev)
        self.inputs_tensor_dev = self.sequences_indexer.idx2tensor(self.inputs_idx_dev)
        self.targets_tensor_dev = self.sequences_indexer.idx2tensor(self.targets_idx_dev)

    def add_test_sequences(self, token_sequences_test, tag_sequences_test):
        self.token_sequences_test = token_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.inputs_idx_test = self.sequences_indexer.token2idx(token_sequences_test)
        self.targets_idx_test = self.sequences_indexer.tag2idx(tag_sequences_test)
        self.inputs_tensor_test = self.sequences_indexer.idx2tensor(self.inputs_idx_test)
        self.targets_tensor_test = self.sequences_indexer.idx2tensor(self.targets_idx_test)

    def get_batch(self, dataset_subset, batch_size=0):
        if dataset_subset == 'train':
            inputs_tensor = self.inputs_tensor_train
            targets_tensor = self.targets_tensor_train
        elif dataset_subset == 'dev':
            inputs_tensor = self.inputs_tensor_dev
            targets_tensor = self.targets_tensor_dev
        elif dataset_subset == 'dev':
            inputs_tensor = self.inputs_tensor_test
            targets_tensor = self.targets_tensor_test
        else:
            raise ValueError('Unknown dataset_part parameter, should be "train", "dev" or "test"')
        if batch_size == 0:
            return inputs_tensor, targets_tensor
        data_num = inputs_tensor.shape[0]
        print('data_num', data_num)
        #batch_indices = random.sample(range(0, len(inputs_idx_train)), batch_size)
        batch_indices = random.sample(range(0, data_num), batch_size)
        return inputs_tensor[batch_indices, :], targets_tensor[batch_indices, :]
