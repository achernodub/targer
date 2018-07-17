import random


class DatasetsBank():
    """
    DatasetsBank provides 1) storing the different dataset subsets (train/dev/test)
                          2) sampling batches from the train dataset subset
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
        self.train_data_num = self.targets_tensor_train.shape[0]
        self.train_max_seq_len = self.targets_tensor_train.shape[1]

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

    def get_train_batch(self, batch_size=0):
        if batch_size == 0:
            return self.inputs_tensor_train, self.targets_tensor_train
        batch_indices = random.sample(range(0, self.train_data_num), batch_size)
        inputs_tensor_train_batch = self.inputs_tensor_train[batch_indices, :].view(batch_size, self.train_max_seq_len)
        targets_tensor_train_batch = self.targets_tensor_train[batch_indices, :].view(batch_size, self.train_max_seq_len)
        return inputs_tensor_train_batch, targets_tensor_train_batch
