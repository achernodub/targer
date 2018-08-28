import random
import numpy as np


class DatasetsBank():
    """
    DatasetsBank provides 1) storing the different dataset subsets (train/dev/test)
                          2) sampling batches from the train dataset subset
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.words_list = list()

    def __expand_words_list(self, word_sequences):
        for word_seq in word_sequences:
            for word in word_seq:
                if word not in self.words_list:
                    self.words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(words_list)=%d' % (len(self.words_list)))

    def add_train_sequences(self, word_sequences_train, tag_sequences_train):
        self.word_sequences_train = word_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.__expand_words_list(word_sequences_train)
        self.train_data_num = len(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.__expand_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.__expand_words_list(word_sequences_test)

    def __get_train_batch(self, batch_indices):
        word_sequences_train_batch = [self.word_sequences_train[i] for i in batch_indices]
        tag_sequences_train_batch = [self.tag_sequences_train[i] for i in batch_indices]
        return word_sequences_train_batch, tag_sequences_train_batch

    def get_train_batches(self, batch_size):
        random_indices = np.random.permutation(np.arange(self.train_data_num))
        word_sequences_train_batch_list = []
        tag_sequences_train_batch_list = []
        for k in range(self.train_data_num // batch_size): # we drop the last batch
            batch_indices = random_indices[k:k + batch_size].tolist()
            inputs_train_batch, targets_train_batch = self.__get_train_batch(batch_indices)
            word_sequences_train_batch_list.append(inputs_train_batch)
            tag_sequences_train_batch_list.append(targets_train_batch)
        return word_sequences_train_batch_list, tag_sequences_train_batch_list