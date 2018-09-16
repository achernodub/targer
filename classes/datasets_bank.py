"""
.. module:: DatasetsBank
    :synopsis: DatasetsBank provides storing the train/dev/test data subsets and sampling batches from the train dataset

.. moduleauthor:: Artem Chernodub
"""

import numpy as np

class DatasetsBank():
    """
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.unique_words_list = list()

    def __add_to_unique_words_list(self, word_sequences):
        for word_seq in word_sequences:
            for word in word_seq:
                if word not in self.unique_words_list:
                    self.unique_words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))

    def add_train_sequences(self, word_sequences_train, tag_sequences_train):
        self.train_data_num = len(word_sequences_train)
        self.word_sequences_train = word_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.__add_to_unique_words_list(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.__add_to_unique_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.__add_to_unique_words_list(word_sequences_test)

    def __get_train_batch(self, batch_indices):
        word_sequences_train_batch = [self.word_sequences_train[i] for i in batch_indices]
        tag_sequences_train_batch = [self.tag_sequences_train[i] for i in batch_indices]
        return word_sequences_train_batch, tag_sequences_train_batch

    def get_train_batches(self, batch_size):
        random_indices = np.random.permutation(np.arange(self.train_data_num))
        for k in range(self.train_data_num // batch_size): # oh yes, we drop the last batch
            batch_indices = random_indices[k:k + batch_size].tolist()
            word_sequences_train_batch, tag_sequences_train_batch = self.__get_train_batch(batch_indices)
            yield word_sequences_train_batch, tag_sequences_train_batch
