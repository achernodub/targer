"""
.. module:: SeqIndexerTag
    :synopsis: SeqIndexerTag converts list of lists of string tags to list of lists of integer indices and back.

.. moduleauthor:: Artem Chernodub
"""

from seq_indexers.seq_indexer_base import SeqIndexerBase

class SeqIndexerTag(SeqIndexerBase):
    def __init__(self, gpu):
        #super(SeqIndexerTag, self).__init__(gpu)
        SeqIndexerBase.__init__(self, gpu=gpu, check_for_lowercase=False, zero_digits=False,
                                      pad='<pad>', unk=None, load_embeddings=False, verbose=True)

    def add_tag(self, tag):
        if not self.item_exists(tag):
            self.add_item(tag)

    def load_items_from_tag_sequences(self, tag_sequences):
        assert self.load_embeddings == False
        for tag_seq in tag_sequences:
            for tag in tag_seq:
                self.add_tag(tag)
        if self.verbose:
            print('\nload_vocabulary_from_tag_sequences:')
            print(' -- class_num = %d' % self.get_class_num())
            print(' --', self.item2idx_dict)
