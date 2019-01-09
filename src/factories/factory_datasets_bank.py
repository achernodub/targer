"""
.. module:: DatasetsBankFactory
    :synopsis: DatasetsBankFactory contains wrappers to create DatasetsBank

.. moduleauthor:: Artem Chernodub
"""
from src.classes.datasets_bank import DatasetsBank, DatasetsBankSorted


class DatasetsBankFactory():
    @staticmethod
    def create(args):
        if args.dataset_sort:
            datasets_bank = DatasetsBankSorted(verbose=True)
        else:
            datasets_bank = DatasetsBank(verbose=True)
        return datasets_bank
