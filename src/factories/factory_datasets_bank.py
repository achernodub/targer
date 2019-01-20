"""creates various datasets banks"""
from src.classes.datasets_bank import DatasetsBank, DatasetsBankSorted


class DatasetsBankFactory():
    """DatasetsBankFactory contains wrappers to create various datasets banks."""
    @staticmethod
    def create(args):
        if args.dataset_sort:
            datasets_bank = DatasetsBankSorted(verbose=True)
        else:
            datasets_bank = DatasetsBank(verbose=True)
        return datasets_bank
