"""creates various data readers/writers"""
from src.data_io.data_io_connl_ner_2003 import DataIOConnlNer2003
from src.data_io.data_io_connl_pe import DataIOConnlPe
from src.data_io.data_io_connl_wd import DataIOConnlWd

class DataIOFactory():
    """DataIOFactory contains wrappers to create various data readers/writers."""
    @staticmethod
    def create(args):
        if args.data_io == 'connl-ner-2003':
            return DataIOConnlNer2003()
        elif args.data_io == 'connl-pe':
            return DataIOConnlPe()
        elif args.data_io == 'connl-wd':
            return DataIOConnlWd()
        else:
            raise ValueError('Unknown DataIO %s.' % args.data_io)
