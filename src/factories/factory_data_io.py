from src.data_io.data_io_connl_abs import DataIOConnlAbs
from src.data_io.data_io_connl_2003 import DataIOConnl2003


class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'connl-abs':
            return DataIOConnlAbs()
        elif args.data_io == 'connl-2003':
            return DataIOConnl2003()
        else:
            raise ValueError('Unknown DataIO %s.' % args.data_io)
