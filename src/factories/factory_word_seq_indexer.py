"""creates various word sequence indexers"""
from os.path import isfile
import torch
from src.seq_indexers.seq_indexer_word import SeqIndexerWord
from src.seq_indexers.seq_indexer_word_bert import SeqIndexerWordBert


class WordSeqIndexerFactory():
    """WordSeqIndexerFactory creates various word sequence indexers"""
    @staticmethod
    def create(args):
        if args.emb_bert:
            return SeqIndexerWordBert(gpu=args.gpu)
        else:
            if args.word_seq_indexer is not None and isfile(args.word_seq_indexer):
                word_seq_indexer = torch.load(args.word_seq_indexer)
            else:
                word_seq_indexer = SeqIndexerWord(gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
                                                  embeddings_dim=args.emb_dim, verbose=True)
                word_seq_indexer.load_items_from_embeddings_file_and_unique_words_list(emb_fn=args.emb_fn,
                                                                                       emb_delimiter=args.emb_delimiter,
                                                                                       emb_load_all=args.emb_load_all,
                                                                                       unique_words_list=datasets_bank.unique_words_list)
            if args.word_seq_indexer is not None and not isfile(args.word_seq_indexer):
                torch.save(word_seq_indexer, args.word_seq_indexer)
            return word_seq_indexer
