"""creates various tagger models"""
import os.path
import torch
from src.models.tagger_birnn import TaggerBiRNN
from src.models.tagger_birnn_cnn import TaggerBiRNNCNN
from src.models.tagger_birnn_crf import TaggerBiRNNCRF
from src.models.tagger_birnn_cnn_crf import TaggerBiRNNCNNCRF


class TaggerFactory():
    """TaggerFactory contains wrappers to create various tagger models."""
    @staticmethod
    def load(checkpoint_fn, gpu=-1):
        if not os.path.isfile(checkpoint_fn):
            raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                             "--save-best-path" param to create it.' % checkpoint_fn)
        tagger = torch.load(checkpoint_fn)
        tagger.gpu = gpu
        tagger.self_ensure_gpu()
        return tagger


    @staticmethod
    def create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train):
        if args.model == 'BiRNN':
            tagger = TaggerBiRNN(word_seq_indexer=word_seq_indexer,
                                 tag_seq_indexer=tag_seq_indexer,
                                 class_num=tag_seq_indexer.get_class_num(),
                                 batch_size=args.batch_size,
                                 rnn_hidden_dim=args.rnn_hidden_dim,
                                 freeze_word_embeddings=args.freeze_word_embeddings,
                                 dropout_ratio=args.dropout_ratio,
                                 rnn_type=args.rnn_type,
                                 gpu=args.gpu)
        elif args.model == 'BiRNNCNN':
            tagger = TaggerBiRNNCNN(word_seq_indexer=word_seq_indexer,
                                    tag_seq_indexer=tag_seq_indexer,
                                    class_num=tag_seq_indexer.get_class_num(),
                                    batch_size=args.batch_size,
                                    rnn_hidden_dim=args.rnn_hidden_dim,
                                    freeze_word_embeddings=args.freeze_word_embeddings,
                                    dropout_ratio=args.dropout_ratio,
                                    rnn_type=args.rnn_type,
                                    gpu=args.gpu,
                                    freeze_char_embeddings=args.freeze_char_embeddings,
                                    char_embeddings_dim=args.char_embeddings_dim,
                                    word_len=args.word_len,
                                    char_cnn_filter_num=args.char_cnn_filter_num,
                                    char_window_size=args.char_window_size)
        elif args.model == 'BiRNNCRF':
            tagger = TaggerBiRNNCRF(word_seq_indexer=word_seq_indexer,
                                    tag_seq_indexer=tag_seq_indexer,
                                    class_num=tag_seq_indexer.get_class_num(),
                                    batch_size=args.batch_size,
                                    rnn_hidden_dim=args.rnn_hidden_dim,
                                    freeze_word_embeddings=args.freeze_word_embeddings,
                                    dropout_ratio=args.dropout_ratio,
                                    rnn_type=args.rnn_type,
                                    gpu=args.gpu)
            tagger.crf_layer.init_transition_matrix_empirical(tag_sequences_train)
        elif args.model == 'BiRNNCNNCRF':
            tagger = TaggerBiRNNCNNCRF(word_seq_indexer=word_seq_indexer,
                                       tag_seq_indexer=tag_seq_indexer,
                                       class_num=tag_seq_indexer.get_class_num(),
                                       batch_size=args.batch_size,
                                       rnn_hidden_dim=args.rnn_hidden_dim,
                                       freeze_word_embeddings=args.freeze_word_embeddings,
                                       dropout_ratio=args.dropout_ratio,
                                       rnn_type=args.rnn_type,
                                       gpu=args.gpu,
                                       freeze_char_embeddings=args.freeze_char_embeddings,
                                       char_embeddings_dim=args.char_embeddings_dim,
                                       word_len=args.word_len,
                                       char_cnn_filter_num=args.char_cnn_filter_num,
                                       char_window_size=args.char_window_size)
            tagger.crf_layer.init_transition_matrix_empirical(tag_sequences_train)
        else:
            raise ValueError('Unknown tagger model, must be one of "BiRNN"/"BiRNNCNN"/"BiRNNCRF"/"BiRNNCNNCRF".')
        return tagger
