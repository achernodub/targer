import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '|__' * (level)
        dir_name = os.path.basename(root)
        if dir_name.startswith('__'):
            continue
        if dir_name.startswith('.'):
            continue
        print('{}{}/'.format(indent, dir_name))
        subindent = '   ' * (level) + '|__'
        for short_fn in files:
            fn = os.path.join(root, short_fn)
            ext = short_fn.split('.')[-1]
            if ext == 'py':
                print('{}{} --> {}'.format(subindent, short_fn, read_description(fn)))
            else:
                print('{}{} --> '.format(subindent, short_fn))


def read_description(fn):
    with open(fn) as f:
        first_line = f.readline()
    return first_line.replace('"""', '').replace('\n', '')

if __name__ == "__main__":
    main_path = os.path.join(os.path.dirname(__file__), '../')
    os.chdir(main_path)
    print('|__ articles/ --> collection of papers related to the tagging, argument mining, etc.')
    print('|__ data/')
    print('        |__ NER/ --> Datasets for Named Entity Recognition')
    print('            |__ CoNNL_2003_shared_task/ --> data for NER CoNLL-2003 shared task (English) in BOI-2')
    print('                                            CoNNL format, from E.F. Tjong Kim Sang and F. De Meulder,')
    print('                                            Introduction to the CoNLL-2003 Shared Task:')
    print('                                            Language-Independent Named Entity Recognition, 2003.')
    print('        |__ AM/ --> Datasets for Argument Mining')
    print('            |__ persuasive_essays/ --> data for persuasive essays in BOI-2-like CoNNL format, from:')
    print('                                       Steffen Eger, Johannes Daxenberger, Iryna Gurevych. Neural')
    print('                                       End-to-End  Learning for Computational Argumentation Mining, 2017')
    print('|__ docs/ --> documentation')
    print('|__ embeddings')
    print('        |__ get_glove_embeddings.sh --> script for downloading GloVe6B 100-dimensional word embeddings')
    print('        |__ get_fasttext_embeddings.sh --> script for downloading Fasttext word embeddings')
    print('|__ pretrained/')
    print('        |__ tagger_NER.hdf5 --> tagger for NER, BiLSTM+CNN+CRF trained on NER-2003 shared task, English')
    list_files(startpath=os.getcwd())
    print('|__ main.py --> main script for training/evaluation/saving tagger models')
    print('|__ run_tagger.py --> run the trained tagger model from the checkpoint file')
    print('|__ conlleval --> "official" Perl script from NER 2003 shared task for evaluating the f1 scores,'
        '\n                   author: Erik Tjong Kim Sang, version: 2004-01-26')
    print('|__ requirements.txt --> file for managing packages requirements')
