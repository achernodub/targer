import codecs

def read_CoNNL(fn, column_no=-1):
    token_sequences = list()
    tag_sequences = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    curr_tokens = list()
    curr_tags = list()
    for k in range(len(lines)):
        line = lines[k].strip()
        if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
            if len(curr_tokens) > 0:
                token_sequences.append(curr_tokens)
                tag_sequences.append(curr_tags)
                curr_tokens = list()
                curr_tags = list()
            continue
        strings = line.split(' ')
        token = strings[0]
        tag = strings[column_no] # be default, we take the last tag
        curr_tokens.append(token)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            token_sequences.append(curr_tokens)
            tag_sequences.append(curr_tags)
    return token_sequences, tag_sequences

def write_CoNNL(fn, token_sequences, tag_sequences_1, tag_sequences_2=None):
    if tag_sequences_2 is not None:
        write_CoNNL_three_columns(fn, token_sequences, tag_sequences_1, tag_sequences_2)
    else:
        write_CoNNL_two_columns(fn, token_sequences, tag_sequences_1)

def write_CoNNL_two_columns(fn, token_sequences, tag_sequences_1):
    text_file = open(fn, mode='w')
    for i, tokens in enumerate(token_sequences):
        text_file.write('-DOCSTART- -X- -X-\n')
        tags_1 = tag_sequences_1[i]
        for j, token in enumerate(tokens):
            tag_1 = tags_1[j]
            text_file.write('%s %s\n' % (token, tag_1))
    text_file.close()

def write_CoNNL_three_columns(fn, token_sequences, tag_sequences_1, tag_sequences_2):
    text_file = open(fn, mode='w')
    for i, tokens in enumerate(token_sequences):
        text_file.write('-DOCSTART- -X- -X- -X-\n')
        tags_1 = tag_sequences_1[i]
        tags_2 = tag_sequences_2[i]
        for j, token in enumerate(tokens):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (token, tag_1, tag_2))
    text_file.close()

def info(name, t):
    print(name, '|', t.type(), '|', t.shape)