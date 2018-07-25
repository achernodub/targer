
class TagComponent():
    def __init__(self, pos_begin, tag):
        self.pos_begin = pos_begin
        self.tag_class_name = TagComponent.get_tag_class_name(tag)
        self.pos_end = self.pos_begin - 1
        self.tokens = list()

    def has_same_tag_class(self, tag):
        return self.tag_class_name == TagComponent.get_tag_class_name(tag)

    def add_token(self, token):
        self.tokens.append(token)
        self.pos_end += 1

    def is_equal(self, tc, match_ratio=0.5):
        return TagComponent.match(self, tc, match_ratio)

    def print(self):
        print('--tag_class_name = %s, pos_begin = %s, pos_end = %s' % (self.tag_class_name, self.pos_begin,
                                                                         self.pos_end))
        token_str = '    '
        for token in self.tokens: token_str += token + ' '
        print(token_str, '\n')

    @staticmethod
    def get_tag_class_name(tag):
        if '-' in tag:
            tag_class_name = tag.split('-')[1] # i.e. 'Claim', 'Premise', etc.
        else:
            tag_class_name = tag # i.e. 'O'
        return tag_class_name

    @staticmethod
    def get_tag_class_name_by_idx(tag_idx, sequences_indexer):
        tag = sequences_indexer.idx2tag_dict[tag_idx]
        return TagComponent.get_tag_class_name(tag)

    @staticmethod
    def match(tc1, tc2, match_ratio):
        if tc1.tag_class_name != tc2.tag_class_name:
            return False
        tc1_positions = set(range(tc1.pos_begin, tc1.pos_end + 1))
        tc2_positions = set(range(tc2.pos_begin, tc2.pos_end + 1))
        common_positions = tc1_positions.intersection(tc2_positions)
        return (float(len(common_positions)) / max(len(tc1_positions), len(tc2_positions)) >= match_ratio)

    @staticmethod
    def extract_tag_components_sequences(token_sequences, tag_sequences):
        tag_components_sequences = list()
        for tokens, tags in zip(token_sequences, tag_sequences):
            tag_components = list()
            # First tag component definitely contains the first token and it's tag class name
            #print(0, tokens[0], tags[0])
            tc = TagComponent(pos_begin=0, tag=tags[0])
            tc.add_token(tokens[0])
            # Iterating over all the rest tokens/tags, starting from the second pair
            for k in range(1, len(tokens)):
                #print(k, tokens[k], tags[k])
                if not tc.has_same_tag_class(tags[k]): # previous tag component has the end
                    tag_components.append(tc)
                    tc = TagComponent(pos_begin=tc.pos_end+1, tag=tags[k])
                tc.add_token(tokens[k])
            # Adding the last token
            tag_components.append(tc)
            #for t in tag_components: t.print()
            tag_components_sequences.append(tag_components)
        return tag_components_sequences

    @staticmethod
    def extract_tag_components_sequences_idx(token_sequences_idx, tag_sequences_idx, sequences_indexer):
        token_sequences = sequences_indexer.token2idx(token_sequences_idx)
        tag_sequences = sequences_indexer.tag2idx(tag_sequences_idx)
        return TagComponent.extract_tag_components_sequences(token_sequences, tag_sequences)
