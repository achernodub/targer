"""abstract base class for f1-micro averaging evaluation for tag components, spans detection + classification"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorF1MicroSpansAlphaMatchBase(EvaluatorBase):
    """
    EvaluatorF1MicroSpansAlphaMatchBase is an abstract base class for f1-micro averaging evaluator for tag components
    Isaac Persing and Vincent Ng. End-to-end argumentation mining in student essays. NAACL 2016.
    http://www.aclweb.org/anthology/N16-1164.
    """
    def __init__(self, match_alpha_ratio):
        self.match_alpha_ratio = match_alpha_ratio

    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences=None):
        targets_tag_components_sequences = TagComponent.extract_tag_components_sequences(targets_tag_sequences)
        outputs_tag_components_sequences = TagComponent.extract_tag_components_sequences(outputs_tag_sequences)
        f1, precision, recall, (TP, FP, FN) = self.__get_f1_components_from_sequences(targets_tag_components_sequences,
                                                                                      outputs_tag_components_sequences)
        msg = '*** f1 alpha match, alpha = %1.1f' % self.match_alpha_ratio
        msg += '\n*** f1 = %1.2f, precision = %1.2f, recall = %1.2f' % (f1, precision, recall)
        msg += '\n*** TP = %d, FP = %d, FN = %d' % (TP, FP, FN)
        return f1, msg

    def __get_f1_components_from_sequences(self, targets_tag_components_sequences, outputs_tag_components_sequences):
        TP, FN, FP = 0, 0, 0
        for targets_tag_components, outputs_tag_components in zip(targets_tag_components_sequences,
                                                                  outputs_tag_components_sequences):
            for target_tc in targets_tag_components:
                found = False
                for output_tc in outputs_tag_components:
                    if output_tc.is_equal(target_tc, self.match_alpha_ratio):
                        found = True
                        break
                if found:
                    TP += 1
                else:
                    FN += 1
            for output_tc in outputs_tag_components:
                found = False
                for target_tc in targets_tag_components:
                    if target_tc.is_equal(output_tc, self.match_alpha_ratio):
                        found = True
                        break
                if not found:
                    FP += 1
        precision = (TP / max(TP + FP, 1))*100
        recall = (TP / max(TP + FN, 1))*100
        f1 = (2 * TP / max(2 * TP + FP + FN, 1))*100
        return f1, precision, recall, (TP, FP, FN)


class TagComponent():
    def __init__(self, pos_begin, tag):
        self.pos_begin = pos_begin
        self.tag_class_name = TagComponent.get_tag_class_name(tag)
        self.pos_end = self.pos_begin - 1
        self.words = list()

    def has_same_tag_class(self, tag):
        return self.tag_class_name == TagComponent.get_tag_class_name(tag)

    def add_word(self, word):
        self.words.append(word)
        self.pos_end += 1

    def is_equal(self, tc, match_alpha_ratio):
        return TagComponent.match(self, tc, match_alpha_ratio)

    def print(self):
        print('--tag_class_name = %s, pos_begin = %s, pos_end = %s' % (self.tag_class_name, self.pos_begin,
                                                                       self.pos_end))
        word_str = '    '
        for word in self.words: word_str += word + ' '
        print(word_str, '\n')

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
        return float(len(common_positions)) / max(len(tc1_positions), len(tc2_positions)) >= match_ratio

    @staticmethod
    def extract_tag_components_sequences_debug(word_sequences, tag_sequences):
        tag_components_sequences = list()
        for words, tags in zip(word_sequences, tag_sequences):
            tag_components = list()
            # First tag component definitely contains the first word and it's tag class name
            #print(0, words[0], tags[0])
            tc = TagComponent(pos_begin=0, tag=tags[0])
            tc.add_word(words[0])
            # Iterating over all the rest words/tags, starting from the second pair
            for k in range(1, len(tags)):
                #print(k, words[k], tags[k])
                if not tc.has_same_tag_class(tags[k]): # previous tag component has the end
                    if tc.tag_class_name != 'O':
                        tag_components.append(tc)
                    tc = TagComponent(pos_begin=tc.pos_end+1, tag=tags[k])
                tc.add_word(words[k])
            # Adding the last word
            if tc.tag_class_name != 'O':
                tag_components.append(tc)
            #for t in tag_components: t.print()
            tag_components_sequences.append(tag_components)
        return tag_components_sequences

    @staticmethod
    def extract_tag_components_sequences(tag_sequences):
        tag_components_sequences = list()
        for tags in tag_sequences:
            tag_components = list()
            tc = TagComponent(pos_begin=0, tag=tags[0])
            tc.add_word('not-debug-mode')
            for k in range(1, len(tags)):
                if not tc.has_same_tag_class(tags[k]): # previous tag component has the end
                    if tc.tag_class_name != 'O':
                        tag_components.append(tc)
                    tc = TagComponent(pos_begin=tc.pos_end+1, tag=tags[k])
                tc.add_word('not-debug-mode')
            # Adding the last word
            if tc.tag_class_name != 'O':
                tag_components.append(tc)
            tag_components_sequences.append(tag_components)
        return tag_components_sequences

    @staticmethod
    def extract_tag_components_sequences_idx(word_sequences_idx, tag_sequences_idx, sequences_indexer):
        word_sequences = sequences_indexer.word2idx(word_sequences_idx)
        tag_sequences = sequences_indexer.tag2idx(tag_sequences_idx)
        return TagComponent.extract_tag_components_sequences_debug(word_sequences, tag_sequences)
