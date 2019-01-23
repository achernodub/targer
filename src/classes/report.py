"""stores evaluation results during the training process as text files"""

from src.classes.utils import get_input_arguments

class Report():
    def __init__(self, fn, args, score_names):
        """Report stores evaluation results during the training process as text files."""
        self.fn = fn
        self.args = args
        self.score_num = len(score_names)
        self.text = 'Evaluation\n\n'
        self.text += '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        header = '\n\n %14s |' % 'epoch '
        for n, score_name in enumerate(score_names):
            header += ' %14s ' % score_name
            if n < len(score_names) - 1: header += '|'
        self.text += header
        self.blank_line = '\n' + '-' * len(header)
        self.text += self.blank_line

    def write_epoch_scores(self, epoch, scores):
        self.text += '\n %14s |' % ('%d'% epoch)
        for n, score in enumerate(scores):
            self.text += ' %14s ' % ('%1.2f' % score)
            if n < len(scores) - 1: self.text += '|'
        self.__save()

    def write_final_score(self, final_score_str):
        self.text += self.blank_line
        self.text += '\n%s' % final_score_str
        self.__save()

    def write_msg(self, msg):
        self.text += self.blank_line
        self.text += msg
        self.__save()

    def write_input_arguments(self):
        self.text += '\nInput arguments:\n%s' % get_input_arguments()
        self.__save()

    def write_final_line_score(self, final_score):
        self.text += '\n\n%1.4f' % final_score
        self.__save()

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)

    def make_print(self):
        print(self.text)
