"""
.. module:: Report
    :synopsis: Report stores evaluation results as text files.

.. moduleauthor:: Artem Chernodub
"""

class Report():
    def __init__(self, fn, args, score_names):
        self.fn = fn
        self.args = args
        self.score_num = len(score_names)
        self.text = 'Evaluation\n\n'
        self.text += '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        header = '\n\n %10s |' % 'epoch '
        for n, score_name in enumerate(score_names):
            header += ' %10s ' % score_name
            if n < len(score_names) - 1: header += '|'
        self.text += header
        self.text += '\n' + '-' * len(header)

    def write_epoch_scores(self, epoch, scores):
        self.text += '\n %10s |' % ('%d'% epoch)
        for n, score in enumerate(scores):
            self.text += ' %10s ' % ('%1.2f' % score)
            if n < len(scores) - 1: self.text += '|'
        self.__save()

    def write_final_score(self, f1_test_final):
        self.text += '\n' + '-' * 40
        self.text += '\n Final eval on test: %s = %1.2f' % (self.score_name, f1_test_final)
        self.__save()

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)
