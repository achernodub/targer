"""
.. module:: Report
    :synopsis: Report stores evaluation results as text files.

.. moduleauthor:: Artem Chernodub
"""

class Report():
    def __init__(self, fn, args, score_name='f1'):
        self.fn = fn
        self.args = args
        self.score_name = score_name
        self.text = 'Evaluation, %s scores.\n\n' % score_name
        self.text += '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        self.text += '\n\n %5s | %5s | %5s | %5s' % ('epoch', 'train', 'dev', 'test')
        self.text += '\n' + '-' * 40
        self.__save()

    def write_epoch_scores(self, epoch, score_train, score_dev, score_test):
        self.text += '\n %5s | %5s | %5s | %5s' % \
                     ('%d'% epoch, '%1.2f' % score_train, '%1.2f' % score_dev, '%1.2f' % score_test)
        self.__save()

    def write_final_score(self, f1_test_final):
        self.text += '\n' + '-' * 40
        self.text += '\n Final eval on test: %s = %1.2f' % (self.score_name, f1_test_final)
        self.__save()

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)
