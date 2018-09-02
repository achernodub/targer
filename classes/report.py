from classes.utils import write_textfile

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

    def __save(self):
        write_textfile(self.fn, self.text)

    def write_epoch_scores(self, epoch, f1_train, f1_dev, f1_test):
        self.text += '\n %5s | %5s | %5s | %5s' % ('%d' % epoch, '%1.2f' % f1_train, '%1.2f' % f1_dev,
                                                         '%1.2f' % f1_test)
        self.__save()

    def write_final_score(self, f1_test_final):
        self.text += '\n' + '-' * 40
        self.text += '\n Final eval on test: %s = %1.2f' % (self.score_name, f1_test_final)
        self.__save()
