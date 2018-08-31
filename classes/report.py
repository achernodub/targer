from classes.utils import write_textfile

class Report():
    def __init__(self, fn, args):
        self.fn = fn
        self.text = 'Evaluation, micro-f1 scores.\n\n'
        self.text += '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        self.text += '\n\n %5s | %5s | %5s | %5s' % ('epoch', 'train', 'dev', 'test')
        self.text += '\n' + '-' * 40
        self.__save()

    def __save(self):
        write_textfile(self.report_fn, self.text)

    def write_epoch_scores(self, epoch, f1_train, f1_dev, f1_test)
        self.text += '\n %5s | %5s | %5s | %5s' % ('%d' % epoch, '%1.2f' % f1_train, '%1.2f' % f1_dev,
                                                         '%1.2f' % f1_test)
        self.__save()

    def write_final_score(self, f1_test_final):
        self.text += '\n' + '-' * 40
        self.text += '\n Final eval on test: micro-f1 = %1.2f' % f1_test_final
        self.__save()
