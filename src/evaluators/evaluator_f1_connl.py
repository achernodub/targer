import os
import random
import time
from src.classes.data_io import DataIO
from src.evaluators.evaluator_base import EvaluatorBase

class EvaluatorF1Connl(EvaluatorBase):
    @staticmethod
    def get_evaluation_score(targets_tag_sequences, outputs_tag_sequences, word_sequences):
        fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        DataIO.write_CoNNL_2003_two_columns(fn_out, word_sequences, targets_tag_sequences, outputs_tag_sequences)
        cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        str = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        str += ''.join(os.popen(cmd).readlines())
        time.sleep(0.5)
        if fn_out.startswith('out_temp_') and os.path.exists(fn_out):
            os.remove(fn_out)
        f1 = float(str.split('\n')[3].split(':')[-1].strip())
        return f1 #, str
