"""f1-micro averaging evaluator for tag components, spans detection + classification, uses standard CoNNL perl script"""
import os
import random
import time
from src.data_io.data_io_connl_ner_2003 import DataIOConnlNer2003
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorF1MicroSpansConnl(EvaluatorBase):
    """EvaluatorF1Connl is f1-micro averaging evaluator for tag components, standard CoNNL perl script."""
    def get_evaluation_score(self, targets_tag_sequences, outputs_tag_sequences, word_sequences):
        fn_out = 'out_temp_%04d.txt' % random.randint(0, 10000)
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        data_io_connl_2003 = DataIOConnlNer2003()
        data_io_connl_2003.write_data(fn_out, word_sequences, targets_tag_sequences, outputs_tag_sequences)
        cmd = 'perl %s < %s' % (os.path.join('.', 'conlleval'), fn_out)
        msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
        msg += ''.join(os.popen(cmd).readlines())
        time.sleep(0.5)
        if fn_out.startswith('out_temp_') and os.path.exists(fn_out):
            os.remove(fn_out)
        f1 = float(msg.split('\n')[3].split(':')[-1].strip())
        return f1, msg
