import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import *

class Evaluator():
    def get_macro_scores_from_inputs_tensor(self, tagger, inputs_tensor, targets_idx):
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        if len(targets_idx) != len(outputs_idx):
            raise ValueError('len(targets_idx) != len(len(outputs_idx))')
        batch_size = len(targets_idx)
        f1_sum = 0
        precision_sum = 0
        recall_sum = 0
        for n in range(batch_size):
            y_true = targets_idx[n]
            y_pred = outputs_idx[n]
            f1_sum += f1_score(y_true, y_pred, average='macro')
            precision_sum += precision_score(y_true, y_pred, average='macro')
            recall_sum += recall_score(y_true, y_pred, average='macro')
        f1 = f1_sum / batch_size
        precision = precision_sum / batch_size
        recall = recall_sum / batch_size
        return f1, precision, recall