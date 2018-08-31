import argparse

from classes.evaluator import Evaluator
from classes.data_io import DataIO

'''
The script is based on paper: Eger S., Daxenberger J., Gurevych I. Neural end-to-end learning for computational
argumentation mining //arXiv preprint arXiv:1704.06104. – 2017.
It reproduces these scripts: https://github.com/UKPLab/acl2017-neural_end2end_am/tree/master/progs/Eval
Output file was also taken from this repository.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculation of F1 score for CoNNL-based BIO-tagging scheme')
    parser.add_argument('--target', default='data/persuasive_essays/Essay_Level/test.dat.abs')
    parser.add_argument('--output', default='data/persuasive_essays/Essay_Level/f1_test/maj_best10.out.corr.abs')
    parser.add_argument('--match_alpha_ratio', type=float, default=0.999, help='Alpha ratio from non-strict matching, options: 0.999 or 0.5')
args = parser.parse_args()
_, target_tag_sequences = DataIO.read_CoNNL_dat_abs(args.target)
_, output_tag_sequences = DataIO.read_CoNNL_dat_abs(args.output)

F1, Precision, Recall, (TP, FP, FN) = Evaluator.get_f1_components_from_words(targets_tag_sequences=target_tag_sequences,
                                                                             outputs_tag_sequences=output_tag_sequences,
                                                                             match_alpha_ratio=args.match_alpha_ratio)
print('target file = "%s", output file = "%s", match_alpha_ratio = %1.3f' % (args.target, args.output,
                                                                             args.match_alpha_ratio))
print('# TP = %d, FP = %d, FN = %d, F1 = %1.2f' % (TP, FP, FN, F1))
