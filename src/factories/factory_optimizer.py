"""creates various optimizers"""
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class OptimizerFactory():
    """OptimizerFactory contains wrappers to create various optimizers."""
    @staticmethod
    def create(args, tagger):
        if args.opt == 'sgd':
            optimizer = optim.SGD(list(tagger.parameters()), lr=args.lr, momentum=args.momentum)
        elif args.opt == 'adam':
            optimizer = optim.Adam(list(tagger.parameters()), lr=args.lr, betas=(0.9, 0.999))
        else:
            raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch))
        return optimizer, scheduler
