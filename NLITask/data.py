from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe
import torch

from nltk import word_tokenize
import numpy as np


class dataset_import():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, include_lengths=True, tokenize=word_tokenize, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        #Here, we suse SNLI, can be replaced by MULTINLI
        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        #Here we use Glove
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_size=args.batch_size,
                                       device=torch.device(args.device))

