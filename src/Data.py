from utils.data import samples_from_conll, get_tags
from utils.ud import get_num_rel, get_num_upos
from pyconll import load_from_file


class Data:
    """
    Data contains a set of dependency tree, pos tags, feature tags samples
    """
    def __init__(self, train, dev, test, use_v1, hack_v2):
        """
        Initializer
        :param train: file name of training set
        :param dev: file name of development set
        :param test: file name of test set
        :param use_v1: True if sentence is annotated using UD V1.2
        """
        self.samples = []
        train_conll = load_from_file(train)
        dev_conll = load_from_file(dev)
        test_conll = load_from_file(test)
        self.train = samples_from_conll(train_conll, use_v1, hack_v2)
        self.dev = samples_from_conll(dev_conll, use_v1, hack_v2)
        self.test = samples_from_conll(test_conll, use_v1, hack_v2)
        self.tags = get_tags(self.train + self.dev + self.test)

        self._num_pos = get_num_upos(use_v1)
        self._num_labels = get_num_rel(use_v1)

    def num_tags(self):
        return len(self.tags)

    def num_pos(self):
        return self._num_pos

    def num_labels(self):
        return self._num_labels

    """
    Iterator
    """

    def __iter__(self):
        self.sample_id = 0
        return self

    def __next__(self):
        if self.sample_id < len(self.train):
            sample = self.train[self.sample_id]
            self.sample_id += 1
            return sample
        else:
            raise StopIteration
