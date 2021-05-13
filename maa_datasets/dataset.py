# ======================================= #
# --------------- DataModel ------------- #
# ======================================= #
import os
import csv
import sys

from maa_datasets.utils import InputExample

from maa_datasets.utils import SentenceProcessor


class IMDB(SentenceProcessor):
    NAME = 'IMDB'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.train.txt.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.dev.txt.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.test.txt.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP_13(SentenceProcessor):
    NAME = 'YELP_13'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        super().__init__()
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP_14(SentenceProcessor):
    NAME = 'YELP_14'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sentences(self):
        train = self._create_sentences(self.d_train)
        dev = self._create_sentences(self.d_dev)
        test = self._create_sentences(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)

class SST2(object):
    NAME = 'SST2'
    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-train'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-dev'))
        self.d_test = self._read_file(os.path.join(data_dir, 'sst2', 'sentiment-test'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def _read_file(self, dataset):
        with open(dataset, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            # text = [sentence for sentence in split_sents(line[2])]
            text = line[0]
            if i % 5000 == 0:
                print(text)
            examples.append(
                InputExample(guid=guid, user=None, product=None, text=text, label=int(line[1])))
        return examples


class SST5(object):
    NAME = 'SST5'
    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'sst5', 'train.txt'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'sst5', 'dev.txt'))
        self.d_test = self._read_file(os.path.join(data_dir, 'sst5', 'test.txt'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def _read_file(self, dataset):
        with open(dataset, "r") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = [self.fromtree(line) for line in f]
            return examples

    def fromtree(self, data):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        tree = Tree.fromstring(data)
        text = ' '.join(tree.leaves())
        label = tree.label()
        return text, label

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            # text = [sentence for sentence in split_sents(line[2])]
            text = line[0].lower()
            if i % 5000 == 0:
                print(text)
            examples.append(
                InputExample(guid=guid, user=None, product=None, text=text, label=int(line[1])))
        return examples