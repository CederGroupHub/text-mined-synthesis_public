import itertools
import os

from .vocabulary import Vocabulary


class TokenStorage(object):
    def feed(self, tokens, label=None):
        """
        Feed a document into the storage.

        :param label: Label of this document.
        :type label: str
        :param tokens: list of tokens.
        :type tokens: list os str
        :return: None
        """
        raise NotImplementedError()


class TokenReader(object):
    def next(self):
        """
        Read the next document.
        :return: tuple(str, list of str)
        """
        raise NotImplementedError()


class LabeledDocumentsReader(TokenReader):
    def __init__(self, input_filename):
        """
        Refer to LabeledDocumentsStorage. This class does the reverse
        work.

        :param input_filename: Filename.
        """
        self.input_filename = input_filename
        self.input_file = None
        self.total_bytes = 0
        self.current_bytes = 0

    def __enter__(self):
        self.input_file = open(self.input_filename)
        self.input_file.seek(0, os.SEEK_END)
        self.total_bytes = self.input_file.tell()
        self.input_file.seek(0, os.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.input_file.close()

    def next(self):
        line = self.input_file.readline().strip()
        self.current_bytes = self.input_file.tell()
        if not line:
            return None

        label, tokens = line.split('\t', maxsplit=1)
        tokens = tokens.split(' ')
        return label, tokens

    def __next__(self):
        n = self.next()
        if n is None:
            raise StopIteration()
        return n

    def __iter__(self):
        return self


class LabeledDocumentsStorage(TokenStorage):
    def __init__(self, output_filename):
        """
        LabeledDocumentsStorage is a storage class that
        saves a corpus by putting each document into one
        line, where the tokens are separated by a whitespace,
        and proceeded by a label and a tab.

        If label in feed() is None, then a default generator will
        be used, and documents are labeled with an integer starting
        from 1.

        :param output_filename: The filename of the output file.
        :type output_filename: str
        """
        self.output_filename = output_filename
        self.output_file = None

        self.doc_counter = itertools.count(1)

    def __enter__(self):
        self.output_file = open(self.output_filename, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output_file.close()

    def feed(self, tokens, label=None):
        doc_id = next(self.doc_counter)
        label = label or str(doc_id)

        self.output_file.write(label)
        self.output_file.write('\t')
        self.output_file.write(' '.join(tokens))
        self.output_file.write('\n')


class LibSVMStorage(TokenStorage):
    def __init__(self, libsvm_filename, dictionary_filename):
        """
        LibSVMStorage is a class that saves corpus into a LibSVM format.
        <label>\t<index1>:<value1> <index2>:<value2> ...

        :param libsvm_filename: Filename of the LibSVM file.
        :type libsvm_filename: str
        :param dictionary_filename: Filename of the dictionary file.
        :type dictionary_filename: str
        """
        self.libsvm_filename = libsvm_filename
        self.dictionary_filename = dictionary_filename
        self.doc_counter = itertools.count(1)
        self.vocab = Vocabulary()

    def __enter__(self):
        self.libsvm_file = open(self.libsvm_filename, 'w')
        self.dictionary_file = open(self.dictionary_filename, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.libsvm_file.close()
        self.vocab.save(self.dictionary_file)
        self.dictionary_file.close()

    def feed(self, tokens, label=None):
        doc_id = next(self.doc_counter)
        label = label or str(doc_id)

        self.libsvm_file.write(label)
        self.libsvm_file.write('\t')
        features = {}
        for token in tokens:
            tokenid = self.vocab[token]
            self.vocab.count(tokenid)
            if tokenid not in features:
                features[tokenid] = 1
            else:
                features[tokenid] += 1

        line = ['%d:%d' % (feature, count) for (feature, count) in features.items()]
        self.libsvm_file.write(' '.join(line))
        self.libsvm_file.write('\n')
