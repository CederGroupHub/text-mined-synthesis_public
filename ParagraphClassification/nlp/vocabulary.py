import functools

import re


class Vocabulary(object):
    def __init__(self):
        """
        A vocabulary object encodes words as integers.
        All index starts from 1. If a word is not in
        the vocabulary, it will automatically be added.

        It can be accessed as following:

        word_id = vocab['start']  # gives 1
        word = vocab[1]           # gives 'start'

        Word count can be accessed from self.count() method.
        """
        self.word2id = {}
        self.id2word = {}

        self.word_count = {}

    def size(self):
        return len(self.word2id)

    def count(self, item):
        if isinstance(item, str):
            if item not in self.word2id:
                raise ValueError('Word %s not in vocabulary' % item)
            item = self.word2id[item]
        elif isinstance(item, int):
            if item not in self.id2word:
                raise ValueError('Word with id %d not in vocabulary' % item)
        else:
            raise ValueError('word must be str or int')

        if item not in self.word_count:
            self.word_count[item] = 1
        else:
            self.word_count[item] += 1

    def add_word(self, word):
        if re.search(r'\s', word):
            raise ValueError('word cannot contain any spaces!')

        if word in self.word2id:
            return

        word_id = self.size() + 1
        self.word2id[word] = word_id
        self.id2word[word_id] = word

    def __getitem__(self, item):
        if isinstance(item, str):
            if item not in self.word2id:
                self.add_word(item)
            return self.word2id[item]
        elif isinstance(item, int):
            if item not in self.id2word:
                raise ValueError('No such word with id %d' % item)
            return self.id2word[item]
        else:
            raise ValueError('item must be either a str or an int.')

    def save(self, f):
        """
        Save the vocabulary to file.

        :param f: Filename or file object.
        :type f: str or file
        :return: None
        """
        if isinstance(f, str):
            file_ = open(f, 'w')
        else:
            file_ = f

        for wordid in sorted(self.id2word):
            word = self.id2word[wordid]
            wordcount = self.word_count.get(wordid, 0)
            file_.write('%s\t%s\t%d\n' % (wordid, word, wordcount))

        if isinstance(f, str):
            file_.close()

    @staticmethod
    def load(f):
        """
        Load the vocabulary from file.

        :param f: Filename or file object.
        :type f: str or file
        :return: Vocabulary
        """
        return load_vocabulary(f)


@functools.lru_cache(maxsize=6)
def load_vocabulary(f):
    """
    Load the vocabulary from file.

    :param f: Filename or file object.
    :type f: str or file
    :return: Vocabulary
    """
    v = Vocabulary()

    if isinstance(f, str):
        file_ = open(f, 'r')
    else:
        file_ = f

    for line in file_:
        wordid, word, wordcount = line.strip().split('\t')
        wordid, wordcount = int(wordid), int(wordcount)
        v.id2word[wordid] = word
        v.word2id[word] = wordid
        if wordcount != 0:
            v.word_count[wordid] = wordcount

    if isinstance(f, str):
        file_.close()
    return v
