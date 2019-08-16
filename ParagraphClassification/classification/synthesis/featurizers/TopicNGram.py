import collections
import itertools
import logging
import operator
from functools import reduce

from .base import FeatureExtraction


class SentenceTopicBiGram(FeatureExtraction):
    """
    Topic N grams.
    """
    ngram = 2
    minimum_freq = 10
    threshold = 0.1
    maximum_frac_topics = 0.4
    maximum_number_topics = 30

    def __init__(self, *args, **kwargs):
        super(SentenceTopicBiGram, self).__init__(*args, **kwargs)

        self.selected_ngrams = None

    def find_ngrams(self, input_list):
        for topic_seq in zip(*[input_list[i:] for i in range(self.ngram)]):
            cleaned_topic_seq = []
            for (start, end), ts in topic_seq:
                r = []
                for t, v in ts.items():
                    if v > self.threshold:
                        r.append((t, v))
                cleaned_topic_seq.append(r)

            for x in itertools.product(*cleaned_topic_seq):
                t, v = zip(*x)
                yield (tuple(t), reduce(operator.__mul__, v))

    def featurize(self, samples):
        featurized = []
        for sample in samples:
            topic_seq = sample['sentence_topics']
            featurized_this_doc = {}

            for ngram, value in self.find_ngrams(topic_seq):
                if ngram in self.selected_ngrams:
                    featurized_this_doc['high_freq_topic_ngram_%r' % (ngram,)] = value

            featurized.append(featurized_this_doc)

        return featurized

    def learn_features(self, samples, targeted_y):
        topic_histogram = collections.defaultdict(int)
        for sample in samples:
            if sample['y'] not in targeted_y:
                continue
            topic_seq = sample['sentence_topics']
            for ngram, value in self.find_ngrams(topic_seq):
                topic_histogram[ngram] += 1

        logging.info('Topic NGram (%d) histogram is: %d', self.ngram, len(topic_histogram))

        topic_by_freq = sorted([(k, v) for k, v in topic_histogram.items()], key=lambda x: x[1], reverse=True)
        total_num_topics = len(topic_by_freq)

        selected_ngrams = []

        for topic, freq in topic_by_freq:
            if freq < self.minimum_freq:
                continue
            if len(selected_ngrams) > self.maximum_frac_topics * total_num_topics or \
                    len(selected_ngrams) > self.maximum_number_topics:
                break
            selected_ngrams.append(topic)

        logging.info('Selecting NGram(%d): %r', self.ngram, selected_ngrams)

        self.selected_ngrams = selected_ngrams

    def report_features(self):
        pass


class SentenceTopicTriGram(SentenceTopicBiGram):
    """
    Topic N grams.
    """
    ngram = 3
    minimum_freq = 8
