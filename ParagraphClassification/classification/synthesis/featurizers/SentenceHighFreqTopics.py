import collections
import logging

from .base import FeatureExtraction


class SentenceHighFreqTopics(FeatureExtraction):
    """
    Accepts list of true samples, extract most significant features.
    Only for sequence of topics: [[(0, 0.1), ...], ...]
    """

    def __init__(self, *args, **kwargs):
        super(SentenceHighFreqTopics, self).__init__(*args, **kwargs)

        # first feature selection: topics with hightest frequencies
        # Minimum amount of topic to be considered
        self.threshold = 0.1

        # Don't take topics with freq less than this
        self.minimum_freq = 5
        # Don't take too many topics larger than this fraction
        self.maximum_frac_topics = 0.4
        self.maximum_number_topics = 10

        self.selected_topics = None

    def featurize(self, samples):
        featurized = []
        for sample in samples:
            topic_seq = sample['sentence_topics']
            featurized_this_doc = {}
            all_topics_this_doc = {}

            for (start, end), topic_list in topic_seq:
                for topic, amount in topic_list.items():
                    if amount > self.threshold:
                        all_topics_this_doc[topic] = max(amount, all_topics_this_doc.get(topic, 0))

            for topic in self.selected_topics:
                if topic in all_topics_this_doc:
                    featurized_this_doc['high_freq_topic_%d' % topic] = all_topics_this_doc[topic]

            featurized.append(featurized_this_doc)

        return featurized

    def learn_features(self, samples, targeted_y):
        topic_histograms = collections.defaultdict(lambda: collections.defaultdict(int))
        for sample in samples:
            y = sample['y']
            if y not in targeted_y:
                continue

            topic_seq = sample['sentence_topics']
            for (start, end), topic_list in topic_seq:
                for topic, amount in topic_list.items():
                    if amount > self.threshold:
                        topic_histograms[y][topic] += 1

        selected_topics = []

        for y, histogram in topic_histograms.items():
            logging.info('Topic histogram %s is: %r', y, dict(histogram))

            topic_by_freq = sorted([(k, v) for k, v in histogram.items()], key=lambda x: x[1], reverse=True)
            total_num_topics = len(topic_by_freq)

            selected_topics_this = []
            for topic, freq in topic_by_freq:
                if freq < self.minimum_freq:
                    continue
                if len(selected_topics_this) > self.maximum_frac_topics * total_num_topics or \
                        len(selected_topics_this) > self.maximum_number_topics:
                    break
                selected_topics_this.append(topic)

            logging.info('Selecting for %s topics: %r', y, selected_topics_this)

            selected_topics += selected_topics_this

        self.selected_topics = selected_topics

    def report_features(self):
        pass
