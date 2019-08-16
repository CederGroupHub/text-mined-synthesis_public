import pickle

import numpy
from sklearn import ensemble
from sklearn.ensemble.forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_array

from .featurizers.base import FeatureExtraction


class SynthesisClassifier(object):
    """
    !WARNING!: Careful when editing this class. You might destroy all the pickle'd classifiers.
    """
    def __init__(self, featurizer_list, lda_sentence_model, lda_paragraph_model, dt_classifier=None):

        self.featurizer_list = [x() for x in featurizer_list]
        self.lda_sentence_model = lda_sentence_model
        self.lda_paragraph_model = lda_paragraph_model

        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()

        if dt_classifier is None:
            self.dt_classifier = ensemble.RandomForestClassifier(oob_score=True)
        else:
            self.dt_classifier = dt_classifier
        self.model_type = 'RandomForest'
        self.training_data_set = None  # if None, then model is empty

    def check_oob(self, x, y):
        n_samples = y.shape[0]
        in_sample_tensor = numpy.zeros(shape=(
            len(self.dt_classifier.estimators_),
            x.shape[0],
        ))
        out_sample_tensor = numpy.zeros(shape=(
            len(self.dt_classifier.estimators_),
            x.shape[0],
        ))

        for i, estimator in enumerate(self.dt_classifier.estimators_):
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            sampled_indices = _generate_sample_indices(
                estimator.random_state, n_samples)

            assert len(set(unsampled_indices) & set(sampled_indices)) == 0

            unsampled_estimated = estimator.predict(x[unsampled_indices, :])
            unsampled_real = y[unsampled_indices]
            sample_estimated = estimator.predict(x[sampled_indices, :])
            sample_real = y[sampled_indices]

            out_sample_success = numpy.where(unsampled_estimated.astype(int) == unsampled_real)
            out_sample_fail = numpy.where(unsampled_estimated.astype(int) != unsampled_real)
            out_sample_success_indices = unsampled_indices[out_sample_success]
            out_sample_fail_indices = unsampled_indices[out_sample_fail]
            out_sample_tensor[i, out_sample_success_indices] = 1.0
            out_sample_tensor[i, out_sample_fail_indices] = -1.0

            in_sample_success = numpy.where(sample_estimated.astype(int) == sample_real)
            in_sample_fail = numpy.where(sample_estimated.astype(int) != sample_real)
            in_sample_success_indices = sampled_indices[in_sample_success]
            in_sample_fail_indices = sampled_indices[in_sample_fail]
            in_sample_tensor[i, in_sample_success_indices] = 1.0
            in_sample_tensor[i, in_sample_fail_indices] = -1.0

        return in_sample_tensor, out_sample_tensor, y

    def train_model(self, training_data, y_to_learn):
        if self.training_data_set is not None:
            raise RuntimeError('Train a existing model is not permitted!')

        features = [{} for x in training_data]

        for featurizer in self.featurizer_list:
            featurizer.learn_features(training_data, y_to_learn)
            for i, j in zip(features, featurizer.featurize(training_data)):
                i.update(j)

        features_vectorized = self.vectorizer.fit_transform(features)
        x = features_vectorized.toarray()

        labels = [x['y'] for x in training_data]
        y = self.label_encoder.fit_transform(labels)

        self.dt_classifier.fit(x, y)

        self.training_data_set = {
            'feature_names': self.vectorizer.get_feature_names(),
            'class_names': self.label_encoder.classes_.tolist(),
            'training_data': []
        }
        for t, _x, _y in zip(training_data, x, y):
            self.training_data_set['training_data'].append({
                'doi': t['doi'],
                'paragraph_id': t['paragraph_id'],
                'x': _x.tolist(),
                'y': _y.tolist()
            })

        return self.check_oob(x, y)

    def dump_pickle(self):
        return pickle.dumps(self)

    @staticmethod
    def load_pickle(s):
        """

        :param s:
        :return:
        :rtype: SynthesisClassifier
        """
        return pickle.loads(s)

    def predict(self, data_with_topics, return_decision_path=False):
        features = [{} for x in data_with_topics]
        for featurizer in self.featurizer_list:
            for i, j in zip(features, featurizer.featurize(data_with_topics)):
                i.update(j)

        features_vectorized = self.vectorizer.transform(features)
        x = features_vectorized.toarray()

        for i, j in zip(data_with_topics, self.label_encoder.inverse_transform(self.dt_classifier.predict(x))):
            i['y_predicted'] = j

        if not return_decision_path:
            return data_with_topics
        else:
            indicators, n_nodes_ptr = self.dt_classifier.decision_path(x)

            starting_nodes = n_nodes_ptr.tolist()
            all_paths = []

            for path in indicators:
                path = numpy.where(path.toarray().flatten() != 0)[0].tolist()

                paths = []
                current_path = [path.pop(0)]
                current_weight = 0
                while path:
                    if path[0] in starting_nodes:
                        paths.append(current_path)
                        current_weight = path[0]
                        current_path = []
                    current_path.append(path.pop(0) - current_weight)
                if current_path:
                    paths.append(current_path)

                decision_path = []
                for n, p in enumerate(paths):
                    classes = self.dt_classifier.estimators_[n].tree_.value[p[-1]]
                    final_vote = self.label_encoder.classes_[classes.argmax()]

                    decision_path.append({
                        'decision_tree_id': n,
                        'path_seq': p,
                        'voted': final_vote
                    })

                all_paths.append(decision_path)
            return data_with_topics, all_paths
