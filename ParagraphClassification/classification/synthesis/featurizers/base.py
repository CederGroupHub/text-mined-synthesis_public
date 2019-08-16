class FeatureExtraction(object):
    def __init__(self, *args, **kwargs):
        pass

    def featurize(self, samples):
        raise NotImplementedError()

    def learn_features(self, samples, targeted_y):
        raise NotImplementedError()

    def report_features(self):
        raise NotImplementedError()
