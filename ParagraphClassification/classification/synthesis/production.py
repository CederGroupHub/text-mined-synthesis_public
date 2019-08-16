import json

import collections
from chemdataextractor.nlp import ChemSentenceTokenizer
from synthesis_project_ceder.classification.synthesis import SynthesisClassifier
from synthesis_project_ceder.topic import LightLDAInference


class SynthesisClassificationWorker(object):
    def __init__(self, classification_model_path, topic_model_configs):
        self.span_tokenizer = ChemSentenceTokenizer()
        with open(classification_model_path, 'rb') as f:
            self.classification_model = SynthesisClassifier.load_pickle(f.read())

        self.paragraph_model = LightLDAInference(**topic_model_configs[self.classification_model.lda_paragraph_model])
        self.sentence_model = LightLDAInference(**topic_model_configs[self.classification_model.lda_sentence_model])

    def __enter__(self):
        self.paragraph_model.__enter__()
        self.sentence_model.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.paragraph_model.__exit__(exc_type, exc_val, exc_tb)
        self.sentence_model.__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _convert_to_fraction(d):
        result = {}
        total = sum(d.values())
        for _id, val in d.items():
            result[_id] = round(val / total, 3)
        return result

    def _calculate_topics(self, text, repeat):
        trials = [{} for i in range(repeat)]

        # paragraph

        paragraph_topics = self.paragraph_model.infer(text, repeat=repeat)
        for i in range(repeat):
            trials[i]['paragraph_topics'] = self._convert_to_fraction(paragraph_topics[i])

        # sentence

        spans = self.span_tokenizer.span_tokenize(text)
        span_start, span_end, sentence_text, topic_result = [], [], [], []
        for span in spans:
            span_start.append(span[0])
            span_end.append(span[1])
            sentence_text.append(text[span[0]:span[1]])

        for i in range(repeat):
            trials[i]['sentence_topics'] = []

        sentence_topics = self.sentence_model.infer(sentence_text, repeat=repeat)
        for i, (a, b) in enumerate(zip(span_start, span_end)):
            for j in range(repeat):
                trials[j]['sentence_topics'].append(
                    (
                        (a, b),
                        self._convert_to_fraction(sentence_topics[i*repeat+j]))
                )

        return trials

    def classify_paragraph(self, paragraph, restart=5):
        repeated_documents = self._calculate_topics(paragraph, restart)

        predictions, decision_paths = self.classification_model.predict(
            repeated_documents,
            return_decision_path=True
        )

        classification_trials = []
        all_voted = []

        for p, d in zip(predictions, decision_paths):
            trial = {
                'paragraph_topics': json.dumps(p['paragraph_topics']),
                'sentence_topics': json.dumps(p['sentence_topics'])
            }

            decisions = collections.defaultdict(list)
            for i in sorted(d, key=lambda x: x['decision_tree_id']):
                decisions[i['voted']].append('%d:%d' % (i['decision_tree_id'], i['path_seq'][-1]))
            decisions = {x: ';'.join(y) for x, y in decisions.items()}
            trial['_decision_path'] = decisions

            sentence_borders = [x[0] for x in p['sentence_topics']]
            if len(sentence_borders) <= 1:
                for i in d:
                    all_voted.append('something_else')
                trial['_decision_path']['remarks'] = 'Classification stopped by the limit on sentence size.'
            else:
                voted = [x['voted'] for x in d]
                all_voted += voted

            classification_trials.append(trial)

        all_predictions = collections.Counter(all_voted)
        all_predictions = {x: float(y) / len(all_voted) for x, y in all_predictions.items()}
        y_predicted, confidence = max(all_predictions.items(), key=lambda x: x[1])

        result = {
            'predictions': all_predictions,
            'trials': classification_trials
        }

        return y_predicted, confidence, result
