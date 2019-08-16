import collections
import logging
import multiprocessing.pool
import os
import pickle
import random

import numpy
from sklearn import ensemble
from sklearn.metrics import f1_score, precision_score, recall_score
from synthesis_project_ceder.classification.synthesis import SentenceHighFreqTopics, SentenceTopicBiGram, \
    SentenceTopicTriGram, SynthesisClassifier
from synthesis_project_ceder.classification.synthesis.featurizers.ParagraphHighFreqTopics import ParagraphHighFreqTopics
from synthesis_project_ceder.database import CederMT
from synthesis_project_ceder.nlp.preprocessing import TextPreprocessor
from tqdm import tqdm

from synthesis_api_hub import Client

paragraph_model = 'lightlda_r0_paragraph_topic_100'
sentence_model = 'lightlda_r0_sentence_topic_50'


def retrieve_paragraphs():
    import textacy

    logging.info('Obtaining paragraphs')
    db = CederMT()
    training_collection = db.SynthesisTypeFilteringData
    syn_20170926 = db.syn_20170926

    paragraphs = list(
        training_collection.find(
            {
                'is_training_data': True, 'human_validated': True
            },
            {'_id': 0, 'time_created': 0}
        )
    )

    for paragraph in tqdm(paragraphs):
        paper = syn_20170926.find_one(
            {
                'doi': paragraph['doi'],
            },
            {
                '_id': 1,
                'paragraphs': {
                    '$slice': [paragraph['paragraph_id'], 1]
                }
            })
        text = paper['paragraphs'][0]['text']
        paragraph['text'] = textacy.preprocess_text(text, fix_unicode=True)

    return paragraphs


def infer_topics_for_paragraph(paragraph):
    # singleton
    if not hasattr(infer_topics_for_paragraph, 'client'):
        client = Client('synthesisproject.lbl.gov', 8005, 'topic')
        client.connect_server()
        setattr(infer_topics_for_paragraph, 'client', client)

    def to_fraction(d):
        return {x: y / sum(d.values()) for x, y in d.items()}

    sentence_boundaries = []
    sentences = []

    doc = TextPreprocessor(paragraph['text'], textacy=False).doc.user_data
    for sentence in doc.sentences:
        sentences.append(sentence.text)
        sentence_boundaries.append((sentence.start, sentence.end))

    client = getattr(infer_topics_for_paragraph, 'client')
    paragraph_topics, _ = client.infer_topics([paragraph['text']], paragraph_model)
    paragraph_topics = paragraph_topics[0]
    sentence_topics, _ = client.infer_topics(sentences, sentence_model)

    p = to_fraction(paragraph_topics)
    s = [(b, to_fraction(st)) for b, st in zip(sentence_boundaries, sentence_topics)]

    paragraph.update({
        'paragraph_topics': p,
        'sentence_topics': s,
    })

    return paragraph


def retrieve_topics_use_inferer(paragraphs):
    # singleton
    if not hasattr(retrieve_topics_use_inferer, 'pool'):
        pool = multiprocessing.pool.Pool(
            processes=10
        )
        setattr(retrieve_topics_use_inferer, 'pool', pool)
    pool = getattr(retrieve_topics_use_inferer, 'pool')

    return list(pool.imap(infer_topics_for_paragraph, tqdm(paragraphs)))


def retrieve_topics(paragraphs):
    topic_collection = CederMT().get_collection('TopicModelResults')

    def to_fraction(d):
        return {x: y / sum(d.values()) for x, y in d.items()}

    paragraphs_with_topics = []

    for paragraph in tqdm(paragraphs):
        topic_object = topic_collection.find_one(
            {'doi': paragraph['doi'], 'paragraph_id': paragraph['paragraph_id']},
            {paragraph_model: 1, sentence_model: 1}
        )
        if paragraph_model not in topic_object or sentence_model not in topic_object:
            logging.warning('DOI: %s, paragraph id: %d does not have topic information',
                            paragraph['doi'], paragraph['paragraph_id'])
        else:
            paragraph_topics = pickle.loads(topic_object[paragraph_model])
            sentence_topics = pickle.loads(topic_object[sentence_model])

            p = to_fraction(paragraph_topics)
            s = [(b, to_fraction(st)) for b, st in sentence_topics.items()]

            paragraph.update({
                'paragraph_topics': p,
                'sentence_topics': s,
            })

            paragraphs_with_topics.append(paragraph)

    return paragraphs_with_topics


def train_new_model(training_paragraphs, target_labels):
    for i in training_paragraphs:
        if i['type'] not in target_labels:
            i['y'] = 'something_else'
        else:
            i['y'] = i['type']

    y_to_learn = target_labels

    classifier = SynthesisClassifier(
        featurizer_list=[
            SentenceHighFreqTopics,
            SentenceTopicBiGram,
            SentenceTopicTriGram,
            ParagraphHighFreqTopics
        ],
        lda_sentence_model=sentence_model,
        lda_paragraph_model=paragraph_model,
        dt_classifier=ensemble.RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            oob_score=True,
        )
    )
    in_sample_tensor, out_sample_tensor, labels = classifier.train_model(training_paragraphs, y_to_learn)
    # sort according to how many times they are predicted wrongly
    in_sample_indices = numpy.sum(in_sample_tensor, axis=0).argsort()
    out_sample_indices = numpy.sum(out_sample_tensor, axis=0).argsort()
    in_sample_tensor = in_sample_tensor[:, in_sample_indices]
    out_sample_tensor = out_sample_tensor[:, out_sample_indices]

    # for i in in_sample_indices[:20]:
    #     print(training_paragraphs[i]['text'], training_paragraphs[i]['type'])

    import matplotlib.pyplot as plt

    # the first plots the number of failures for all training elements
    # the second plots again the failures/successes for all elements
    # the third plots the distribution of labels for all elements
    fig, ax = plt.subplots(3, 1)
    fail_counts = numpy.zeros((len(training_paragraphs), ))
    for i in range(len(training_paragraphs)):
        fail_counts[i] = in_sample_tensor[:, i].tolist().count(-1.)
    ax[0].scatter(range(len(training_paragraphs)), fail_counts, s=1)
    ax[0].set_xlim([0, len(training_paragraphs)])
    label_dist = numpy.zeros((len(classifier.label_encoder.classes_), len(training_paragraphs)))
    for i, j in enumerate(labels[in_sample_indices]):
        label_dist[j, i] = 1.0
    im = ax[1].imshow(in_sample_tensor, aspect='auto', cmap='bwr')
    # plt.colorbar(im, ax=ax[0])
    im = ax[2].imshow(label_dist, aspect='auto', cmap='Blues')
    ax[2].set_yticks(range(len(classifier.label_encoder.classes_)))
    ax[2].set_yticklabels([x[:5] for x in classifier.label_encoder.classes_])
    ax[0].set_title('In sample result')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(3, 1)
    fail_counts = numpy.zeros((len(training_paragraphs),))
    for i in range(len(training_paragraphs)):
        fail_counts[i] = out_sample_tensor[:, i].tolist().count(-1.)
    ax[0].scatter(range(len(training_paragraphs)), fail_counts, s=1)
    label_dist = numpy.zeros((len(classifier.label_encoder.classes_), len(training_paragraphs)))
    for i, j in enumerate(labels[out_sample_indices]):
        label_dist[j, i] = 1.0
    im = ax[1].imshow(out_sample_tensor, aspect='auto', cmap='bwr')
    # plt.colorbar(im, ax=ax[0])
    im = ax[2].imshow(label_dist, aspect='auto', cmap='Blues')
    ax[2].set_yticks(range(len(classifier.label_encoder.classes_)))
    ax[2].set_yticklabels([x[:5] for x in classifier.label_encoder.classes_])

    ax[0].set_title('Out sample result')
    plt.tight_layout()
    plt.show()

    y_predicted = [x['y_predicted'] for x in classifier.predict(training_paragraphs)]
    y_real = [x['y'] for x in training_paragraphs]

    logging.info('Targets: %r', y_to_learn)
    logging.info(
        'Precision: %r, Recall: %r, F1: %r',
        precision_score(y_real, y_predicted, labels=y_to_learn, average=None),
        recall_score(y_real, y_predicted, labels=y_to_learn, average=None),
        f1_score(y_real, y_predicted, labels=y_to_learn, average=None)
    )
    logging.info(
        'OOB_Score: %r',
        classifier.dt_classifier.oob_score_
    )

    return classifier


def predict_for_paragraph(model, input_paragraphs):
    paragraphs = []
    for p in input_paragraphs:
        paragraphs.append({'text': p['text']})

    paragraphs_with_topics = retrieve_topics(paragraphs)
    predictions, paths = model.predict(paragraphs_with_topics, return_decision_path=True)

    results = []
    for paragraph, prediction, path in zip(input_paragraphs, predictions, paths):
        y_predicted = prediction['y_predicted']
        voted = collections.Counter(x['voted'] for x in path)
        voted = {x: y / sum(voted.values()) for x, y in voted.items()}

        result = dict(paragraph)
        result.update({
            'y_predicted': y_predicted,
            'confidences': voted
        })
        results.append(result)

    return results


def load_training_data(data_filename):
    if data_filename is not None and os.path.exists(data_filename):
        with open(data_filename, 'rb') as f:
            training_paragraphs = pickle.load(f)
    else:
        training_paragraphs = retrieve_topics(retrieve_paragraphs())
        random.seed(42)
        random.shuffle(training_paragraphs)

        if data_filename is not None:
            with open(data_filename, 'wb') as f:
                pickle.dump(training_paragraphs, f)

    return training_paragraphs


def load_model(model_path, training_data):
    if model_path is not None and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_new_model(
            training_data,
            target_labels=[
                'solid_state_ceramic_synthesis',
                'hydrothermal_ceramic_synthesis',
                'sol-gel_ceramic_synthesis',
                # 'precipitation_ceramic_synthesis'
            ]
        )
        if model_path is not None:
            with open(model_path, 'wb') as f:
                f.write(model.dump_pickle())

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build new paragraph classifier.')
    parser.add_argument('--verbose', action='store_true', default=False, help='More debugging logging.')
    parser.add_argument('--data-file', action='store', type=str, help='Training data filename.')
    parser.add_argument('--model-file', action='store', type=str, help='Model filename.')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s(PID-%(process)d) - %(levelname)s - %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    training_paragraphs = load_training_data(args.data_file)
    training_paragraphs = [x for x in training_paragraphs if x['type'] != 'precipitation_ceramic_synthesis']
    logging.info('We have %d paragraphs now: %r',
                 len(training_paragraphs), collections.Counter(x['type'] for x in training_paragraphs))

    model = load_model(args.model_file, training_paragraphs)
    return model


if __name__ == '__main__':
    main()
