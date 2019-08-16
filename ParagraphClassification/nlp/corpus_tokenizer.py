import itertools

import datetime
import logging

import time
from bson import ObjectId

from . import preprocessing
from ..utils.time_utils import estimate_eta

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"


class CorpusTokenizer(object):
    def __init__(self, document_generator, token_storage, token_filter):
        """
        Tokenize documents in a corpus to individual tokens.

        :param document_generator: A generator that gives documents, or (label, document) tuples.
        :param token_storage: Token storage instance.
        :type token_storage: TokenStorage
        :param token_filter:
        """
        self.document_generator = document_generator
        self.token_storage = token_storage
        self.token_filter = token_filter

    def _feed_storage(self, tokens, label):
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError('toke must be a str object!')
            if ' ' in token:
                raise ValueError('token must not contain whitespace!')
            if '\n' in token:
                raise ValueError('token must not contain newline!')

        self.token_storage.feed(tokens, label=label)

    def _process_sentences(self, sentences, document_label, processor):
        if not sentences:
            return

        orths, lemma, pos = zip(*sentences)
        orths = list(itertools.chain(*orths))
        lemma = list(itertools.chain(*lemma))
        pos = list(itertools.chain(*pos))
        tokens = self.token_filter(orths, lemma, pos)
        if not tokens:
            return

        self._feed_storage(tokens, document_label)

    def _process_document(self, document, label):
        document = document.strip()
        processor = preprocessing.TextPreprocessor(document)
        cde_doc = processor.doc.user_data
        all_lemmas = processor.get_words(lemma=True)

        sentences = []
        for sentence in cde_doc.sentences:
            orths, pos = zip(*sentence.pos_tagged_tokens)
            orths, pos = list(orths), list(pos)
            lemma = all_lemmas[:len(orths)]
            all_lemmas = all_lemmas[len(orths):]

            sentences.append((orths, lemma, pos))

        self._process_sentences(sentences, label, processor)

    def tokenize(self, callback=None):
        for i_doc, document in enumerate(self.document_generator, start=1):
            if isinstance(document, tuple):
                if len(document) != 2:
                    raise ValueError('document generator must yield (label, document) each time, '
                                     'expected size 2, got %d!' % len(document))
                label, document = document
            else:
                label = str(i_doc)

            self._process_document(document, label)

            if callback is not None:
                callback(i_doc, document)


class CorpusSentenceTokenizer(CorpusTokenizer):
    def _process_sentences(self, sentences, document_label, processor):
        cde_doc = processor.doc.user_data
        for (orths, lemma, pos), sentence in zip(sentences, cde_doc.sentences):
            sent_start, sent_end = sentence.start, sentence.end
            tokens = self.token_filter(orths, lemma, pos)
            if not tokens:
                return

            self._feed_storage(tokens, '%s:%d-%d' % (document_label, sent_start, sent_end))


class CorpusToken(object):
    def __init__(self, syn_20170926, destination_collection):
        """Generate a collection of tokenized words from syn_20170926.

        :param syn_20170926: Documents collection.
        :type syn_20170926: pymongo.collection.Collection or None
        :param destination_collection: Destination collection.
        :type destination_collection: pymongo.collection.Collection
        """
        self.syn_20170926 = syn_20170926
        self.destination_collection = destination_collection

        self._logger = logging.getLogger('CorpusToken')

    def is_collection_ready(self):
        """Test if we have a ready to use collection.

        :rtype: bool
        """
        return self.destination_collection.find_one({}) is not None

    def _iter_paragraphs(self, document_id_fn, paragraph_filter, token_filter):
        num_total_docs = self.destination_collection.find().count()
        check_time = time.time()
        start_time = check_time
        n = 0

        doc_id_f = open(document_id_fn, 'w') if document_id_fn is not None else None

        for n, obj in enumerate(self.destination_collection.find()):
            for m, p in enumerate(obj['paragraphs']):

                if paragraph_filter is not None:
                    p = paragraph_filter(p)
                    if p is None:
                        continue

                tokens = []

                for sent in p['sentences']:
                    orth = sent['orth']
                    lemma = sent['lemmas']
                    pos = sent['pos']

                    if token_filter is not None:
                        _tokens = token_filter(orth, lemma, pos)
                    else:
                        _tokens = lemma

                    if _tokens is None:
                        continue
                    tokens += _tokens

                if tokens:
                    if doc_id_f:
                        doc_id_f.write('{}:{}\n'.format(obj['doi'], m))
                    yield tokens

            n += 1
            if time.time() - check_time > 5:
                self._logger.info('Processed %d/%d documents in collection. ETA: %s',
                                  n, num_total_docs, estimate_eta(start_time, n, num_total_docs))
                check_time = time.time()

        self._logger.info('Processed %d/%d documents in collection.', n, num_total_docs)
        if doc_id_f is not None:
            doc_id_f.close()

    def _iter_sentences(self, document_id_fn, paragraph_filter, token_filter):
        num_total_docs = self.destination_collection.find().count()
        check_time = time.time()
        start_time = check_time
        n = 0

        if document_id_fn is None:
            doc_id_f = None
        elif isinstance(document_id_fn, str):
            doc_id_f = open(document_id_fn, 'w')
        else:
            doc_id_f = document_id_fn

        for n, obj in enumerate(self.destination_collection.find()):
            for m, p in enumerate(obj['paragraphs']):
                if paragraph_filter is not None:
                    p = paragraph_filter(p)
                    if p is None:
                        continue

                current_sent_end = 0

                for sent in p['sentences']:
                    orth = sent['orth']
                    lemma = sent['lemmas']
                    pos = sent['pos']

                    current_sent_end += len(orth)

                    if token_filter is not None:
                        tokens = token_filter(orth, lemma, pos)
                    else:
                        tokens = lemma

                    if tokens is None:
                        continue

                    if tokens:
                        if doc_id_f:
                            doc_id_f.write('{}:{}:{}:{}\n'.format(
                                obj['doi'], m, current_sent_end - len(orth), current_sent_end
                            ))
                        yield tokens

            n += 1
            if time.time() - check_time > 5:
                self._logger.info('Processed %d/%d documents in collection. ETA: %s',
                                  n, num_total_docs, estimate_eta(start_time, n, num_total_docs))
                check_time = time.time()

        self._logger.info('Processed %d/%d documents in collection.', n, num_total_docs)
        if doc_id_f is not None and isinstance(document_id_fn, str):
            doc_id_f.close()

    def iter_recipe_sentence(self, token_filter=None, document_id_fn=None):
        """Iterate over all recipe paragraphs (MIT result).

        :param token_filter: A filter function applied to all sentence tokens lists.
                             The function will be called by filter(orth, lemma, pos)
                             If this function returns None, that means drop this paragraph.
                             Default filter does nothing and takes the lemma of each word.
        :type token_filter: callable
        :param document_id_fn: Filename of the file of storing document ids. The format is:
                             doi:paragraph_id:word_start_idx:word_end_idx
        :type document_id_fn: str
        :rtype generator
        """

        def paragraph_filter(p):
            if p['classification_MIT']['recipe']:
                return p
            else:
                return None

        return self._iter_sentences(document_id_fn=document_id_fn,
                                    paragraph_filter=paragraph_filter,
                                    token_filter=token_filter)

    def iter_all_sentence(self, token_filter=None, document_id_fn=None):
        return self._iter_sentences(document_id_fn=document_id_fn,
                                    paragraph_filter=None,
                                    token_filter=token_filter)

    def iter_recipe_paragraph(self, token_filter=None, document_id_fn=None):
        """Iterate over all recipe paragraphs.

        :param token_filter: A filter function applied to all sentence tokens lists.
                             The function will be called by filter(orth, lemma, pos)
                             If this function returns None, that means drop this paragraph.
                             Default filter does nothing and takes the lemma of each word.
        :type token_filter: callable
        :param document_id_fn: Filename of the file of storing document ids.
        :type document_id_fn: str
        :rtype generator
        """

        def paragraph_filter(p):
            if p['classification_MIT']['recipe']:
                return p
            else:
                return None

        return self._iter_paragraphs(document_id_fn=document_id_fn,
                                     paragraph_filter=paragraph_filter,
                                     token_filter=token_filter)

    def iter_paragraph(self, token_filter=None, document_id_fn=None):
        """Iterate over all paragraphs.

        :param token_filter: A filter function applied to all sentence tokens lists.
                             The function will be called by filter(orth, lemma, pos)
                             If this function returns None, that means drop this paragraph.
                             Default filter does nothing and takes the lemma of each word.
        :type token_filter: callable
        :param document_id_fn: Filename of the file of storing document ids.
        :type document_id_fn: str
        :rtype generator
        """

        return self._iter_paragraphs(document_id_fn=document_id_fn,
                                     paragraph_filter=None,
                                     token_filter=token_filter)

    def tokenize_corpus(self, object_id_list=None, clean_database=False):
        """Tokenize all corpus in the syn_20170926.

        :param object_id_list: ObjectId list to tokenize.
        :type object_id_list: list
        :param clean_database: Remove old data in collection.
        :type clean_database: bool
        :returns: Statistics about number of documents processed.
        :rtype: dict
        """
        if self.is_collection_ready() and clean_database:
            self._logger.info('Clearing old collection.')
            self.destination_collection.delete_many({})

        if object_id_list is None:
            num_documents = self.syn_20170926.find({}).count()
        else:
            num_documents = len(object_id_list)
        self._logger.info('Processing %d documents.', num_documents)

        def doc_iterator():
            if object_id_list is None:
                for _i in self.syn_20170926.find():
                    yield _i
            else:
                for _i in object_id_list:
                    d = self.syn_20170926.find_one({'_id': ObjectId(_i)})
                    if d is None:
                        raise RuntimeError('No such object %s' % _i)
                    yield d

        statistics = {
            'number_docs': 0,
            'number_sentences': 0,
            'number_words': 0
        }

        check_time = time.time()
        start_time = check_time

        for i, doc in enumerate(doc_iterator()):
            doc_token = {
                'doi': doc['doi'],
                'syn_20170926_id': doc['_id'],
                'paragraphs': []
            }

            for j, paragraph in enumerate(doc['paragraphs']):
                processor = preprocessing.TextPreprocessor(paragraph['text'])
                cde_doc = processor.doc.user_data

                all_lemmas = processor.get_words(lemma=True)
                sentences = []
                for sentence in cde_doc.sentences:
                    orths, pos = zip(*sentence.pos_tagged_tokens)
                    lemmas = all_lemmas[:len(pos)]

                    all_lemmas = all_lemmas[len(pos):]

                    sentences.append({
                        'orth': orths,
                        'pos': pos,
                        'lemmas': lemmas
                    })

                    statistics['number_words'] += len(orths)
                    statistics['number_sentences'] += 1

                assert len(all_lemmas) == 0

                doc_token['paragraphs'].append({
                    'id': j,
                    'sentences': sentences,
                    'classification_MIT': {
                        'recipe': paragraph['type'] == 'recipe'
                    }
                })

            self.destination_collection.insert_one(doc_token)
            statistics['number_docs'] += 1

            if time.time() - check_time > 5:
                check_time = time.time()
                logging.info('Tokenization in progress. Current %d documents, %d sentences, %d words. ETA: %s',
                             statistics['number_docs'], statistics['number_sentences'], statistics['number_words'],
                             estimate_eta(start_time, i, num_documents))

        return statistics
