import logging

import chemdataextractor.doc
import re
import spacy
import spacy.tokens
import textacy
from chemdataextractor.nlp import CemTagger, ChemCrfPosTagger

__all__ = ['TextPreprocessor']

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"

_nlp = None


class ChemDataInfoTokenizer(object):
    cached_cem_tagger = CemTagger()
    cached_pos_tagger = ChemCrfPosTagger()

    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        doc = chemdataextractor.doc.Paragraph(
            text,
            ner_tagger=self.cached_cem_tagger,
            pos_tagger=self.cached_pos_tagger
        )
        tokens = [x.text for x in sum(doc.tokens, [])]
        spaces = [True] * len(tokens)
        return spacy.tokens.Doc(self.vocab, words=tokens, spaces=spaces, user_data=doc)


class TextPreprocessor(object):
    VERSION = '0.1.0'

    def __init__(self, text, textacy=True):
        """
        Create a new TextPreprocessor instance from a paragraph.

        :param text: the paragraph to be processed.
        :type text: str
        """
        self.doc = self._process(text, textacy)

    def _process(self, text, do_textacy):
        if do_textacy:
            text = textacy.preprocess_text(text, fix_unicode=True)
        doc = self._make_doc(text)

        return doc

    def _make_doc(self, text):
        global _nlp
        if _nlp is None:
            _nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            # _nlp.make_doc = ChemDataInfoTokenizer(_nlp)
            logging.info('Loading SpaCy model, with custom tokenizer.')
            for name, obj in _nlp.pipeline:
                logging.info('SpaCy model has pipeline %s: %r', name, obj)

        return _nlp(text)

    def get_verbs(self, lemma=False):
        """
        Get all verbs in the paragraph.

        :param lemma: return lemma instead of orth.
        :type lemma: bool
        :return: a list of lists of verbs.
        :rtype: list
        """
        tokens = [x for x in self.doc if x.pos_ == 'VERB']
        if lemma:
            return [x.lemma_ for x in tokens]
        else:
            return [x.orth_ for x in tokens]

    def get_words(self, lemma=False):
        """
        Get all words in the paragraph.

        :param lemma: return lemma instead of orth.
        :type lemma: bool
        :return: a list of lists of words.
        :rtype: list
        """
        if lemma:
            return [x.lemma_ for x in self.doc]
        else:
            return [x.orth_ for x in self.doc]

    def get_pos(self):
        """
        Get POS tags for all words.
        """
        return [x.pos_ for x in self.doc]

    def get_cde_tokens(self):
        """
        Get all tokens from chemdataextractor.

        :return:
        """
        cde_obj = self.doc.user_data
        return cde_obj.tokens
