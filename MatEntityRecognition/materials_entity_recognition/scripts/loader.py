import codecs
import numpy as np

from .utils import create_dico, create_mapping, zero_digits
from .utils import iob2, iob_iobes
from .sent_ele_func import get_ele_features

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    :param lower: use lower case or not
    :param zeros: convert digit numbers to zeros or not
    :return sentences: list of sentence
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.

    :param sentences: list of sentence
    :param tag_scheme: iob or iobes
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.

    :param sentences: list of sentence
    :param lower: use lower case or not
    :return dico: dictionary of all words
    :return word_to_id: mapping from a word to a number (id)
    :return id_to_word: mapping from a number (id) to a word
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    # dico['<MAT>'] = 9999999
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    
    :param sentences: list of sentence
    :return dico: dictionary of all characters
    :return char_to_id: mapping from a character to a number (id)
    :return id_to_char: mapping from a number (id) to a character
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    
    :param sentences: list of sentence
    :return dico: dictionary of all tags
    :return tag_to_id: mapping from a tag to a number (id)
    :return id_to_tag: mapping from a number (id) to a tag
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)

    :param s: text
    :return captial characcter feature
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words,
                     word_to_id,
                     char_to_id,
                     lower=False,
                     use_CHO=False,
                     use_eleNum=False,
                     input_tokens=[],
                     original_para_text='',
                     ):
    """
    Prepare a sentence for evaluation.

    :param str_words: list of words
    :param word_to_id: mapping from a word to a number (id)
    :param char_to_id: mapping from a character to a number (id)
    :param lower: use lower case or not
    :param use_key_word: use key words or not
    :param use_topic: use topic feature or not
    :return dict corresponding to input features
    """
    def f(x): return x.lower() if lower else x
    words = []
    for w in str_words:
        if f(w) in word_to_id:
            tmp_word = f(w)
        else:
            tmp_word = '<UNK>'
        words.append(word_to_id[tmp_word])
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]

    ele_nums = [[0] for w in str_words]
    has_CHOs = [[0] for w in str_words]
    if use_CHO or use_eleNum:
        # get ele_num
        ele_nums, has_CHOs = get_ele_features(input_tokens, original_para_text)

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'ele_nums': ele_nums,
        'has_CHOs': has_CHOs,
    }


