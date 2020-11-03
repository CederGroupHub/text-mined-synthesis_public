import codecs
import json
import random
import time
from pprint import pprint

import numpy as np
import spacy
import re
import tensorflow as tf

from .utils import create_dico, zero_digits
from .utils import iob2, iob_iobes
from .utils import create_mapping, pad_word_chars
from .utils import find_sub_token, offset_tokens
from .utils import get_entities, reformat_bert_tokens, get_bert_input
from .sent_ele_func import get_ele_features

from typing import List
from typing import Dict

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

nlp = spacy.load('en')

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    :param lower: use lower case or not
    :param zeros: convert digit numbers to zeros or not
    :return sentences: list of sentence

    """

    with open(path, 'r', encoding='utf-8') as fr:
        sentences = json.load(fr)
    sentences = [s['tokens'] for s in sentences]
    sentences = preprocess_sentences(sentences, zeros, lower)
    return sentences

def preprocess_sentences(sentences, zeros, lower):
    if zeros:
        for s in sentences:
            for t in s:
                t['text'] = zero_digits(t['text'])
    if lower:
        for s in sentences:
            for t in s:
                t['text'] = t['text'].lower()
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.

    :param sentences: list of sentence
    :param tag_scheme: iob or iobes
    """
    for i, s in enumerate(sentences):
        tags = [t['label'] for t in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for t, new_tag in zip(s, tags):
                t['label'] = new_tag
        elif tag_scheme == 'iobes':
            # If format was IOB1, we convert to IOB1
            new_tags = iob_iobes(tags)
            for t, new_tag in zip(s, new_tags):
                t['label'] = new_tag
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
    words = [[t['text'].lower() if lower else t['text'] for t in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    # dico['<MAT>'] = 9999999
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences, use_ori_text_char=False):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    
    :param sentences: list of sentence
    :return dico: dictionary of all characters
    :return char_to_id: mapping from a character to a number (id)
    :return id_to_char: mapping from a number (id) to a character
    """
    if use_ori_text_char:
        chars = ["".join([t['original_text'] for t in s]) for s in sentences]
    else:
        chars = ["".join([t['text'] for t in s]) for s in sentences]
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
    tags = [[t['label'] for t in s] for s in sentences]
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


def prepare_sentences(sentences,
                      word_to_id,
                      char_to_id,
                      tag_to_id,
                      zeros=False,
                      lower=False,
                      element_feature=False,
                      batch_size=1,
                      use_ori_text_char=False,
                      original_para_text=[],
                      bert_tokenizer=None):
    """
    Prepare a sentence for evaluation.

    :param sentences: list of list of tokens
    :param word_to_id: mapping from a word to a number (id)
    :param char_to_id: mapping from a character to a number (id)
    :param lower: use lower case or not
    :return dict corresponding to input features
    """
    if element_feature:
        for i, sent in enumerate(sentences):
            ele_num, only_CHO = get_ele_features(sent, original_para_text[i])
            for i in range(len(sent)):
                sent[i]['ele_num'] = ele_num[i]
                sent[i]['only_CHO'] = only_CHO[i]

    sentences = preprocess_sentences(sentences, zeros, lower)
    data_X, data_Y, data, sentences = prepare_dataset(
        sentences=sentences,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tag_to_id=tag_to_id,
        lower=lower,
        batch_size=batch_size,
        use_ori_text_char=use_ori_text_char,
        bert_tokenizer=bert_tokenizer,
    )

    return data_X, data_Y, data, sentences


def prepare_embedding_matrix(id_to_word, word_dim, emb_path):
    """
    load embeddings as a numpy 2d array. Source of embeddings can be provided by
    specifying either emb_path or emb_dict

    :param id_to_tag: mapping from a number (id) to a tag of word
    :param word_dim: dimension of embedding for each word
    :param emb_path: path to embedding file (str)
    :return embedding_matrix: a embedding matrix (numpy 2d array), the index of 
                              matrix is consist with the id in id_to_word
    """
    n_words = len(id_to_word)
    embedding_matrix = np.random.uniform(low=-1.0, high=1.0, size=(n_words, word_dim))

    # get embedding dict: dict of embedding (dict such as {w0: emb0, w1: emb1})
    # load from emb_path
    print('Loading pretrained embeddings from %s...' % emb_path)
    emb_dict = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            emb_dict[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines in embeddings file' % emb_invalid)
    assert emb_invalid < 1

    # Lookup embedding dict
    c_found = 0
    c_lower = 0
    c_zeros = 0
    c_lemma = 0
    for i in range(n_words):
        # all ids starts from 1
        word = id_to_word[i+1]
        if word in emb_dict:
            embedding_matrix[i] = emb_dict[word]
            c_found += 1
        elif word.lower() in emb_dict:
            embedding_matrix[i] = emb_dict[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in emb_dict:
            embedding_matrix[i] = emb_dict[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
        elif word not in {'<MAT>', '<UNK>'}:
            word_lemma = nlp(word)[0].lemma_
            if word_lemma in emb_dict:
                embedding_matrix[i] = emb_dict[word_lemma]
                c_lemma += 1
    print('Loaded %i pretrained embeddings.' % len(emb_dict))
    print(('%i / %i (%.4f%%) words have been initialized with '
           'pretrained embeddings.') % (
                c_found + c_lower + c_zeros + c_lemma, n_words,
                100. * (c_found + c_lower + c_zeros + c_lemma) / n_words
          ))
    print(('%i found directly, %i after lowercasing, '
           '%i after lowercasing + zero. %i after lemma.') % (
              c_found, c_lower, c_zeros, c_lemma
          ))

    return embedding_matrix

def dict_to_tf_dataset(data,
                       data_type,
                       data_shape,
                       padded_shape=None,
                       preprocess_func={},
                       column_y=None,
                       batch_size=1,
                       mode='static'):
    """

    :param data: list of dict contain all different types of data,
            all variables should be numerical numbers
    :param data_type: tensorflow data type of the numerical numbers
    :param data_shape: tensorflow shape of the values data
    :param column_y: name of y columns.
            If not specified, the value of dataset_y returned
            would be zeros with the same length of X.
            It seems that tf v2.0 only support y as a array rather than a dict.
    :param batch_size: size of batch
    :return: dataset_x: dataset formatted from dict of features
    :return: dataset_y: dataset formatted from array of y
    """
    features = set(data[0].keys()) & set(data_type.keys())
    if column_y:
        features -= {column_y, }
    features_type = {k: data_type[k] for k in features}
    features_shape = {k: data_shape[k] for k in features}
    if padded_shape == None:
        padded_shape = data_shape
    padded_features_shape = {k: padded_shape[k] for k in features}
    for k in data[0].keys():
        if k not in preprocess_func:
            preprocess_func[k] = lambda x: x

    random_seed = time.time()
    # another way to generate dict is to use dataset.map
    # https://github.com/tensorflow/tensorflow/issues/28643
    def feature_dict_gen_static():
        for d in data:
            feature_dict = {}
            for k, v in d.items():
                if k not in features:
                    continue
                feature_dict[k] = preprocess_func[k](v)
            # print('generted x sample')
            yield feature_dict

    def feature_dict_gen_dynamic():
        random.seed(random_seed)
        while (True):
            d = random.choice(data)
            feature_dict = {}
            for k, v in d.items():
                if k not in features:
                    continue
                feature_dict[k] = preprocess_func[k](v)
            # print('generted x sample')
            yield feature_dict

    def y_array_gen_static():
        for d in data:
            if column_y:
                assert column_y in d
                # print('generted y sample')
                yield preprocess_func[column_y](d[column_y])
            else:
                yield 0.0

    def y_array_gen_dynamic():
        random.seed(random_seed)
        while True:
            d = random.choice(data)
            if column_y:
                assert column_y in d
                # print('generted y sample')
                yield  preprocess_func[column_y](d[column_y])
            else:
                yield 0.0

    generator = {
        'static': {
            'x': feature_dict_gen_static,
            'y': y_array_gen_static,
        },
        'dynamic': {
            'x': feature_dict_gen_dynamic,
            'y': y_array_gen_dynamic,
        }
    }

    # or use dataset.map
    # https://github.com/tensorflow/tensorflow/issues/28643
    dataset_x = tf.data.Dataset.from_generator(
        generator[mode]['x'],
        output_types=features_type,
        output_shapes=features_shape,
    )
    dataset_y = tf.data.Dataset.from_generator(
        generator[mode]['y'],
        output_types=data_type[column_y] \
            if column_y else tf.float32,
        output_shapes=data_shape[column_y] \
            if column_y else tf.TensorShape([]),
    )
    data_batch_x = dataset_x.padded_batch(
        batch_size,
        padded_shapes=padded_features_shape
    )
    if column_y:
        data_batch_y = dataset_y.padded_batch(
            batch_size,
            padded_shapes=padded_features_shape[column_y]
        )
    else:
        data_batch_y = dataset_y.padded_batch(
            batch_size,
            padded_shapes=[]
        )
    return data_batch_x, data_batch_y

def prepare_datadict(
    sentences,
    tag_to_id,
    word_to_id=None,
    char_to_id=None,
    lower=False,
    use_ori_text_char=False,
    bert_tokenizer=None
):
    """
    Prepare the dataset. Return a list of dictionaries containing:
        - word ids
        - tag ids
    """

    if bert_tokenizer is None:
        data = prepare_datadict_base(
            sentences=sentences,
            word_to_id=word_to_id,
            char_to_id=char_to_id,
            tag_to_id=tag_to_id,
            lower=lower,
            use_ori_text_char=use_ori_text_char,
        )
    else:
        data = prepare_datadict_bert(
            sentences=sentences,
            bert_tokenizer=bert_tokenizer,
            tag_to_id=tag_to_id,
        )

    return data

def prepare_datadict_base(
    sentences: List[List[Dict]],
    word_to_id,
    char_to_id,
    tag_to_id: Dict[str, int],
    lower=False,
    use_ori_text_char=False,
):
    """
    Prepare the dataset. Return a dictionary

    :param sentences: list of lists of tokens
    :param word_to_id:
    :param char_to_id:
    :param tag_to_id:
    :param lower:
    :param use_ori_text_char:
    :param bert_format:
    :return:
    """
    # TODO: remove lower because duplicated with load_sentences
    def f(x):
        return x.lower() if lower else x

    data = []

    for sent in sentences:
        str_words = [t['text'] for t in sent]
        words = []
        for tmp_index, w in enumerate(str_words):
            if f(w) in word_to_id:
                tmp_word = f(w)
            else:
                tmp_word = '<UNK>'
            words.append(word_to_id[tmp_word])

        if use_ori_text_char:
            ori_text = [t.get('original_text', t['text']) for t in sent]
            # TODO: is that helpful to add an unk char in the mapping
            #     in that way, singleton for char is also neccessary in training
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in ori_text]
        else:
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in str_words]
        char_for, char_rev, char_pos = pad_word_chars(chars)

        ele_num = [t.get('ele_num', 0.0) for t in sent]
        only_CHO = [t.get('only_CHO', 0) for t in sent]
        tar_tag = [int('Tar' in t.get('label', '')) for t in sent]
        pre_tag = [int('Pre' in t.get('label', '')) for t in sent]

        # use get() because we don't know label in prediction, where tags is just a placehold
        tags = [tag_to_id[t.get('label', 'O')] for t in sent]
        score_mask = np.ones_like(words, dtype=np.int32)
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': char_for,
            'ele_num': ele_num,
            'only_CHO': only_CHO,
            'tar_tag': tar_tag,
            'pre_tag': pre_tag,
            'tags': tags,
            'length': len(sent),
            'num_tokens': len(sent),
            'score_mask': score_mask,
        })
    return data

def prepare_datadict_bert(
    sentences: List[List[Dict]],
    bert_tokenizer,
    tag_to_id: Dict[str, int]
):
    """

    :param sentences:
    :param bert_tokenizer: a fast tokenizer inheriting from PreTrainedTokenizerFast.
    :param tag_to_id:
    :return:
    """
    data = []

    for sent in sentences:
        bert_input = get_bert_input(
            tokenizer=bert_tokenizer, pre_tokens=sent
        )
        bert_tokens = bert_input['tokens']

        # convert all labels (str) to tags (int)
        # TODO: unify the name from label to tag in the future
        # use get() because we don't know label in prediction, where tags is just a placehold
        labels = [
            sent[t['source_token_idx']].get('label', 'O')
            if 'source_token_idx' in t else 'O'
            for t in bert_tokens
        ]
        tags = [tag_to_id[l] for l in labels]

        # get score_mask
        # [CLS] and [SEP] does not contribute to loss in NER
        # Only the first word piece of a token contributes to loss
        # The rest pieces correspond to the same prediction as the first one
        score_mask = []
        last_token_idx = -1
        for t in bert_tokens:
            if t.get('source_token_idx', -1) > last_token_idx:
                score_mask.append(1)
                last_token_idx = t['source_token_idx']
            else:
                score_mask.append(0)

        # append new entry to all data
        data.append({
            'words': bert_input['ids'],
            'bert_token_type_ids': bert_input['type_ids'],
            'bert_attention_mask': bert_input['attention_mask'],
            'tags': tags,
            'length': len(bert_tokens),
            'num_tokens': len(sent),
            'score_mask': score_mask,
        })

    return data


def get_input_format(model_type='MER'):
    data_type = None
    data_shape = None
    padded_data_shape = None
    if model_type == 'MER':
        data_type = {
            'words': tf.int32,
            'chars': tf.int32,
            'caps': tf.int32,
            'ele_num': tf.float32,
            'only_CHO': tf.int32,
            'tar_tag': tf.int32,
            'pre_tag': tf.int32,
            'tags': tf.int32,
            'length': tf.int32,
            'num_tokens': tf.int32,
            'bert_token_type_ids': tf.int32,
            'bert_attention_mask': tf.int32,
            'score_mask': tf.int32,
        }

        data_shape = {
            'words': tf.TensorShape([None]),
            'chars': tf.TensorShape([None, None]),
            'caps': tf.TensorShape([None]),
            'ele_num': tf.TensorShape([None]),
            'only_CHO': tf.TensorShape([None]),
            'tar_tag': tf.TensorShape([None]),
            'pre_tag': tf.TensorShape([None]),
            'tags': tf.TensorShape([None]),
            'length': tf.TensorShape([]),
            'num_tokens': tf.TensorShape([]),
            'bert_token_type_ids': tf.TensorShape([None]),
            'bert_attention_mask': tf.TensorShape([None]),
            'score_mask': tf.TensorShape([None]),
        }

        padded_data_shape = {
            'words': tf.TensorShape([None]),
            'chars': tf.TensorShape([None, None]),
            'caps': tf.TensorShape([None]),
            'ele_num': tf.TensorShape([None]),
            'only_CHO': tf.TensorShape([None]),
            'tar_tag': tf.TensorShape([None]),
            'pre_tag': tf.TensorShape([None]),
            'tags': tf.TensorShape([None]),
            'length': tf.TensorShape([]),
            'num_tokens': tf.TensorShape([]),
            'bert_token_type_ids': tf.TensorShape([None]),
            'bert_attention_mask': tf.TensorShape([None]),
            'score_mask': tf.TensorShape([None]),
        }

    return data_type, data_shape, padded_data_shape

def prepare_dataset(sentences,
                    tag_to_id,
                    word_to_id=None,
                    char_to_id=None,
                    lower=False,
                    batch_size=1,
                    sampling_ratio=1.0,
                    ds_gen_mode='static',
                    use_ori_text_char=False,
                    singletons=set(),
                    singleton_unk_probability=0.0,
                    bert_tokenizer=None):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word ids
        - tag ids
    """
    if sampling_ratio < 1.0:
        sampled_sentences = random.sample(
            sentences,
            k=int(len(sentences)*sampling_ratio)
        )
    else:
        sampled_sentences = sentences
    data_dicts = prepare_datadict(
        sentences=sampled_sentences,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tag_to_id=tag_to_id,
        lower=lower,
        use_ori_text_char=use_ori_text_char,
        bert_tokenizer=bert_tokenizer
    )
    data_type, data_shape, padded_data_shape = get_input_format()

    preprocess_func = {}
    if singleton_unk_probability > 0.0:
        preprocess_func['words'] = get_insert_singletons_func(
            singletons=singletons,
            word_to_id=word_to_id,
            p=singleton_unk_probability,
        )

    # flexible dimension for model input
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#class_dataset
    # https://stackoverflow.com/questions/51136862/creating-a-tensorflow-dataset-that-outputs-a-dict
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict
    # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    # https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
    data_X, data_Y = dict_to_tf_dataset(
        data_dicts,
        data_type,
        data_shape,
        padded_shape=padded_data_shape,
        preprocess_func=preprocess_func,
        column_y=None,
        batch_size=batch_size,
        mode=ds_gen_mode,
    )
    return data_X, data_Y, data_dicts, sampled_sentences

def get_insert_singletons_func(singletons, word_to_id, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    def insert_singletons(words):
        new_words = []
        for word in words:
            if word in singletons and np.random.uniform() < p:
                new_words.append(word_to_id['<UNK>'])
            else:
                new_words.append(word)
        return new_words
    return insert_singletons

