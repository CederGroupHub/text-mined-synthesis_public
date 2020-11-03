import os
import re
import codecs
import numpy as np
import theano

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

def get_name(parameters):
    """
    Generate a model name from its parameters.

    :param parameters
    :return name
    """
    l = []
    for k, v in list(parameters.items()):
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.

    :param name: name of network
    :param param: parameters
    :param pretrained: pretrained values 
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.

    :param shape: shape of array
    :param name: name of array
    :return shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.

    :param item_list: a list of list of items.
    :return dico: dictionary of items
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.

    :param dico: dictionary of items
    :return item_to_id: mapping from an item to a number (id) 
    :return item_to_id: mapping from a number (id) to an item
    """
    sorted_items = sorted(list(dico.items()), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in list(id_to_item.items())}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.

    :param s: a word
    :return modified word where digit numbers are replaced by zero
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.

    :param tags
    :return True or False
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES

    :param tags: iob tags
    :return new_tags: iobes tags
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB

    :param tags: iobes tags
    :return new_tags: iob tags
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.

    :param words: list of word ids
    :param singletons: set of words only appear one time in training set
    :param p: probability for replacement
    :return new_words: modified list of word ids 
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words

def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    
    :param words: list of lists of ints (list of words, a word being a list of char indexes)
    :return char_for: padded list of lists of ints in the forward direction
    :return char_rev: padded list of lists of ints in the reversed direction
    :return char_pos: list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.

    :param data: a dict as sentence data
    :param parameters: parameters of model
    :param add_label: add tags for words or not
    :param singletons: set of words only appear one time in training set
    :return input: all input features correponding to a sentence 
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if 'ele_num' in parameters and parameters['ele_num']:
        input.append(data['ele_nums'])
    if 'has_CHO' in parameters and parameters['has_CHO']:
        input.append(data['has_CHOs'])
    if 'topic_dim' in parameters and parameters['topic_dim']:
        input.append(data['topics'])
    if 'keyword_dim' in parameters and parameters['keyword_dim']:
        input.append(data['key_words'])
    if add_label:
        input.append(data['tags'])
    return input


def find_one_entity_token_ids(sent_tokens, entity, start_to_calibrate=False):
    ids = []
    if 'token_ids' not in entity and start_to_calibrate:
        entity['start'] += sent_tokens[0]['start']
        entity['end'] += sent_tokens[0]['start']
    for i, t in enumerate(sent_tokens):
        if (t['start'] >= entity['start'] and t['end'] <= entity['end']):
            ids.append(i)
    if 'token_ids' in entity:
        assert ids == entity['token_ids']
    assert sent_tokens[ids[0]]['start'] == entity['start']
    assert sent_tokens[ids[-1]]['end'] == entity['end']
    entity['token_ids'] = ids
    return ids


