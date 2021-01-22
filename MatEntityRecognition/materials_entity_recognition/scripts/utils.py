import copy
import os
import re
import sys
import copy

import numpy as np
import collections
import tensorflow as tf
import psutil
import importlib
from pprint import pprint
from typing import List, Dict

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

NEAR_ZERO = 1e-6

def print_gpu_info():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('print_gpu_info: ', len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def allow_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        num_python_proc = 0
        path = os.path.abspath('.')
        tmp_m = re.match('.*MER_([0-9]+).*', path)
        if tmp_m:
            num_python_proc = int(tmp_m.group(1))
        else:
            for proc in psutil.process_iter():
                if 'python' in proc.name():
                    num_python_proc += 1
        try:
            # Currently, memory growth needs to be the same across GPUs
            # use gpu
            gpu_to_use = gpus[num_python_proc % len(gpus)]
            tf.config.experimental.set_visible_devices(gpu_to_use, 'GPU')
            tf.config.experimental.set_memory_growth(gpu_to_use, True)
            # # use cpu
            # tf.config.experimental.set_visible_devices([], 'GPU')
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('allow_gpu_growth: ', len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if os.environ.get('tf_allow_gpu_growth', 'False') != 'True':
    allow_gpu_growth()
    os.environ['tf_allow_gpu_growth'] = 'True'

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def found_package(package_name):
    pkg_check = importlib.util.find_spec(package_name)
    found = pkg_check is not None
    return found

def use_file_as_stdout(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    sys.stdout = open(file_path, 'w')
    sys.stdout = Unbuffered(sys.stdout)
    print('this is printed in the console')

def parse_lr_method(lr_method, delimiter='@'):
    # Parse optimization method parameters
    if delimiter in lr_method:
        lr_method_name = lr_method[:lr_method.find(delimiter)]
        lr_method_parameters = {}
        for x in lr_method[lr_method.find(delimiter) + 1:].split(delimiter):
            split = x.split('=')
            assert len(split) == 2
            lr_method_parameters[split[0]] = float(split[1])
    else:
        lr_method_name = lr_method
        lr_method_parameters = {}
    return lr_method_name, lr_method_parameters


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
    0 is not used as id because 0 is usually used as padding/masks
    to avoid confusion, 0 is not used as id

    :param dico: dictionary of items
    :return item_to_id: mapping from an item to a number (id) 
    :return item_to_id: mapping from a number (id) to an item
    """
    sorted_items = sorted(list(dico.items()), key=lambda x: (-x[1], x[0]))
    id_to_item = {i+1: v[0] for i, v in enumerate(sorted_items)}
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
    # TODO: pad along batch axis
    max_length = max([len(word) for word in words])
    # avoiding zero-length tokens makes the program more robust
    max_length = max(max_length, 1)
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos

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

def find_sub_token(target_token, all_tokens):
    """
    find tokens in all_tokens which is substrings of target_token

    :param target_token:
    :param all_tokens:
    :return:
    """
    sub_tokens = []
    for tmp_token in all_tokens:
        # skip some speical tokens (from BERT) such as [cls] [sep]
        if tmp_token['start'] >= tmp_token['end']:
            continue
        if tmp_token['start'] >= target_token['start'] and \
            tmp_token['end'] <= target_token['end']:
            sub_tokens.append(tmp_token)
    return sub_tokens


def offset_tokens(tokens, offset):
    """
    shift token start and end with an offset

    :param tokens:
    :param offset:
    :return:
    """
    reformated_tokens = copy.deepcopy(tokens)
    for t in reformated_tokens:
        t['start'] += offset
        t['end'] += offset
    return reformated_tokens


def reformat_bert_tokens(bert_tokens, text: str):
    """
    reformat bert tokens to list of dicts

    :param bert_tokens:
    :param text:
    :return:
    """
    reformated_tokens = []
    for offset, token in zip(bert_tokens.offsets, bert_tokens.tokens):
        reformated_tokens.append({
            'start': offset[0],
            'end': offset[1],
            'text': text[offset[0]: offset[1]],
            'bert_text': token,
        })
    return reformated_tokens


def fix_missed_bert_tokens(
    span_text: str,
    tokens: List[Dict],
    unk_token: str = '[UNK]'
):
    """
    BertTokenizerFast from transformers incorrectly miss some non-ascii special characters.
    We use this function to find them back

    :param span_text:
    :param tokens:
    :param unk_token:
    :return:
    """
    all_tokens = []
    last_end = 0
    if len(tokens) == 0:
        tokens = [{
            'start': len(span_text),
            'end': len(span_text),
            'text': '',
        }]
    for token in tokens:
        if (
            token['start'] > last_end
            and len(span_text[last_end: token['start']].strip(' ')) > 0
        ):
            text = span_text[last_end: token['start']]
            all_tokens.append({
                'start': last_end + len(text) - len(text.lstrip(' ')),
                'end': last_end + len(text.rstrip(' ')),
                'text': text.strip(' '),
                'bert_text': unk_token,
            })
        if token['end'] > token['start']:
            all_tokens.append(token)
        last_end = token['end']
    return all_tokens


def get_bert_tokens_in_span(
        a_span: Dict,
        bert_pieces: List,
        unk_token: str = '[UNK]'
):
    """
    Tokenize a span with bert tokenization results.
    Input bert_pieces because bert_pieces can be obtained in batch.
    Output does not contain [CLS] and [SEP]

    :param a_span:
    :param bert_pieces:
    :param unk_token:
    :return:
    """
    pieces_tokens = reformat_bert_tokens(
        bert_tokens=bert_pieces,
        text=a_span['text']
    )
    # For a span, we remove [CLS] and [SEP]
    pieces_tokens = pieces_tokens[1:-1]

    # transformers' fast tokenizer misses some special characters
    pieces_tokens = fix_missed_bert_tokens(
        a_span['text'], pieces_tokens, unk_token=unk_token
    )
    pieces_tokens = offset_tokens(pieces_tokens, a_span['start'])
    return pieces_tokens


def get_bert_tokens(tokenizer, sentence_text=None, pre_tokens=None):
    """
    Tokenize a sentence with a bert tokenizer.
    Pure tokenization: [CLS] and [SEP] are not included in the output tokens.

    :param tokenizer:
    :param sentence_text:
    :param pre_tokens:
    :return:
    """
    bert_tokens = []
    if pre_tokens is None:
        pre_tokens = [{
            'start': 0,
            'end': len(sentence_text),
            'text': sentence_text,
        }]

    # batch tokenization for acceleration
    words = [token['text'] for token in pre_tokens]
    word_pieces = tokenizer(words)
    for i, token in enumerate(pre_tokens):
        pieces_tokens = get_bert_tokens_in_span(
            token, word_pieces[i], unk_token=tokenizer.unk_token
        )
        for t in pieces_tokens:
            t['source_token_idx'] = i
        bert_tokens.extend(pieces_tokens)

    return bert_tokens


def get_bert_input(tokenizer, sentence_text=None, pre_tokens=None):
    """
    Tokenize and encode a sentence with a bert tokenizer.
    Used as input to a bert encoder: [CLS] and [SEP] are included in the output tokens.

    :param tokenizer:
    :param sentence_text:
    :param pre_tokens:
    :return:
    """
    bert_tokens = get_bert_tokens(
        tokenizer=tokenizer,
        sentence_text=sentence_text,
        pre_tokens=pre_tokens
    )
    bert_tokens.insert(0, {
        'start': 0,
        'end': 0,
        'text': '',
        'bert_text': tokenizer.cls_token,
    })
    bert_tokens.append({
        'start': 0,
        'end': 0,
        'text': '',
        'bert_text': tokenizer.sep_token,
    })
    bert_pieces = [t['bert_text'] for t in bert_tokens]
    bert_input = {
        'tokens': bert_tokens,
        'ids': tokenizer.convert_tokens_to_ids(bert_pieces),
        'type_ids': [0] * len(bert_pieces),
        'attention_mask': [1] * len(bert_pieces),
    }
    return bert_input


def get_entities(sent_tokens):
    """
    Extract entities and their corresponding labels from annotated data
    Assume the annotated labels are using IOB format

    :param sent_tokens:
    :return:
    """
    entities = []
    # collect entities with labels other than 'O'
    for token in sent_tokens:
        if token['label'] != 'O':
            tag = token['label'][2:].strip()
            if token['label'][0] == 'B':
                entities.append({
                    'text': token['text'],
                    'start': token['start'],
                    'end': token['end'],
                    'label': tag,
                })
                ner_tag = tag
            elif token['label'][0] == 'S':
                entities.append({
                    'text': token['text'],
                    'start': token['start'],
                    'end': token['end'],
                    'label': tag,
                })
                ner_tag = 'O'
            elif (
                token['label'][0] == 'I'
                and tag == ner_tag
            ):
                entities[-1]['text'] += ' '*(token['start']-entities[-1]['end']) + token['text']
                entities[-1]['end'] = token['end']
            elif (
                token['label'][0] == 'E'
                and tag == ner_tag
            ):
                entities[-1]['text'] += ' '*(token['start']-entities[-1]['end']) + token['text']
                entities[-1]['end'] = token['end']
                ner_tag = 'O'
            else:
                print('Error! token tag invalid!')
        else:
            ner_tag = 'O'

    return entities


