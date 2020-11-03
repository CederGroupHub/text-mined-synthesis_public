import json
import regex
from .sent_ele_func import pattern_species

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

name_keywords = {
    'PUBCHEM_IUPAC_OPENEYE_NAME',
    'PUBCHEM_IUPAC_CAS_NAME',
    'PUBCHEM_IUPAC_NAME_MARKUP',
    'PUBCHEM_IUPAC_NAME',
    'PUBCHEM_IUPAC_SYSTEMATIC_NAME',
    'PUBCHEM_IUPAC_TRADITIONAL_NAME',
}
property_keywords = {
    'PUBCHEM_COMPOUND_CID': 'cid',
    'PUBCHEM_MOLECULAR_FORMULA': 'formula',
    'PUBCHEM_SUBSTANCE_ID': 'sid',
    'PUBCHEM_CID_ASSOCIATIONS': 'cid_associations',
    'PUBCHEM_SUBSTANCE_SYNONYM': 'name_list',
}
property_keywords.update({k: 'name' for k in name_keywords})
pattern_capital = regex.compile('^[A-Z][^A-Z]*$')
pattern_comma = regex.compile(',\p{Z}')

def start_with_capital(text):
    result = False
    tokens = text.split()
    for t in tokens:
        if pattern_species.match(t):
            result = False
            break
        if pattern_capital.match(t):
            result = True
    return result

def regulate_name(text):
    regulated = text
    if start_with_capital(text):
        regulated = text.lower()
    regulated = pattern_comma.sub(',', regulated)
    return regulated

def load_dico(dico_paths):
    mat_dico = set()
    for path in dico_paths:
        # init dico
        # with open(, 'r') as fr:
        with open(path, 'r') as fr:
            compounds = json.load(fr)

        mat_dico_1 = set()
        for entry in compounds:
            for k, v in entry.items():
                if property_keywords[k] == 'name':
                    mat_dico_1.add(regulate_name(v))
                if property_keywords[k] == 'name_list':
                    mat_dico_1.update(set(map(regulate_name, v)))
        mat_dico.update(mat_dico_1)

    print('len(mat_dico)', len(mat_dico))
    return mat_dico

def match_mat_in_dico(tokens, para_text, dico):
    matched_mat = []
    tokens_num_in_phrase = list(reversed(range(1, 4)))
    for tokens_num in tokens_num_in_phrase:
        for i in range(len(tokens) - tokens_num + 1):
            text = para_text[tokens[i]['start']: tokens[i+tokens_num-1]['end']]
            if regulate_name(text) in dico:
                matched_mat.append(
                    {
                        'text': text,
                        'start': tokens[i]['start'],
                        'end': tokens[i+tokens_num-1]['end'],
                        'token_ids': list(range(i, i+tokens_num)),
                    }
                )
    return matched_mat


def overlap(phrase_1, phrase_2):
    if (phrase_1['start'] >= phrase_2['end']
            or phrase_1['end'] <= phrase_2['start']):
        return False
    else:
        return True

def equal(phrase_1, phrase_2):
    if (phrase_1['start'] == phrase_2['start']
            and phrase_1['end'] == phrase_2['end']):
        return True
    else:
        return False


def a_contain_b(phrase_a, phrase_b):
    if (phrase_a['start'] <= phrase_b['start']
            and phrase_a['end'] >= phrase_b['end']):
        return True
    else:
        return False

def solve_conflicts(phrase_list, useSuperSetToken=True):
    """
        the mat in the front has better priority
    """
    to_add = []
    to_remove = []
    length = len(phrase_list)
    for i, entry in enumerate(phrase_list):
        if (entry in to_remove
                or entry in to_add):
            continue
        to_add.append(entry)
        for j in range(i + 1, length):
            if overlap(to_add[-1], phrase_list[j]):
                if (a_contain_b(phrase_list[j], to_add[-1])
                    and useSuperSetToken):
                    to_remove.append(to_add.pop())
                    to_add.append(phrase_list[j])
                else:
                    to_remove.append(phrase_list[j])
    to_add = sorted(to_add, key=lambda x: x['start'])
    return to_add