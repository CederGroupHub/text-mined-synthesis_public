import json
import os
import regex as re

import numpy as np
import spacy
from gensim.models import Word2Vec

import keras
from keras import backend as K
from spacy.tokens import Doc


def remove_parentheses(sentence_tokens):

    new_sentence_tokens = []
    sentence_tokens_ids = []
    while ')' in sentence_tokens:
        i = len(sentence_tokens) - 1
        t = sentence_tokens.pop()
        while t != ')' and len(sentence_tokens) > 0:
            new_sentence_tokens.append(t)
            sentence_tokens_ids.append(i)
            t = sentence_tokens.pop()
            i = i - 1

        t = sentence_tokens.pop()
        i = i - 1
        while t != '(' and len(sentence_tokens) > 0:
            t = sentence_tokens.pop()
            i = i - 1

    new_sentence_tokens.extend(t for t in reversed(sentence_tokens))
    sentence_tokens_ids.extend(len(sentence_tokens) - i - 1 for i in range(len(sentence_tokens)))

    new_sentence_tokens.reverse()
    sentence_tokens_ids.reverse()

    if len(new_sentence_tokens) == 0:
        new_sentence_tokens = sentence_tokens[1:-1].copy()
        sentence_tokens_ids = [i for i in range(1, len(sentence_tokens) - 1)]

    return new_sentence_tokens, sentence_tokens_ids


def valid_token(tok):

    if any(c.isdigit() for c in tok.text): return False

    if len(re.findall('-', tok.text)) > 1: return False

    if len(tok.text) > 2:
        if tok.pos_ == 'VERB' and tok.tag_ in ['VB', 'VBG', 'VBD', 'VBN']: return True
        if tok.pos_ == 'NOUN' and tok.tag_ in ['NN'] and \
                (tok.text[-3:] in ['ing', 'ion'] or tok.text[-2:] in ['ed']): return True
        if tok.pos_ == 'ADJ' and tok.tag_ in ['JJ']: return True
        if tok.pos_ == 'ADV' and tok.tag_ in ['RB']: return True

    return False

def is_symbol(tok):
    if len(tok.text) == 1 and not tok.is_alpha and not tok.is_digit and tok.text != '.':
        return True
    return False

def is_conj(tok):
    if tok.pos_ in ['SCONJ', 'DET', 'CONJ'] and tok.is_alpha:
        return True
    return False


def is_num_like(tok):
    if len(re.findall('^[0-9]-[a-z]', tok)) != 0 and len(tok) > 10:
        return False

    if all(not c.isalpha() for c in tok): return True
    if tok[0].isdigit() and tok.islower():
        return True

    return False


def is_formula_like(tok):
    elements_1 = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'K', 'V', 'Y', 'I', 'W', 'U']
    elements_2 = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr',
                  'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe',
                  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                  'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                  'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                  'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
                  'Fl', 'Lv']
    r = "[A-Z]{1}[a-z]{0,1}\s*[-*\.\da-z\+/]*"
    if tok[0].isupper() and len(set(re.findall(r, tok)) & set(elements_1 + elements_2)) > 0:
        return True

    return False


def valid_starting(tok):
    if tok.i == 0: return False
    if tok.nbor(-1).text.lower() in ['as', 'the']: return False
    if tok.lemma_ == 'carry' \
        and tok.children.__next__().lemma_ not in ['process', 'experiment', 'synthesis', 'study', 'procedure',
                                                     'treatment', 'preparation']:
        return False

    return True


class OperationsExtractor:
    def __init__(self,
                 w2v_model="models/w2v_embeddings_lemmas_v3",
                 classifier_model="models/fnn-model-1_7classes_dense32_perSentence_3",
                 spacy_model="models/SpaCy_updated_v1.model"):

        print("Operations Extractor v2.9")

        my_folder = os.path.dirname(os.path.realpath(__file__))
        self.__nlp = spacy.load(os.path.join(my_folder, spacy_model))
        self.__embeddings = Word2Vec.load(os.path.join(my_folder, w2v_model))
        self.__model = keras.models.load_model(os.path.join(my_folder, classifier_model))

        ## Building grammatical dictionaries for one-hot grammatical features
        self.__gramma_vector_dict = json.loads(open(os.path.join(my_folder, 'satellites/gramma_vec_dict.json')).read())
        self.__deps_vec = sorted(self.__gramma_vector_dict['deps'].keys())
        self.__deps_len = len(set(self.__deps_vec))

        self.__pos_vec = {}
        pos_exclusion = ['SYM', 'NUM', 'PUNCT', 'PART', 'X', 'INTJ']
        i = 1
        for k in sorted(self.__gramma_vector_dict['pos'].keys()):
            if k in pos_exclusion:
                self.__pos_vec[k] = 0
            else:
                self.__pos_vec[k] = i
                i = i + 1
        self.__pos_len = len(set(self.__pos_vec.values()))

        self.__tags_vec = {}
        symbs_exclusion = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', '#', '``', 'ADD', 'CD', 'FW', 'HYPH', 'LS',
                           'NFP', 'POS', 'RP', '_SP', 'SYM', 'UH', 'XX']
        conjs_exclusion = ['DT', 'TO', 'CC']
        i = 2
        for k in sorted(self.__gramma_vector_dict['tags'].keys()):
            if k not in symbs_exclusion+conjs_exclusion:
                self.__tags_vec[k] = i
                i = i + 1
            else:
                if k in symbs_exclusion:
                    self.__tags_vec[k] = 0
                if k in conjs_exclusion:
                    self.__tags_vec[k] = 1
        self.__tags_len = len(set(self.__tags_vec.values()))

        ## Operations types
        self.__operations = ['NotOperation',
                             'DryingOperation',
                             'HeatingOperation',
                             'Mixing',
                             'QuenchingOperation',
                             'ShapingOperation',
                             'StartingSynthesis']

        ## List of excluding lemmas
        self.__lemmas_exclusion = []
        for line in open(os.path.join(my_folder, 'satellites/excluding_lemmas'), 'r'):
            self.__lemmas_exclusion.append(line[:-1])

        self.__invalid_terms = json.loads(open(os.path.join(my_folder, 'satellites/excluding_terms.json')).read())

        self.__solvent_terms = []
        for line in open(os.path.join(my_folder, 'satellites/aqueous_terms'), 'r'):
            self.__solvent_terms.append(line[:-1])
        self.__solution_verbs = ['dilute',
                                'dissolve',
                                'drop']
        self.__dispersion_verbs = ['disperse',
                                   'pulverize',
                                   'pulverise',
                                   'suspend',
                                   'stir',
                                   'spray',
                                   'moisten'
                                   ]

        self.tf_graph = K.get_session().graph

        print ("Done initialization.")


    def __get_embeddings(self, word):

        if word == '.':
            return [0.0 for _ in range(100)]

        if is_num_like(word):
            word = '<NUM>'

        if is_formula_like(word):
            word = '<CHEM>'

        if word not in self.__embeddings:
            word = '<UNK>'

        return [round(v, 3) for v in self.__embeddings[word]]

    def __get_grammatical_features(self, tok):

        return list(keras.utils.to_categorical(self.__tags_vec[tok.tag_], self.__tags_len)) \
                    + list(keras.utils.to_categorical(self.__pos_vec[tok.pos_], self.__pos_len)) \
                    + list(keras.utils.to_categorical(self.__deps_vec.index(tok.dep_), self.__deps_len))


    def get_operations(self, sentence_tokens):
        """
            finds operations among sentence tokens and classifies them according to its type
        :param sentence_tokens: tokenized sentence
        :return: list of structures dict(token_id, operation_type), tokens parsed by SpaCy
        """

        words_upd, words_ids = remove_parentheses(sentence_tokens.copy())
        spacy_tokens = spacy.tokens.Doc(self.__nlp.vocab, words=sentence_tokens)
        self.__nlp.tagger(spacy_tokens)
        self.__nlp.parser(spacy_tokens)

        reduced_sentence = [spacy_tokens[word_num] for word_num in words_ids]

        sentence_features = []
        valid_tokens = []

        for word_num, word in enumerate(reduced_sentence):
            if valid_token(word) and not word.lemma_ in self.__lemmas_exclusion \
                    and not is_symbol(word) and not is_conj(word):

                embed_vec = self.__get_embeddings(word.lemma_)

                gramma_vec = self.__get_grammatical_features(word)

                sentence_features.append(embed_vec + gramma_vec)
                valid_tokens.append(word_num)

        output = []
        if len(sentence_features) != 0:
            fnn_prediction = self.__model.predict_classes(np.array(sentence_features))

            for word, pred_f in zip(valid_tokens, fnn_prediction):
                if pred_f != 0:
                    output.append((words_ids[word], self.__operations[pred_f]))

        return output, spacy_tokens

    def __tok_to_remove(self, tok, operation):
        if any(c.isdigit() for c in tok.text): return True
        if str(tok.dep_) in ['advmod', 'csubjpass', 'nmod']: return True
        if str(tok.lemma_) in self.__invalid_terms[operation] + ['thermally']: return True
        if operation == 'StartingSynthesis' and tok.dep_ == 'amod': return True
        if 'as-' in tok.text.lower(): return True
        if tok.i > 0:
            if tok.nbor(-1).lemma_ in ['no', 'without']:
                return True

        return False


    def operations_refinement(self, paragraph_, parsed_tokens=False):
        """
            refinement of operations with respect to entire paragraph
            this is fix due to small amount of training data
            use on your own risk
        :param paragraph_: list of tuples (tokenized sentence, operations=get_operations output)
        :param parsed_tokens: True if paragraph sentences are given as tokens parsed by SpaCy (reduces computation time)
        :return: list of tuples (spacy_tokens, operations=get_operations output) with updated operations
        """

        previous_tokens = []
        previous_sentence_tokens = []
        output_structure = []

        if parsed_tokens:
            paragraph = paragraph_
        else:
            paragraph = []
            for sentence, operations in paragraph_:
                spacy_tokens = spacy.tokens.Doc(self.__nlp.vocab, words=sentence)
                self.__nlp.tagger(spacy_tokens)
                self.__nlp.parser(spacy_tokens)
                paragraph.append((spacy_tokens, operations))

        for spacy_tokens, operations in paragraph:
            updated_operations = []


            for op_id, op_type in operations:
                to_remove = False
                if op_type == 'StartingSynthesis':
                    if not valid_starting(spacy_tokens[op_id]):
                        to_remove = True
                    else:
                        previous_tokens = []
                        previous_sentence_tokens = []

                # print (spacy_tokens[op_id])
                # print (op_type)
                # print ('---\n')
                if self.__tok_to_remove(spacy_tokens[op_id], op_type):
                    to_remove = True
                else:
                    if spacy_tokens[op_id].dep_ == 'amod' or 'After' in [a.text for a in spacy_tokens[op_id].ancestors]:
                        if any(op_type == p_op_type for p_op_id, p_op_type, p_tok in previous_tokens + previous_sentence_tokens):
                            to_remove = True
                    else:
                        previous_tokens.append((op_id, op_type, spacy_tokens[op_id]))

                if not to_remove:
                    updated_operations.append((op_id, op_type))

            previous_sentence_tokens = [(op_id, op_type, spacy_tokens[op_id]) for op_id, op_type in operations]
            previous_tokens = []

            output_structure.append((spacy_tokens, updated_operations))

        return output_structure


    def operations_correction(self, sentence, operations, parsed_tokens=False):

        if parsed_tokens:
            sentence_tokens = sentence
        else:
            sentence_tokens = spacy.tokens.Doc(self.__nlp.vocab, words=sentence)
            self.__nlp.tagger(sentence_tokens)
            self.__nlp.parser(sentence_tokens)

        updated_operations = operations.copy()

        ##AGED, DECOMPOSED, REACTED, MELTED
        for k_word in ['aged', 'decomposed','reacted', 'melted']:
            token = [t.i for t in sentence_tokens if t.text == k_word and t.nbor(1).text in ['at', 'for']]
            if token:
                updated_operations = [(op_id, op_type) for op_id, op_type in updated_operations if sentence_tokens[op_id].text != k_word]
                updated_operations.append((token[0], 'HeatingOperation'))
                updated_operations = sorted(updated_operations)

        ## HOT-PRESSED
        k_word = 'pressed'
        try:
            token = [t.i for t in sentence_tokens if t.text == k_word and t.nbor(1).text in ['at', 'for'] and 'hot' in [t.nbor(-1).text, t.nbor(-2).text]]
        except:
            token = []
        if token:
            updated_operations = [(op_id, op_type) for op_id, op_type in updated_operations if sentence_tokens[op_id].text != k_word]
            updated_operations.append((token[0], 'ShapingOperation'))
            updated_operations.append((token[0], 'HeatingOperation'))
            updated_operations = sorted(updated_operations)

        ## DISSOLVED
        for k_word in ['dissolved', 'powdered', 'diluted']:
            token = [t.i for t in sentence_tokens if t.text == k_word]
            if token:
                updated_operations = [(op_id, op_type) for op_id, op_type in updated_operations if sentence_tokens[op_id].text != k_word]
                updated_operations.append((token[0], 'Mixing'))
                updated_operations = sorted(updated_operations)


        ## CORRECTION OF FOLLOWED BY NOUN
        verbs = [t.i for t in sentence_tokens[:-1] if t.lemma_ == 'follow' and t.nbor(1).text == 'by']
        for idx in verbs:
            children = [a for a in sentence_tokens[idx].children]
            if children != [] and [c for c in children[-1].children] != []:
                word = [c for c in children[-1].children][0]
                if word.pos_ == 'NOUN':
                    embed_vec = self.__get_embeddings(word.lemma_)
                    gramma_vec = self.__get_grammatical_features(word)

                    fnn_prediction = self.__model.predict_classes(np.array([embed_vec+gramma_vec]))[0]
                    op_class = self.__operations[fnn_prediction]
                    if op_class != 'NotOperation':
                        updated_operations = [(op_id, op_type) for op_id, op_type in updated_operations if op_id != word.i]
                        updated_operations.append((word.i, op_class))
                        updated_operations = sorted(updated_operations)

        ## CORRECTION OF SINTERING WAS DONE
        sent = ''.join([s.text + ' ' for s in sentence_tokens]).rstrip(' ')
        if 'sintering was' in sent.lower():
            updated_operations = [(op_id, op_type) for op_id, op_type in updated_operations if
                              'sintering' not in sentence_tokens[op_id].text.lower()]
            for token in sentence_tokens[:-2]:
                if 'sintering' in token.text.lower() and token.nbor(2).pos_ == 'VERB':
                    updated_operations.append((token.nbor(2).i, 'HeatingOperation'))
            updated_operations = sorted(updated_operations)

        return updated_operations


    # def __is_solution_mix(self, subtree):
    #     if any(m in subtree for m in self.__solvent_terms + ['wet', 'dropwise']):
    #         return True
    #
    #     return False


    def find_aqueous_mixing(self, sentence, operations, parsed_tokens=False):
        """

        :param sentence:
        :param operations:
        :param parsed_tokens:
        :return:
        """

        updated_operations = operations.copy()

        if parsed_tokens:
            spacy_tokens = sentence
        else:
            spacy_tokens = spacy.tokens.Doc(self.__nlp.vocab, words=sentence)
            self.__nlp.tagger(spacy_tokens)
            self.__nlp.parser(spacy_tokens)

        mixing_operations = [(i, op_id) for i, (op_id, op_type) in enumerate(operations) if op_type == 'Mixing']
        #mixing_operations.reverse()
        #aqueous_mix = []
        solution_mix = []
        grind_in_liquid = []
        regular_mix = []
        for op_id, mix in mixing_operations:

            # subtree splitting
            conj = len(spacy_tokens)
            if list(spacy_tokens[mix].conjuncts) != []:
                conj_tok = list(spacy_tokens[mix].conjuncts)[0]
                if conj_tok.i in [op_id for op_id, op_type in operations]:
                    conj = conj_tok.i

            mix_tree = [t.text for t in spacy_tokens[mix].subtree if t.i < conj]
            if spacy_tokens[mix].lemma_ in self.__solution_verbs or any(w in mix_tree for w in ['wet', 'dropwise']):
                #print('Solution mix:', spacy_toks[mix].text)
                solution_mix.append(op_id)
                continue

            if spacy_tokens[mix].lemma_ in self.__dispersion_verbs:
                #print('Liquid Grind', spacy_toks[mix].text)
                grind_in_liquid.append(op_id)
                continue

            if any(w.lower() in self.__solvent_terms for w in mix_tree):
                if any(w in mix_tree for w in ['ball', 'pestel', 'mortar', 'balls', 'ballmilling']) or\
                    spacy_tokens[mix].lemma_ in ['cruch', 'grind', 'mill', 'milling']:
                    #print('Liquid Grind', spacy_toks[mix].text)
                    grind_in_liquid.append(op_id)
                else:
                    #print('Solution mix', spacy_toks[mix].text)
                    solution_mix.append(op_id)
            else:
                #print('Regular mix', spacy_toks[mix].text)
                regular_mix.append(mix)

        for mix_id in solution_mix:
            op_id, op_type = updated_operations[mix_id]
            updated_operations[mix_id] = (op_id, 'SolutionMixing')

        for mix_id in grind_in_liquid:
            op_id, op_type = updated_operations[mix_id]
            updated_operations[mix_id] = (op_id, 'LiquidGrinding')

        return updated_operations
