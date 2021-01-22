from pprint import pprint

import numpy as np
import chemdataextractor as CDE
import os
import transformers

from .model_framework import NERModel
from .loader import prepare_sentences
from .utils import iobes_iob, find_one_entity_token_ids, found_package
from .pubchem_utils import load_dico, match_mat_in_dico, solve_conflicts


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'


class MatIdentification(object):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, bert_path='default', pubchem_path=None):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        file_path = os.path.dirname(__file__)
        if model_path is None:
            self.model_path = os.path.join(file_path, '..', 'models/matIdentification')
        else:
            self.model_path = model_path
        if bert_path == 'default':
            self.bert_path = os.path.join(file_path, '..', 'models/Bert_config')
        else:
            self.bert_path = bert_path

        self.model = NERModel.reload_model(model_path=self.model_path, bert_path=self.bert_path)
        if self.bert_path:
            config_template = 'bert-base-cased'
            self.bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=self.bert_path,
                model_max_length=transformers.BertTokenizerFast.max_model_input_sizes[config_template],
                **transformers.BertTokenizerFast.pretrained_init_configuration[config_template]
            )
        else:
            self.bert_tokenizer = None

        if self.model.id_to_word:
            self.word_to_id = {v: k for k, v in list(self.model.id_to_word.items())}
        else:
            self.word_to_id = None

        if self.model.id_to_char:
            self.char_to_id = {v: k for k, v in list(self.model.id_to_char.items())}
        else:
            self.char_to_id = None

        if self.model.id_to_tag:
            self.tag_to_id = {v: k for k, v in list(self.model.id_to_tag.items())}
        else:
            self.tag_to_id = None

        if pubchem_path is None:
            file_path = os.path.dirname(__file__)
            self.pubchem_path = [
                os.path.join(file_path, '..', 'models/pubchem/all_names_compound_4.json' ),
                os.path.join(file_path, '..', 'models/pubchem/all_names_substance_4.json' ),
            ]
        else:
            self.pubchem_path = pubchem_path
        self.mat_dico = None

    def mat_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        all_materials = self.mat_identify_sents([input_sent])
        materials = all_materials[0]
        return materials

    def mat_identify_sents(self, input_sents):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        all_materials = []
        # Prepare input
        data_X, data_Y, data, sentences = prepare_sentences(
            sentences=input_sents,
            word_to_id=self.word_to_id,
            char_to_id=self.char_to_id,
            tag_to_id=self.tag_to_id,
            zeros=self.model.zeros,
            lower=self.model.lower,
            batch_size=self.model.batch_size,
            use_ori_text_char=self.model.use_ori_text_char,
            bert_tokenizer=self.bert_tokenizer,
        )
        # Prediction
        all_y_preds = self.model.predict_label(x_batches=data_X)

        for input_sent, y_preds in zip(input_sents, all_y_preds):
            materials = []
            if self.model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            mat_begin = False
            for tmp_index, y_pred in enumerate(y_preds):
                if y_pred == 'B-Mat':
                    materials.append(input_sent[tmp_index].copy())
                    materials[-1]['token_ids'] = [tmp_index, ]
                    mat_begin = True
                elif y_pred == 'I-Mat' and mat_begin == True:
                    materials[-1]['token_ids'].append(tmp_index)
                    materials[-1]['end'] = input_sent[tmp_index]['end']
                    materials[-1]['text'] += ' ' + input_sent[tmp_index]['text']
                else:
                    mat_begin = False
            all_materials.append(materials)
        return all_materials

    def mat_identify_by_pubchem(self, input_text, sent_tokens):
        """

        :param input_text: a list of sents (materials in sent would be detected,
                            but the input is actually paragraphs because tokens
                            are indexed with paragraphs)
        :param sent_tokens: a list of list of tokens
        :return:
        """
        all_mats = []
        if self.mat_dico is None:
            if self.pubchem_path is not None:
                self.mat_dico = load_dico(self.pubchem_path)
            else:
                self.mat_dico = set()
        for text, tokens in zip(input_text, sent_tokens):

            db_match_mat = match_mat_in_dico(
                tokens=tokens,
                para_text=text,
                dico=self.mat_dico
            )
            db_match_mat = solve_conflicts(db_match_mat)
            all_mats.append(db_match_mat)
        return all_mats


    def mat_identify(self,
                     input_para,
                     pre_tokens=None,
                     use_pubchem=False):
        """
		Identify materials in a paragraph, which is plain text.

        :param input_para: str representing a paragraph
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                     If none, use CDE to get tokens.
		:return materials: dict containing materials from CDE (dict['CDE']) and materials from LSTM (dict['LSTM'])
		"""
        # goal
        materials = []

        if isinstance(input_para, str):
            all_input_paras = [input_para]
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0):
                    raise ValueError('pre_token input should be a list of list (2D list) when input_para is str!')
                pre_tokens = [pre_tokens]
        elif isinstance(input_para, list):
            all_input_paras = input_para
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0 and len(pre_tokens[0][0]) > 0):
                    raise ValueError('pre_token input should be a 3D list when input_para is a list!')
        else:
            raise ValueError('input_para input is neither a str nor a list')

        all_sents = []
        if pre_tokens == None:
            for i, para in enumerate(all_input_paras):
                # CDE tokenization
                CDE_para = CDE.doc.Paragraph(para)
                if len(CDE_para) > 0:
                    for j, tmp_sent in enumerate(CDE_para):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': tmp_sent.text,
                            'tokens': [
                                {
                                    'text': tmp_token.text,
                                    'start': tmp_token.start,
                                    'end': tmp_token.end,
                                } for tmp_token in tmp_sent.tokens
                            ]
                        }
                        all_sents.append(input_sent)
                else:
                    all_sents.append({
                        'paragraph_index': i,
                        'sentence_index': 0,
                        'sentence': para,
                        'tokens': [{'text': 'None', 'start': 0, 'end': 0}],
                    })
        else:
            for i in range(len(all_input_paras)):
                para = all_input_paras[i]
                tokens = pre_tokens[i]
                # get tokens in dict style
                token_style = None
                if len(tokens) > 0 and len(tokens[0]) > 0:
                    if (isinstance(tokens[0][0], dict)
                            and 'text' in tokens[0][0]
                            and 'start' in tokens[0][0]
                            and 'end' in tokens[0][0]):
                        token_style = 'dict'
                    elif ('text' in tokens[0][0].__dict__
                          and 'start' in tokens[0][0].__dict__
                          and 'end' in tokens[0][0].__dict__):
                        token_style = 'attribute'

                if token_style == 'dict':
                    for j, tmp_sent in enumerate(tokens):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': para[tmp_sent[0]['start']: tmp_sent[-1]['end']],
                            'tokens': [
                                {
                                    'text': tmp_token['text'],
                                    'start': tmp_token['start'],
                                    'end': tmp_token['end'],
                                } for tmp_token in tmp_sent
                            ]
                        }
                        all_sents.append(input_sent)
                elif token_style == 'attribute':
                    for j, tmp_sent in enumerate(tokens):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': para[tmp_sent[0].start: tmp_sent[-1].end],
                            'tokens': [
                                {
                                    'text': tmp_token.text,
                                    'start': tmp_token.start,
                                    'end': tmp_token.end,
                                } for tmp_token in tmp_sent
                            ]
                        }
                        all_sents.append(input_sent)
                else:
                    print('Error! Improper input of tokens!')

        all_sent_tokens = [
            list(filter(lambda tmp_token: tmp_token['text'].strip() != '', input_sent['tokens']))
            for input_sent in all_sents
        ]
        mat_in_sents = self.mat_identify_sents(all_sent_tokens)

        if use_pubchem:
            # get materials from existing dico
            all_paras = [all_input_paras[s['paragraph_index']] for s in all_sents]
            mat_from_db = self.mat_identify_by_pubchem(
                input_text=all_paras,
                sent_tokens=all_sent_tokens,
            )
            print('len(mat_from_db)', len(mat_from_db))
            for i in range(len(mat_in_sents)):
                if len(mat_from_db[i]) > 0:
                    mat_in_sents[i] = solve_conflicts(
                        mat_in_sents[i] + mat_from_db[i],
                        useSuperSetToken=True
                    )

        # collect results together and put in list (for paras) of list (for sents)
        for i in range(len(mat_in_sents)):
            input_sent = all_sents[i]
            result = mat_in_sents[i]
            paragraph_index = input_sent['paragraph_index']
            sentence_index = input_sent['sentence_index']
            if len(materials) == 0 or materials[-1]['paragraph_index'] != paragraph_index:
                materials.append({
                    'paragraph_index': paragraph_index,
                    'paragraph': all_input_paras[paragraph_index],
                    'mats_in_para': [],
                })
            materials[-1]['mats_in_para'].append({
                'sentence_index': sentence_index,
                'sentence': input_sent['sentence'],
                'tokens': input_sent['tokens'],
                'materials': result,
            })
        # sort result as input order
        materials = sorted(materials, key=lambda x: x['paragraph_index'])
        for para in materials:
            para['mats_in_para'] = sorted(
                para['mats_in_para'],
                key=lambda x: x['sentence_index']
            )

        # reformated as the exact words in the original paragraph
        # and output as simplified structure
        for para in materials:
            for sent in para['mats_in_para']:
                for tmp_mat in sent['materials']:
                    tmp_mat['text'] = para['paragraph'][tmp_mat['start']: tmp_mat['end']]
                del sent['sentence_index']
        # simplify output
        materials = [para['mats_in_para'] for para in materials]

        if isinstance(input_para, str):
            return materials[0]
        else:
            return materials


class MatTPIdentification(object):
    """
    Use LSTM for materials/target/precursor identification in one step
    """

    def __init__(self, model_path=None, bert_path=None):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        file_path = os.path.dirname(__file__)
        if model_path is None:
            self.model_path = os.path.join(file_path, '..', 'models/matIdentification')
        else:
            self.model_path = model_path
        if bert_path == 'default':
            self.bert_path = os.path.join(file_path, '..', 'models/MATBert_config')
        else:
            self.bert_path = bert_path

        self.model = NERModel.reload_model(model_path=self.model_path, bert_path=self.bert_path)
        if self.bert_path:
            config_template = 'bert-base-cased'
            self.bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=self.bert_path,
                model_max_length=transformers.BertTokenizerFast.max_model_input_sizes[config_template],
                **transformers.BertTokenizerFast.pretrained_init_configuration[config_template]
            )
        else:
            self.bert_tokenizer = None

        if self.model.id_to_word:
            self.word_to_id = {v: k for k, v in list(self.model.id_to_word.items())}
        else:
            self.word_to_id = None

        if self.model.id_to_char:
            self.char_to_id = {v: k for k, v in list(self.model.id_to_char.items())}
        else:
            self.char_to_id = None

        if self.model.id_to_tag:
            self.tag_to_id = {v: k for k, v in list(self.model.id_to_tag.items())}
        else:
            self.tag_to_id = None

    def matTP_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        all_recognition_results = self.matTP_identify_sents([input_sent])
        recognitionResult = all_recognition_results[0]
        return recognitionResult

    def matTP_identify_sents(self, input_sents):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sents: list of list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        all_recognition_results = []
        # constant
        type_to_abbr = {'precursors': 'Pre', 'targets': 'Tar', 'other_materials': 'Mat'}
        abbr_to_type = {v: k for (k, v) in type_to_abbr.items()}
        # Prepare input
        data_X, data_Y, data, sentences = prepare_sentences(
            sentences=input_sents,
            word_to_id=self.word_to_id,
            char_to_id=self.char_to_id,
            tag_to_id=self.tag_to_id,
            zeros=self.model.zeros,
            lower=self.model.lower,
            batch_size=self.model.batch_size,
            use_ori_text_char=self.model.use_ori_text_char,
            bert_tokenizer=self.bert_tokenizer,
        )
        # Prediction
        all_y_preds = self.model.predict_label(x_batches=data_X)

        for input_sent, y_preds in zip(input_sents, all_y_preds):
            recognitionResult = {'all_materials': [], 'precursors': [], 'targets': [], 'other_materials': []}
            if self.model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            mat_begin = None
            for tmp_index, y_pred in enumerate(y_preds):
                if y_pred.startswith('B-'):
                    mat_begin = y_pred[2:]
                    recognitionResult['all_materials'].append(input_sent[tmp_index].copy())
                    recognitionResult['all_materials'][-1]['token_ids'] = [tmp_index, ]
                    recognitionResult[abbr_to_type[mat_begin]].append(input_sent[tmp_index].copy())
                    recognitionResult[abbr_to_type[mat_begin]][-1]['token_ids'] = [tmp_index, ]
                elif y_pred.startswith('I-') and mat_begin == y_pred[2:]:
                    recognitionResult['all_materials'][-1]['token_ids'].append(tmp_index)
                    recognitionResult['all_materials'][-1]['end'] = input_sent[tmp_index]['end']
                    recognitionResult['all_materials'][-1]['text'] += ' ' + input_sent[tmp_index]['text']
                    recognitionResult[abbr_to_type[mat_begin]][-1]['token_ids'].append(tmp_index)
                    recognitionResult[abbr_to_type[mat_begin]][-1]['end'] = input_sent[tmp_index]['end']
                    recognitionResult[abbr_to_type[mat_begin]][-1]['text'] += ' ' + input_sent[tmp_index]['text']
                else:
                    mat_begin = None
            all_recognition_results.append(recognitionResult)
        return all_recognition_results

    def matTP_identify(self, input_para, pre_tokens=None):
        """
        Identify materials in a paragraph, which is plain text.

        :param input_para: str representing a paragraph
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                     If none, use CDE to get tokens.
        :return materials: dict containing materials from CDE (dict['CDE']) and materials from LSTM (dict['LSTM'])
        """
        # goal
        all_results = []
        all_materials = []
        precursors = []
        targets = []
        other_materials = []

        if isinstance(input_para, str):
            all_input_paras = [input_para]
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0):
                    raise ValueError('pre_token input should be a list of list (2D list) when input_para is str!')
                pre_tokens = [pre_tokens]
        elif isinstance(input_para, list):
            all_input_paras = input_para
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0 and len(pre_tokens[0][0]) > 0):
                    raise ValueError('pre_token input should be a 3D list when input_para is a list!')
        else:
            raise ValueError('input_para input is neither a str nor a list')

        all_sents = []
        if pre_tokens == None:
            for i, para in enumerate(all_input_paras):
                # CDE tokenization
                CDE_para = CDE.doc.Paragraph(para)
                if len(CDE_para) > 0:
                    for j, tmp_sent in enumerate(CDE_para):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': tmp_sent.text,
                            'tokens': [
                                {
                                    'text': tmp_token.text,
                                    'start': tmp_token.start,
                                    'end': tmp_token.end,
                                } for tmp_token in tmp_sent.tokens
                            ]
                        }
                        all_sents.append(input_sent)
                else:
                    all_sents.append({
                        'paragraph_index': i,
                        'sentence_index': 0,
                        'sentence': para,
                        'tokens': [{'text': 'None', 'start': 0, 'end': 0}],
                    })
        else:
            for i in range(len(all_input_paras)):
                para = all_input_paras[i]
                tokens = pre_tokens[i]
                # get tokens in dict style
                token_style = None
                if len(tokens) > 0 and len(tokens[0]) > 0:
                    if (isinstance(tokens[0][0], dict)
                            and 'text' in tokens[0][0]
                            and 'start' in tokens[0][0]
                            and 'end' in tokens[0][0]):
                        token_style = 'dict'
                    elif ('text' in tokens[0][0].__dict__
                          and 'start' in tokens[0][0].__dict__
                          and 'end' in tokens[0][0].__dict__):
                        token_style = 'attribute'

                if token_style == 'dict':
                    for j, tmp_sent in enumerate(tokens):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': para[tmp_sent[0]['start']: tmp_sent[-1]['end']],
                            'tokens': [
                                {
                                    'text': tmp_token['text'],
                                    'start': tmp_token['start'],
                                    'end': tmp_token['end'],
                                } for tmp_token in tmp_sent
                            ]
                        }
                        all_sents.append(input_sent)
                elif token_style == 'attribute':
                    for j, tmp_sent in enumerate(tokens):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': para[tmp_sent[0].start: tmp_sent[-1].end],
                            'tokens': [
                                {
                                    'text': tmp_token.text,
                                    'start': tmp_token.start,
                                    'end': tmp_token.end,
                                } for tmp_token in tmp_sent
                            ]
                        }
                        all_sents.append(input_sent)
                else:
                    print('Error! Improper input of tokens!')
                    all_sents = []

        all_sent_tokens = [
            list(filter(lambda tmp_token: tmp_token['text'].strip() != '', input_sent['tokens']))
            for input_sent in all_sents
        ]
        result_in_sents = self.matTP_identify_sents(all_sent_tokens)
        for i in range(len(result_in_sents)):
            input_sent = all_sents[i]
            result = result_in_sents[i]
            paragraph_index = input_sent['paragraph_index']
            sentence_index = input_sent['sentence_index']
            if len(all_results) == 0 or all_results[-1]['paragraph_index'] != paragraph_index:
                all_results.append({
                    'paragraph_index': paragraph_index,
                    'paragraph': all_input_paras[paragraph_index],
                    'results_in_para': [],
                })
            all_results[-1]['results_in_para'].append({
                'sentence_index': sentence_index,
                'sentence': input_sent['sentence'],
                'tokens': input_sent['tokens'],
                'all_materials': result['all_materials'],
                'precursors': result['precursors'],
                'targets': result['targets'],
                'other_materials': result['other_materials'],
            })
        # sort result as input order
        all_results = sorted(all_results, key=lambda x: x['paragraph_index'])
        for para in all_results:
            para['results_in_para'] = sorted(
                para['results_in_para'],
                key=lambda x: x['sentence_index']
            )

        # reformated as the exact words in the original paragraph
        # and output as simplified structure
        for para in all_results:
            for sent in para['results_in_para']:
                for k in {'all_materials', 'precursors', 'targets', 'other_materials'}:
                    for tmp_mat in sent[k]:
                        tmp_mat['text'] = para['paragraph'][tmp_mat['start']: tmp_mat['end']]
                del sent['sentence_index']
        # simplify output
        all_results = [para['results_in_para'] for para in all_results]

        if isinstance(input_para, str):
            return all_results[0]
        else:
            return all_results


class MatRecognition():
    """
	Use LSTM for materials recognition
	"""

    def __init__(self,
                 model_path=None,
                 mat_identify_model_path=None,
                     bert_path='default',
                 mat_identify_bert_path='default',
                 pubchem_path=None):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        :param mat_identify_model_path: path to the model for materials identification. If None input, default initialize.
        :param parse_dependency: parse dependency or not. If True, the parsed dependency will be used as the key word feature.
        """
        file_path = os.path.dirname(__file__)
        if model_path is None:
            self.model_path = os.path.join(file_path, '..', 'models/matRecognition')
        else:
            self.model_path = model_path
        if bert_path == 'default':
            self.bert_path = os.path.join(file_path, '..', 'models/MATBert_config')
        else:
            self.bert_path = bert_path
        if mat_identify_bert_path is not None:
            self.mat_identify_bert_path = mat_identify_bert_path
        else:
            self.mat_identify_bert_path = self.bert_path

        self.model = NERModel.reload_model(model_path=self.model_path, bert_path=self.bert_path)
        if self.bert_path:
            config_template = 'bert-base-cased'
            self.bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=self.bert_path,
                model_max_length=transformers.BertTokenizerFast.max_model_input_sizes[config_template],
                **transformers.BertTokenizerFast.pretrained_init_configuration[config_template]
            )
        else:
            self.bert_tokenizer = None
        print('model_path', model_path)

        if self.model.id_to_word:
            self.word_to_id = {v: k for k, v in list(self.model.id_to_word.items())}
        else:
            self.word_to_id = None

        if self.model.id_to_char:
            self.char_to_id = {v: k for k, v in list(self.model.id_to_char.items())}
        else:
            self.char_to_id = None

        if self.model.id_to_tag:
            self.tag_to_id = {v: k for k, v in list(self.model.id_to_tag.items())}
        else:
            self.tag_to_id = None

        self.identify_model = MatIdentification(
            model_path=mat_identify_model_path,
            pubchem_path=pubchem_path,
            bert_path=self.mat_identify_bert_path
        )

    def mat_recognize_sent(self, input_sent, ori_para_text=''):
        all_recognition_result = self.mat_recognize_sents(
            [input_sent],
            ori_para_text=[ori_para_text]
        )
        recognitionResult = all_recognition_result[0]
        return recognitionResult

    def mat_recognize_sents(self, input_sents, ori_para_text=[], return_scores=False):
        """
		Recognize target/precursor in a sentence, which is a list of tokens.

        :param input_sents: list of list of tokens representing a sentence
		:return recognitionResult: dict containing keys of precursors, targets, and other materials,
				the value of each one is a list of index of token in the sentence
		"""
        # goal
        all_recognition_result = []

        # Prepare input
        if self.model.ele_num or self.model.only_CHO:
            element_feature = True
        else:
            element_feature = False
        # Prepare input
        data_X, data_Y, data, sentences = prepare_sentences(
            sentences=input_sents,
            word_to_id=self.word_to_id,
            char_to_id=self.char_to_id,
            tag_to_id=self.tag_to_id,
            zeros=self.model.zeros,
            lower=self.model.lower,
            element_feature=element_feature,
            batch_size=self.model.batch_size,
            use_ori_text_char=self.model.use_ori_text_char,
            original_para_text=ori_para_text,
            bert_tokenizer=self.bert_tokenizer,
        )
        # Prediction
        all_y_preds = self.model.predict_label(x_batches=data_X)
        if return_scores:
            all_scores = self.model.predict(x_batches=data_X)

        for i in range(len(input_sents)):
            input_sent = input_sents[i]
            y_preds = all_y_preds[i]
            recognitionResult = {'precursors': [], 'targets': [], 'other_materials': []}
            if return_scores:
                scores = all_scores[i]
                recognitionResult['scores'] = scores
            if self.model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            for tmp_index, y_pred in enumerate(y_preds):
                if y_pred == 'B-Pre':
                    recognitionResult['precursors'].append(tmp_index)
                if y_pred == 'B-Tar':
                    recognitionResult['targets'].append(tmp_index)
                if y_pred == 'B-Mat':
                    recognitionResult['other_materials'].append(tmp_index)
            all_recognition_result.append(recognitionResult)
        return all_recognition_result

    def mat_recognize(self,
                      input_para,
                      pre_tokens=None,
                      materials=None,
                      return_scores=False,
                      use_pubchem=False):
        """
		Recognize target/precursor in a paragraph, which is plain text.

        :param input_para: str representing a paragraph or list of str represeting a list of paragraphs
        :param materials: list of materials tokens. If none, use default LSTM model to get materials tokens.
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                            If none, use CDE to get tokens.
                            if input_para is a list, pre_token is a list of list of list of tokens.
		:return mat_to_recognize: list of all materials
        :return precursors: list of all precursors
        :return targets: list of all targets
        :return other_materials: list of all materials other than targets and precursors
        :return results: a list of dict containing the four categories if list of paragraphs input
		"""
        # goal
        all_results = []

        if isinstance(input_para, str):
            all_input_paras = [input_para]
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0):
                    raise ValueError('pre_token input should be a list of list (2D list) when input_para is str!')
                pre_tokens = [pre_tokens]
            if materials != None:
                if not (len(materials) > 0):
                    raise ValueError('materials should be a list when input_para in str!')
        elif isinstance(input_para, list):
            all_input_paras = input_para
            if pre_tokens != None:
                if not (len(pre_tokens) > 0 and len(pre_tokens[0]) > 0 and len(pre_tokens[0][0]) > 0):
                    raise ValueError('pre_token input should be a 3D list when input_para is a list!')
            if materials != None:
                if not (len(materials) > 0 and isinstance(materials[0], list)):
                    raise ValueError('materials should be a list of list (2D list) when input_para in str!')
        else:
            raise ValueError('input_para input is neither a str nor a list')

        if return_scores and isinstance(input_para, str):
            raise ValueError('input_para must be a list when return_scores is True!')


        # if no materials given, use identify_model to generate default materials
        if materials == None:
            mat_to_recognize = self.identify_model.mat_identify(
                all_input_paras,
                pre_tokens=pre_tokens,
                use_pubchem=use_pubchem
            )
        else:
            mat_to_recognize = materials

        all_sents = []
        all_SL_sents = []
        if pre_tokens == None:
            for i in range(len(all_input_paras)):
                para = all_input_paras[i]
                # CDE tokenization
                CDE_para = CDE.doc.Paragraph(para)
                materials_copy = sum([r['materials'] for r in mat_to_recognize[i]], [])
                if len(CDE_para) > 0:
                    for j, tmp_sent in enumerate(CDE_para):
                        # prepare input sentences for LSTM
                        input_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': tmp_sent.text,
                            'tokens': [],
                        }
                        input_SL_sent = {
                            'paragraph_index': i,
                            'sentence_index': j,
                            'sentence': tmp_sent.text,
                            'tokens': [],
                        }
                        tag_started = False
                        for t in tmp_sent.tokens:
                            tmp_token = {'text': t.text, 'start': t.start, 'end': t.end}
                            if tmp_token['text'].strip() == '':
                                continue
                            while (len(materials_copy) > 0):
                                if tmp_token['start'] >= materials_copy[0]['end']:
                                    materials_copy.pop(0)
                                else:
                                    break
                            NER_label = 'O'
                            if len(materials_copy) > 0:
                                if tmp_token['start'] >= materials_copy[0]['start'] and \
                                        tmp_token['end'] <= materials_copy[0]['end']:
                                    # beginning of a material
                                    if tmp_token['start'] == materials_copy[0]['start']:
                                        NER_label = 'B-Mat'
                                        tag_started = True
                                    elif tag_started:
                                        NER_label = 'I-Mat'
                            if NER_label == 'O':
                                input_SL_sent['tokens'].append(tmp_token.copy())
                            elif NER_label == 'B-Mat':
                                input_SL_sent['tokens'].append({
                                    'text': '<MAT>',
                                    'start': materials_copy[0]['start'],
                                    'end': materials_copy[0]['end'],
                                })
                            else:
                                pass
                            input_sent['tokens'].append(tmp_token)
                        all_sents.append(input_sent)
                        all_SL_sents.append(input_SL_sent)
                else:
                    all_sents.append({
                        'paragraph_index': i,
                        'sentence_index': 0,
                        'sentence': para,
                        'tokens': [{'text': 'None', 'start': 0, 'end': 0}],
                    })
                    all_SL_sents.append({
                        'paragraph_index': i,
                        'sentence_index': 0,
                        'sentence': para,
                        'tokens': [{'text': 'None', 'start': 0, 'end': 0}],
                    })
        else:
            for i in range(len(all_input_paras)):
                para = all_input_paras[i]
                tokens = pre_tokens[i]
                materials_copy = sum([r['materials'] for r in mat_to_recognize[i]], [])
                # get tokens in dict style
                token_style = None
                all_tokens = []
                if len(tokens) > 0 and len(tokens[0]) > 0:
                    if (isinstance(tokens[0][0], dict)
                            and 'text' in tokens[0][0]
                            and 'start' in tokens[0][0]
                            and 'end' in tokens[0][0]):
                        token_style = 'dict'
                    elif ('text' in tokens[0][0].__dict__
                          and 'start' in tokens[0][0].__dict__
                          and 'end' in tokens[0][0].__dict__):
                        token_style = 'attribute'

                if token_style == 'dict':
                    for tmp_sent in tokens:
                        # prepare input sentences for LSTM
                        input_sent = [
                            {
                                'text': tmp_token['text'],
                                'start': tmp_token['start'],
                                'end': tmp_token['end'],
                                'paragraph_index': i,
                            }
                            for tmp_token in tmp_sent
                        ]
                        all_tokens.append(input_sent)
                elif token_style == 'attribute':
                    for tmp_sent in tokens:
                        # prepare input sentences for LSTM
                        input_sent = [
                            {
                                'text': tmp_token.text,
                                'start': tmp_token.start,
                                'end': tmp_token.end,
                                'paragraph_index': i,
                            } for tmp_token in tmp_sent
                        ]
                        all_tokens.append(input_sent)
                else:
                    print('Error! Improper input of tokens!')
                    all_tokens = []

                for j, tmp_sent in enumerate(all_tokens):
                    # prepare input sentences for LSTM
                    input_sent = {
                        'paragraph_index': i,
                        'sentence_index': j,
                        'sentence': para[tmp_sent[0]['start']: tmp_sent[-1]['end']],
                        'tokens': [],
                    }
                    input_SL_sent = {
                        'paragraph_index': i,
                        'sentence_index': j,
                        'sentence': para[tmp_sent[0]['start']: tmp_sent[-1]['end']],
                        'tokens': [],
                    }
                    tag_started = False
                    for tmp_token in tmp_sent:
                        if tmp_token['text'].strip() == '':
                            continue
                        while (len(materials_copy) > 0):
                            if tmp_token['start'] >= materials_copy[0]['end']:
                                materials_copy.pop(0)
                            else:
                                break
                        NER_label = 'O'
                        if len(materials_copy) > 0:
                            if tmp_token['start'] >= materials_copy[0]['start'] and \
                                    tmp_token['end'] <= materials_copy[0]['end']:
                                # beginning of a material
                                if tmp_token['start'] == materials_copy[0]['start']:
                                    NER_label = 'B-Mat'
                                    tag_started = True
                                elif tag_started:
                                    NER_label = 'I-Mat'
                        if NER_label == 'O':
                            input_SL_sent['tokens'].append(tmp_token.copy())
                        elif NER_label == 'B-Mat':
                            input_SL_sent['tokens'].append({
                                'text': '<MAT>',
                                'start': materials_copy[0]['start'],
                                'end': materials_copy[0]['end'],
                            })
                        else:
                            pass
                        input_sent['tokens'].append(tmp_token)
                    all_sents.append(input_sent)
                    all_SL_sents.append(input_SL_sent)

        all_SL_sent_tokens = [
            list(filter(lambda tmp_token: tmp_token['text'].strip() != '', input_sent['tokens']))
            for input_sent in all_SL_sents
        ]
        all_paras = [all_input_paras[s['paragraph_index']] for s in all_SL_sents]
        result_in_sents = self.mat_recognize_sents(
            input_sents=all_SL_sent_tokens,
            ori_para_text=all_paras,
            return_scores=return_scores
        )
        for i in range(len(result_in_sents)):
            input_sent = all_sents[i]
            input_SL_sent_tokens = all_SL_sent_tokens[i]
            result = result_in_sents[i]
            paragraph_index = input_sent['paragraph_index']
            sentence_index = input_sent['sentence_index']
            if len(all_results) == 0 or all_results[-1]['paragraph_index'] != paragraph_index:
                all_results.append({
                    'paragraph_index': paragraph_index,
                    'paragraph': all_input_paras[paragraph_index],
                    'results_in_para': [],
                })
            result_in_sent = {
                'sentence_index': sentence_index,
                'sentence': input_sent['sentence'],
                'tokens': input_sent['tokens'],
                'all_materials': mat_to_recognize[paragraph_index][sentence_index]['materials'],
                'precursors': [
                    input_SL_sent_tokens[tmp_index]
                    for tmp_index in result['precursors']
                ],
                'targets': [
                    input_SL_sent_tokens[tmp_index]
                    for tmp_index in result['targets']
                ],
                'other_materials': [
                    input_SL_sent_tokens[tmp_index]
                    for tmp_index in result['other_materials']
                ],
            }
            if return_scores:
                result_in_sent['scores'] = result['scores'].tolist()
                result_in_sent['all_tags'] = list(self.model.all_tags)
            all_results[-1]['results_in_para'].append(result_in_sent)

        # sort result as input order
        all_results = sorted(all_results, key=lambda x: x['paragraph_index'])
        for para in all_results:
            para['results_in_para'] = sorted(
                para['results_in_para'],
                key=lambda x: x['sentence_index']
            )

        # reformated as the exact words in the original paragraph
        # and output as simplified structure
        for para in all_results:
            for sent in para['results_in_para']:
                for k in {'all_materials', 'precursors', 'targets', 'other_materials'}:
                    for tmp_mat in sent[k]:
                        find_one_entity_token_ids(sent['tokens'], tmp_mat)
                        tmp_mat['text'] = para['paragraph'][tmp_mat['start']: tmp_mat['end']]
                del sent['sentence_index']

        # simplify output
        all_results = [para['results_in_para'] for para in all_results]

        if isinstance(input_para, str):
            return all_results[0]
        else:
            return all_results


class MatIdentificationBagging(MatIdentification):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, bert_path=None, bagging=[]):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        self.identify_models = []
        if bagging:
            for tmp_path in bagging:
                self.identify_models.append(
                    MatIdentification(
                        model_path=tmp_path,
                        bert_path=bert_path
                    )
                )
        else:
            self.identify_models.append(
                MatIdentification(
                    model_path=model_path,
                    bert_path=bert_path
                )
            )
        # attention: models should use the same tag_scheme and same tags
        self.get_standard_tags()

    def get_standard_tags(self):
        self.all_tag_to_idx = []
        self.all_idx_to_tag = []
        # each idx in all_idxs corresponds to the order of stardard idx
        self.all_idxs = []
        for tmp_model in self.identify_models:
            tag_to_id = {v: k for (k, v) in tmp_model.model.id_to_tag.items()}
            all_tags = sorted(
                tag_to_id.keys(),
                key=lambda x: tag_to_id[x]
            )
            tag_to_idx = {
                t: i for (i, t) in enumerate(all_tags)
            }
            idx_to_tag = {v: k for (k, v) in tag_to_idx.items()}
            self.all_tag_to_idx.append(tag_to_idx)
            self.all_idx_to_tag.append(idx_to_tag)

        self.standard_tag_to_idx = self.all_tag_to_idx[0]
        self.standard_idx_to_tag = self.all_idx_to_tag[0]
        self.standard_idx = sorted(self.standard_idx_to_tag.keys())

        for i in range(len(self.identify_models)):
            old_idx = [
                self.all_tag_to_idx[i][
                    self.standard_idx_to_tag[tmp_new_idx]
                ]
                for tmp_new_idx in self.standard_idx
            ]
            self.all_idxs.append(old_idx)

    def mat_identify_sents(self, input_sents):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        all_materials = []
        raw_y_preds = []
        raw_tags_scores = []

        for tmp_model in self.identify_models:
            # Prepare input
            data_X, data_Y, data, sentences = prepare_sentences(
                sentences=input_sents,
                word_to_id=tmp_model.word_to_id,
                char_to_id=tmp_model.char_to_id,
                tag_to_id=tmp_model.tag_to_id,
                zeros=tmp_model.model.zeros,
                lower=tmp_model.model.lower,
                batch_size=tmp_model.model.batch_size,
                use_ori_text_char=tmp_model.model.use_ori_text_char,
                bert_tokenizer=tmp_model.bert_tokenizer,
            )

            # Prediction
            y_preds = tmp_model.model.predict_label(x_batches=data_X)
            tags_scores = tmp_model.model.predict(x_batches=data_X)
            raw_y_preds.append(y_preds)
            raw_tags_scores.append(tags_scores)

        for i in range(len(input_sents)):
            input_sent = input_sents[i]
            materials = []
            all_y_preds = []
            all_tags_scores = []

            for j in range(len(self.identify_models)):
                y_preds = raw_y_preds[j][i]
                tags_scores = raw_tags_scores[j][i]

                y_preds_score = np.zeros((len(y_preds), len(self.standard_idx)))
                for k, y_pred in enumerate(y_preds):
                    y_preds_score[
                        k,
                        self.standard_tag_to_idx[y_pred]
                    ] = 1.0
                all_y_preds.append(y_preds_score)

                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:, None]
                tags_scores_normalized[:, self.standard_idx] = tags_scores_normalized[:, self.all_idxs[j]]
                all_tags_scores.append(tags_scores_normalized)

            # bagging
            bagged_idx = []
            all_y_preds = sum(all_y_preds)
            all_tags_scores = sum(all_tags_scores)
            sequence_len = len(all_y_preds)
            for k in range(sequence_len):
                sorted_y_preds = sorted(all_y_preds[k], reverse=True)
                if (sorted_y_preds[0] == sorted_y_preds[1]):
                    bagged_idx.append(np.argmax(all_tags_scores[k]))
                else:
                    bagged_idx.append(np.argmax(all_y_preds[k]))
            y_preds = [self.standard_idx_to_tag[tmp_idx] for tmp_idx in bagged_idx]

            # result
            if self.identify_models[0].model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            mat_begin = False
            for k, y_pred in enumerate(y_preds):
                if y_pred == 'B-Mat':
                    materials.append(input_sent[k])
                    materials[-1]['token_ids'] = [k, ]
                    mat_begin = True
                elif y_pred == 'I-Mat' and mat_begin == True:
                    materials[-1]['token_ids'].append(k)
                    materials[-1]['end'] = input_sent[k]['end']
                    materials[-1]['text'] += ' ' + input_sent[k]['text']
                else:
                    mat_begin = False
            all_materials.append(materials)

        return all_materials


class MatRecognitionBagging(MatRecognition):
    """
    Use LSTM for materials recognition
    """

    def __init__(self,
                 model_path=None,
                 mat_identify_model_path=None,
                 bagging=[],
                 mat_identify_bagging=[],
                 bert_path=None,
                 mat_identify_bert_path=None,
                 ):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        :param mat_identify_model_path: path to the model for materials identification. If None input, default initialize.
        :param parse_dependency: parse dependency or not. If True, the parsed dependency will be used as the key word feature.
        """
        if mat_identify_bagging:
            self.mat_identify_model_path = mat_identify_bagging[0]
        else:
            self.mat_identify_model_path = mat_identify_model_path
        if mat_identify_bert_path is not None:
            self.mat_identify_bert_path = mat_identify_bert_path
        else:
            self.mat_identify_bert_path = self.bert_path

        self.recognition_models = []

        if bagging:
            for tmp_path in bagging:
                self.recognition_models.append(
                    MatRecognition(
                        model_path=tmp_path,
                        mat_identify_model_path=self.mat_identify_model_path,
                        bert_path=bert_path,
                        mat_identify_bert_path=self.mat_identify_bert_path,
                    )
                )
        else:
            self.recognition_models.append(
                MatRecognition(
                    model_path=model_path,
                    mat_identify_model_path=self.mat_identify_model_path,
                    bert_path=bert_path,
                    mat_identify_bert_path=self.mat_identify_bert_path,
                )
            )
        # attention: models should use the same tag_scheme and same tags
        self.get_standard_tags()

        self.identify_model = MatIdentificationBagging(
            model_path=self.mat_identify_model_path,
            bagging=mat_identify_bagging,
            bert_path=self.mat_identify_bert_path,
        )

    def get_standard_tags(self):
        self.all_tag_to_idx = []
        self.all_idx_to_tag = []
        # each idx in all_idxs corresponds to the order of stardard idx
        self.all_idxs = []
        for tmp_model in self.recognition_models:
            tag_to_id = {v: k for (k, v) in tmp_model.model.id_to_tag.items()}
            all_tags = sorted(
                tag_to_id.keys(),
                key=lambda x: tag_to_id[x]
            )
            tag_to_idx = {
                t: i for (i, t) in enumerate(all_tags)
            }
            idx_to_tag = {v: k for (k, v) in tag_to_idx.items()}
            self.all_tag_to_idx.append(tag_to_idx)
            self.all_idx_to_tag.append(idx_to_tag)

        self.standard_tag_to_idx = self.all_tag_to_idx[0]
        self.standard_idx_to_tag = self.all_idx_to_tag[0]
        self.standard_idx = sorted(self.standard_idx_to_tag.keys())

        for i in range(len(self.recognition_models)):
            old_idx = [
                self.all_tag_to_idx[i][
                    self.standard_idx_to_tag[tmp_new_idx]
                ]
                for tmp_new_idx in self.standard_idx
            ]
            self.all_idxs.append(old_idx)

    def mat_recognize_sents(self, input_sents, ori_para_text=[], return_scores=False):
        """
        Recognize target/precursor in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return recognitionResult: dict containing keys of precursors, targets, and other materials,
                the value of each one is a list of index of token in the sentence
        """
        # goal
        all_recognition_result = []
        raw_y_preds = []
        raw_tags_scores = []

        for tmp_model in self.recognition_models:
            # Prepare input
            if tmp_model.model.ele_num or tmp_model.model.only_CHO:
                element_feature = True
            else:
                element_feature = False
            # Prepare input
            data_X, data_Y, data, sentences = prepare_sentences(
                sentences=input_sents,
                word_to_id=tmp_model.word_to_id,
                char_to_id=tmp_model.char_to_id,
                tag_to_id=tmp_model.tag_to_id,
                zeros=tmp_model.model.zeros,
                lower=tmp_model.model.lower,
                element_feature=element_feature,
                batch_size=tmp_model.model.batch_size,
                use_ori_text_char=tmp_model.model.use_ori_text_char,
                original_para_text=ori_para_text,
                bert_tokenizer=tmp_model.bert_tokenizer,
            )
            # Prediction
            y_preds = tmp_model.model.predict_label(x_batches=data_X)
            tags_scores = tmp_model.model.predict(x_batches=data_X)
            raw_y_preds.append(y_preds)
            raw_tags_scores.append(tags_scores)

        for i in range(len(input_sents)):
            input_sent = input_sents[i]
            recognitionResult = {'precursors': [], 'targets': [], 'other_materials': []}
            all_y_preds = []
            all_tags_scores = []

            for j in range(len(self.recognition_models)):
                y_preds = raw_y_preds[j][i]
                tags_scores = raw_tags_scores[j][i]

                y_preds_score = np.zeros((len(y_preds), len(self.standard_idx)))
                for k, y_pred in enumerate(y_preds):
                    y_preds_score[
                        k,
                        self.standard_tag_to_idx[y_pred]
                    ] = 1.0
                all_y_preds.append(y_preds_score)

                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:, None]
                tags_scores_normalized[:, self.standard_idx] = tags_scores_normalized[:, self.all_idxs[j]]
                all_tags_scores.append(tags_scores_normalized)

            # bagging
            bagged_idx = []
            all_y_preds = sum(all_y_preds)
            all_tags_scores = sum(all_tags_scores)
            sequence_len = len(all_y_preds)
            for k in range(sequence_len):
                sorted_y_preds = sorted(all_y_preds[k], reverse=True)
                if (sorted_y_preds[0] == sorted_y_preds[1]):
                    bagged_idx.append(np.argmax(all_tags_scores[k]))
                else:
                    bagged_idx.append(np.argmax(all_y_preds[k]))
            y_preds = [self.standard_idx_to_tag[tmp_idx] for tmp_idx in bagged_idx]

            # result
            if self.recognition_models[0].model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            for tmp_index, y_pred in enumerate(y_preds):
                if y_pred == 'B-Pre':
                    recognitionResult['precursors'].append(tmp_index)
                if y_pred == 'B-Tar':
                    recognitionResult['targets'].append(tmp_index)
                if y_pred == 'B-Mat':
                    recognitionResult['other_materials'].append(tmp_index)
            all_recognition_result.append(recognitionResult)

        return all_recognition_result


class MatTPIdentificationBagging(MatTPIdentification):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, bert_path=None, bagging=[]):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        self.matTP_identify_models = []
        if bagging:
            for tmp_path in bagging:
                self.matTP_identify_models.append(
                    MatTPIdentification(
                        model_path=tmp_path,
                        bert_path=bert_path,
                    )
                )
        else:
            self.matTP_identify_models.append(
                MatTPIdentification(
                    model_path=model_path,
                    bert_path=bert_path,
                )
            )
        # attention: models should use the same tag_scheme and same tags
        self.get_standard_tags()

    def get_standard_tags(self):
        self.all_tag_to_idx = []
        self.all_idx_to_tag = []
        # each idx in all_idxs corresponds to the order of stardard idx
        self.all_idxs = []
        for tmp_model in self.matTP_identify_models:
            tag_to_id = {v: k for (k, v) in tmp_model.model.id_to_tag.items()}
            all_tags = sorted(
                tag_to_id.keys(),
                key=lambda x: tag_to_id[x]
            )
            tag_to_idx = {
                t: i for (i, t) in enumerate(all_tags)
            }
            idx_to_tag = {v: k for (k, v) in tag_to_idx.items()}
            self.all_tag_to_idx.append(tag_to_idx)
            self.all_idx_to_tag.append(idx_to_tag)

        self.standard_tag_to_idx = self.all_tag_to_idx[0]
        self.standard_idx_to_tag = self.all_idx_to_tag[0]
        self.standard_idx = sorted(self.standard_idx_to_tag.keys())

        for i in range(len(self.matTP_identify_models)):
            old_idx = [
                self.all_tag_to_idx[i][
                    self.standard_idx_to_tag[tmp_new_idx]
                ]
                for tmp_new_idx in self.standard_idx
            ]
            self.all_idxs.append(old_idx)

    def matTP_identify_sents(self, input_sents):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        all_recognition_results = []
        raw_y_preds = []
        raw_tags_scores = []

        # constant
        type_to_abbr = {'precursors': 'Pre', 'targets': 'Tar', 'other_materials': 'Mat'}
        abbr_to_type = {v: k for (k, v) in type_to_abbr.items()}

        for tmp_model in self.matTP_identify_models:
            # Prepare input
            data_X, data_Y, data, sentences = prepare_sentences(
                sentences=input_sents,
                word_to_id=tmp_model.word_to_id,
                char_to_id=tmp_model.char_to_id,
                tag_to_id=tmp_model.tag_to_id,
                zeros=tmp_model.model.zeros,
                lower=tmp_model.model.lower,
                batch_size=tmp_model.model.batch_size,
                use_ori_text_char=tmp_model.model.use_ori_text_char,
                bert_tokenizer=tmp_model.bert_tokenizer,
            )

            # Prediction
            y_preds = tmp_model.model.predict_label(x_batches=data_X)
            tags_scores = tmp_model.model.predict(x_batches=data_X)
            raw_y_preds.append(y_preds)
            raw_tags_scores.append(tags_scores)


        for i in range(len(input_sents)):
            input_sent = input_sents[i]
            recognitionResult = {'all_materials': [], 'precursors': [], 'targets': [], 'other_materials': []}
            all_y_preds = []
            all_tags_scores = []

            for j in range(len(self.matTP_identify_models)):
                y_preds = raw_y_preds[j][i]
                tags_scores = raw_tags_scores[j][i]

                y_preds_score = np.zeros((len(y_preds), len(self.standard_idx)))
                for k, y_pred in enumerate(y_preds):
                    y_preds_score[
                        k,
                        self.standard_tag_to_idx[y_pred]
                    ] = 1.0
                all_y_preds.append(y_preds_score)

                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:, None]
                tags_scores_normalized[:, self.standard_idx] = tags_scores_normalized[:, self.all_idxs[j]]
                all_tags_scores.append(tags_scores_normalized)

            # bagging
            bagged_idx = []
            all_y_preds = sum(all_y_preds)
            all_tags_scores = sum(all_tags_scores)
            sequence_len = len(all_y_preds)
            for k in range(sequence_len):
                sorted_y_preds = sorted(all_y_preds[k], reverse=True)
                if (sorted_y_preds[0] == sorted_y_preds[1]):
                    bagged_idx.append(np.argmax(all_tags_scores[k]))
                else:
                    bagged_idx.append(np.argmax(all_y_preds[k]))
            y_preds = [self.standard_idx_to_tag[tmp_idx] for tmp_idx in bagged_idx]

            # result
            if self.matTP_identify_models[0].model.tag_scheme == 'iobes':
                y_preds = iobes_iob(y_preds)
            mat_begin = None
            for k, y_pred in enumerate(y_preds):
                if y_pred.startswith('B-'):
                    mat_begin = y_pred[2:]
                    recognitionResult['all_materials'].append(input_sent[k].copy())
                    recognitionResult['all_materials'][-1]['token_ids'] = [k, ]
                    recognitionResult[abbr_to_type[mat_begin]].append(input_sent[k].copy())
                    recognitionResult[abbr_to_type[mat_begin]][-1]['token_ids'] = [k, ]
                elif y_pred.startswith('I-') and mat_begin == y_pred[2:]:
                    recognitionResult['all_materials'][-1]['token_ids'].append(k)
                    recognitionResult['all_materials'][-1]['end'] = input_sent[k]['end']
                    recognitionResult['all_materials'][-1]['text'] += ' ' + input_sent[k]['text']
                    recognitionResult[abbr_to_type[mat_begin]][-1]['token_ids'].append(k)
                    recognitionResult[abbr_to_type[mat_begin]][-1]['end'] = input_sent[k]['end']
                    recognitionResult[abbr_to_type[mat_begin]][-1]['text'] += ' ' + input_sent[k]['text']
                else:
                    mat_begin = None
            all_recognition_results.append(recognitionResult)

        return all_recognition_results

