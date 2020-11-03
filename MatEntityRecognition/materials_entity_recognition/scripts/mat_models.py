import numpy as np
import chemdataextractor as CDE
import os

from .model import Model
from .loader import prepare_sentence
from .utils import create_input, iobes_iob, find_one_entity_token_ids

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'


class MatIdentification(object):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, emb_path=None):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        if model_path is None:
            file_path = os.path.dirname(__file__)
            model_path = os.path.join(file_path, '..', 'models/matIdentification')
            self.model = Model(model_path=model_path)
        else:
            self.model = Model(model_path=model_path)

        self.parameters = self.model.parameters
        if 'topic_dim' not in self.parameters:
            self.parameters['topic_dim'] = 0
        if 'keyword_dim' not in self.parameters:
            self.parameters['keyword_dim'] = 0
        if 'has_CHO' not in self.model.parameters:
            self.model.parameters['has_CHO'] = False
        if 'ele_num' not in self.model.parameters:
            self.model.parameters['ele_num'] = False
        self.model.parameters['pre_emb'] = emb_path

        self.word_to_id, self.char_to_id, self.tag_to_id = [
            {v: k for k, v in list(x.items())}
            for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
        ]
        _, f_eval = self.model.build(training=False, **self.parameters)
        self.f_eval = f_eval
        self.model.reload()

    def mat_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        materials = []
        # Prepare input
        words = [tmp_token['text'] for tmp_token in input_sent]
        sentence = prepare_sentence(words, self.word_to_id, self.char_to_id,
                                    lower=self.parameters['lower'])
        input = create_input(sentence, self.parameters, False)
        # Prediction
        if self.parameters['crf']:
            tags_scores, output = self.f_eval(*input)
            y_preds = np.array(output)[1:-1]
        else:
            y_preds = self.f_eval(*input).argmax(axis=1)
        y_preds = [self.model.id_to_tag[y_pred] for y_pred in y_preds]
        y_preds = iobes_iob(y_preds)
        mat_begin = False
        for tmp_index, y_pred in enumerate(y_preds):
            if (
                y_pred == 'B-Mat'
                # or (y_pred == 'I-Mat' and mat_begin == False)
            ):
                materials.append(input_sent[tmp_index].copy())
                materials[-1]['token_ids'] = [tmp_index, ]
                mat_begin = True
            elif y_pred == 'I-Mat' and mat_begin == True:
                materials[-1]['token_ids'].append(tmp_index)
                materials[-1]['end'] = input_sent[tmp_index]['end']
                materials[-1]['text'] += ' ' + input_sent[tmp_index]['text']
            else:
                mat_begin = False
        return materials

    def mat_identify(self, input_para, pre_tokens=None):
        """
		Identify materials in a paragraph, which is plain text.
		
        :param input_para: str representing a paragraph
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                     If none, use CDE to get tokens.
		:return materials: dict containing materials from CDE (dict['CDE']) and materials from LSTM (dict['LSTM'])
		"""
        # goal
        result = []

        all_sents = []
        if pre_tokens == None:
            # CDE tokenization
            CDE_para = CDE.doc.Paragraph(input_para)
            for tmp_sent in CDE_para:
                # prepare input sentences for LSTM
                input_sent = [
                    {
                        'text': tmp_token.text,
                        'start': tmp_token.start,
                        'end': tmp_token.end,
                    } for tmp_token in tmp_sent.tokens
                ]
                all_sents.append(input_sent)
        else:
            # get tokens in dict style
            token_style = None
            if len(pre_tokens) > 0 and len(pre_tokens[0]) > 0:
                if isinstance(pre_tokens[0][0], dict) and 'text' in pre_tokens[0][0] and \
                    'start' in pre_tokens[0][0] and 'end' in pre_tokens[0][0]:
                    token_style = 'dict' 
                elif 'text' in pre_tokens[0][0].__dict__ and 'start' in pre_tokens[0][0].__dict__ and \
                    'end' in pre_tokens[0][0].__dict__:
                    token_style = 'attribute'

            if token_style == 'dict':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token['text'], 'start': tmp_token['start'], \
                                    'end': tmp_token['end']} for tmp_token in tmp_sent]
                    all_sents.append(input_sent)
            elif token_style == 'attribute':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token.text, 'start': tmp_token.start, \
                                    'end': tmp_token.end} for tmp_token in tmp_sent]
                    all_sents.append(input_sent)
            else:
                print('Error! Improper input of pre_tokens!')
                all_sents = []

        for input_sent in all_sents:
            input_sent = list(filter(
                lambda tmp_token: tmp_token['text'].strip() != '', 
                input_sent
            ))
            # use LSTM to identify materials
            result.append({
                'sentence': input_para[input_sent[0]['start']: input_sent[-1]['end']],
                'tokens': input_sent,
                'materials': self.mat_identify_sent(input_sent),
            })
            # reformated as the exact words in the original paragraph
            for tmp_mat in result[-1]['materials']:
                tmp_mat['text'] = input_para[tmp_mat['start']: tmp_mat['end']]

        return result


class MatTPIdentification(object):
    """
    Use LSTM for materials/target/precursor identification in one step
    """

    def __init__(self, model_path=None, emb_path=None):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        if model_path is None:
            file_path = os.path.dirname(__file__)
            model_path = os.path.join(file_path, '..', 'models/matIdentification')
            self.model = Model(model_path=model_path)
        else:
            self.model = Model(model_path=model_path)

        self.parameters = self.model.parameters
        if 'topic_dim' not in self.parameters:
            self.parameters['topic_dim'] = 0
        if 'keyword_dim' not in self.parameters:
            self.parameters['keyword_dim'] = 0
        if 'has_CHO' not in self.model.parameters:
            self.model.parameters['has_CHO'] = False
        if 'ele_num' not in self.model.parameters:
            self.model.parameters['ele_num'] = False
        self.model.parameters['pre_emb'] = emb_path

        self.word_to_id, self.char_to_id, self.tag_to_id = [
            {v: k for k, v in list(x.items())}
            for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
        ]
        _, f_eval = self.model.build(training=False, **self.parameters)
        self.f_eval = f_eval
        self.model.reload()

    def matTP_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        recognitionResult = {'all_materials': [], 'precursors': [], 'targets': [], 'other_materials': []}
        type_to_abbr = {'precursors': 'Pre', 'targets': 'Tar', 'other_materials': 'Mat'}
        abbr_to_type = {v: k for (k, v) in type_to_abbr.items()}
        # Prepare input
        words = [tmp_token['text'] for tmp_token in input_sent]
        sentence = prepare_sentence(words, self.word_to_id, self.char_to_id,
                                    lower=self.parameters['lower'])
        input = create_input(sentence, self.parameters, False)
        # Prediction
        if self.parameters['crf']:
            tags_scores, output = self.f_eval(*input)
            y_preds = np.array(output)[1:-1]
        else:
            y_preds = self.f_eval(*input).argmax(axis=1)
        y_preds = [self.model.id_to_tag[y_pred] for y_pred in y_preds]
        y_preds = iobes_iob(y_preds)
        mat_begin = None
        for tmp_index, y_pred in enumerate(y_preds):
            if y_pred.startswith('B-'):
                mat_begin = y_pred[2:]
                recognitionResult['all_materials'].append(input_sent[tmp_index])
                recognitionResult[abbr_to_type[mat_begin]].append(input_sent[tmp_index])
            elif y_pred.startswith('I-') and mat_begin == y_pred[2:]:
                recognitionResult['all_materials'][-1]['end'] = input_sent[tmp_index]['end']
                recognitionResult['all_materials'][-1]['text'] += ' ' + input_sent[tmp_index]['text']
                recognitionResult[abbr_to_type[mat_begin]][-1]['end'] = input_sent[tmp_index]['end']
                recognitionResult[abbr_to_type[mat_begin]][-1]['text'] += ' ' + input_sent[tmp_index]['text']
            else:
                mat_begin = None
        return recognitionResult

    def matTP_identify(self, input_para, pre_tokens=None):
        """
        Identify materials in a paragraph, which is plain text.
        
        :param input_para: str representing a paragraph
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                     If none, use CDE to get tokens.
        :return materials: dict containing materials from CDE (dict['CDE']) and materials from LSTM (dict['LSTM'])
        """
        # goal
        all_materials = []
        precursors = []
        targets = []
        other_materials = []

        all_sents = []
        if pre_tokens == None:
            # CDE tokenization
            CDE_para = CDE.doc.Paragraph(input_para)
            for tmp_sent in CDE_para:
                # prepare input sentences for LSTM
                input_sent = [
                    {'text': tmp_token.text, 'start': tmp_token.start, 'end': tmp_token.end, 'sentence': tmp_sent.text} \
                    for tmp_token in tmp_sent.tokens]
                all_sents.append(input_sent)
        else:
            # get tokens in dict style
            token_style = None
            if len(pre_tokens) > 0 and len(pre_tokens[0]) > 0:
                if isinstance(pre_tokens[0][0], dict) and 'text' in pre_tokens[0][0] and \
                    'start' in pre_tokens[0][0] and 'end' in pre_tokens[0][0]:
                    token_style = 'dict' 
                elif 'text' in pre_tokens[0][0].__dict__ and 'start' in pre_tokens[0][0].__dict__ and \
                    'end' in pre_tokens[0][0].__dict__:
                    token_style = 'attribute'

            if token_style == 'dict':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token['text'], 'start': tmp_token['start'], \
                                    'end': tmp_token['end']} for tmp_token in tmp_sent]
                    all_sents.append(input_sent)
            elif token_style == 'attribute':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token.text, 'start': tmp_token.start, \
                                    'end': tmp_token.end} for tmp_token in tmp_sent]
                    all_sents.append(input_sent)
            else:
                print('Error! Improper input of pre_tokens!')
                all_sents = []

        for input_sent in all_sents:
            input_sent = list(filter(lambda tmp_token: tmp_token['text'].strip() != '', input_sent))
            recognitionResult = self.matTP_identify_sent(input_sent)
            all_materials.extend(recognitionResult['all_materials'])
            precursors.extend(recognitionResult['precursors'])
            targets.extend(recognitionResult['targets'])
            other_materials.extend(recognitionResult['other_materials'])

        # reformated as the exact words in the original paragraph
        for tmp_mat in all_materials + precursors + targets + other_materials:
            # debug
            # tmp_mat['text'] = 'test'
            tmp_mat['text'] = input_para[tmp_mat['start']: tmp_mat['end']]

        return all_materials, precursors, targets, other_materials


class MatRecognition(object):
    """
	Use LSTM for materials recognition
	"""

    def __init__(self, model_path=None, emb_path=None, 
                    mat_identify_model_path=None, mat_identify_emb_path=None, 
                    parse_dependency=False, use_topic=False):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        :param mat_identify_model_path: path to the model for materials identification. If None input, default initialize.
        :param parse_dependency: parse dependency or not. If True, the parsed dependency will be used as the key word feature.
        """
        if model_path == None:
            file_path = os.path.dirname(__file__)
            if parse_dependency:
                self.model = Model(model_path=os.path.join(file_path, '..', 'models/matRecognition2'))
            elif use_topic:
                self.model = Model(model_path=os.path.join(file_path, '..', 'models/matRecognition3'))
            else:
                self.model = Model(model_path=os.path.join(file_path, '..', 'models/matRecognition'))
        else:
            self.model = Model(model_path=model_path)
            print('model_path', model_path)
        if 'topic_dim' not in self.model.parameters:
            self.model.parameters['topic_dim'] = 0
        if 'keyword_dim' not in self.model.parameters:
            self.model.parameters['keyword_dim'] = 0
        if 'has_CHO' not in self.model.parameters:
            self.model.parameters['has_CHO'] = False
        if 'ele_num' not in self.model.parameters:
            self.model.parameters['ele_num'] = False
        self.model.parameters['pre_emb'] = emb_path
        if mat_identify_model_path == None:
            file_path = os.path.dirname(__file__)
            self.identify_model = MatIdentification()
        else:
            self.identify_model = MatIdentification(model_path=mat_identify_model_path, emb_path=mat_identify_emb_path)
        parameters = self.model.parameters
        word_to_id, char_to_id, tag_to_id = [
            {v: k for k, v in list(x.items())}
            for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
        ]
        self.parameters = parameters
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.tag_to_id = tag_to_id
        _, f_eval = self.model.build(training=False, **self.parameters)
        self.f_eval = f_eval
        self.model.reload()

    def mat_recognize_sent(self, input_sent, ori_para_text=''):
        """
		Recognize target/precursor in a sentence, which is a list of tokens.
		
        :param input_sent: list of tokens representing a sentence 
		:return recognitionResult: dict containing keys of precursors, targets, and other materials, 
				the value of each one is a list of index of token in the sentence
		"""
        # goal
        recognitionResult = {'precursors': [], 'targets': [], 'other_materials': []}
        # Prepare input
        words = [tmp_token['text'] for tmp_token in input_sent]
        if self.parameters['has_CHO'] or self.parameters['ele_num']:
            sentence = prepare_sentence(words,
                                        self.word_to_id,
                                        self.char_to_id,
                                        lower=self.parameters['lower'],
                                        use_CHO=self.parameters['has_CHO'],
                                        use_eleNum=self.parameters['ele_num'],
                                        input_tokens=input_sent,
                                        original_para_text=ori_para_text)
        else:
            sentence = prepare_sentence(words,
                                        self.word_to_id,
                                        self.char_to_id,
                                        lower=self.parameters['lower'],
                                        use_CHO=False,
                                        use_eleNum=False)
        input = create_input(sentence, self.parameters, False)
        # Prediction
        if self.parameters['crf']:
            tags_scores, output = self.f_eval(*input)
            y_preds = np.array(output)[1:-1]
        else:
            y_preds = self.f_eval(*input).argmax(axis=1)
        y_preds = [self.model.id_to_tag[y_pred] for y_pred in y_preds]
        y_preds = iobes_iob(y_preds)
        mat_begin = False
        for tmp_index, y_pred in enumerate(y_preds):
            if y_pred == 'B-Pre':
                recognitionResult['precursors'].append(tmp_index)
            if y_pred == 'B-Tar':
                recognitionResult['targets'].append(tmp_index)
            if y_pred == 'B-Mat':
                recognitionResult['other_materials'].append(tmp_index)
        return recognitionResult

    def mat_recognize(self, input_para, materials=None, pre_tokens=None):
        """
		Recognize target/precursor in a paragraph, which is plain text.
		
        :param input_para: str representing a paragraph
        :param materials: list of list of materials in each sentence. If none, use default LSTM model to get materials tokens.
        :param pre_tokens: list of list of tokens. Each list inside is a sentence.
                            If none, use CDE to get tokens.
		:return mat_to_recognize: list of all materials
        :return precursors: list of all precursors
        :return targets: list of all targets
        :return other_materials: list of all materials other than targets and precursors
		"""
        # goal
        result = []

        # if no materials given, use identify_model to generate default materials
        if materials == None:
            if pre_tokens == None:
                mat_to_recognize = self.identify_model.mat_identify(input_para)
            else:
                mat_to_recognize = self.identify_model.mat_identify(input_para, pre_tokens=pre_tokens)
        else:
            mat_to_recognize = materials
        materials_copy =sum([r['materials'] for r in mat_to_recognize], [])
        all_sents = []
        all_SL_sents = []
        if pre_tokens == None:
            # CDE tokenization
            CDE_para = CDE.doc.Paragraph(input_para)

            for tmp_sent in CDE_para:
                # prepare input sentences for LSTM
                input_sent = []
                input_SL_sent = []
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
                        input_SL_sent.append(tmp_token.copy())
                    elif NER_label == 'B-Mat':
                        input_SL_sent.append({
                            'text': '<MAT>', 
                            'start': materials_copy[0]['start'], 
                            'end': materials_copy[0]['end'],
                        })
                    else:
                        pass
                    input_sent.append(tmp_token)
                all_sents.append(input_sent)
                all_SL_sents.append(input_SL_sent)
        else:
            # get tokens in dict style
            token_style = None
            all_tokens = []
            if len(pre_tokens) > 0 and len(pre_tokens[0]) > 0:
                if isinstance(pre_tokens[0][0], dict) and 'text' in pre_tokens[0][0] and \
                    'start' in pre_tokens[0][0] and 'end' in pre_tokens[0][0]:
                    token_style = 'dict' 
                elif 'text' in pre_tokens[0][0].__dict__ and 'start' in pre_tokens[0][0].__dict__ and \
                    'end' in pre_tokens[0][0].__dict__:
                    token_style = 'attribute'

            if token_style == 'dict':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token['text'], 'start': tmp_token['start'], \
                                   'end': tmp_token['end']} for tmp_token in tmp_sent]
                    all_tokens.append(input_sent)
            elif token_style == 'attribute':
                for tmp_sent in pre_tokens:
                    # prepare input sentences for LSTM
                    input_sent = [{'text': tmp_token.text, 'start': tmp_token.start, \
                                    'end': tmp_token.end} for tmp_token in tmp_sent]
                    all_tokens.append(input_sent)
            else:
                print('Error! Improper input of pre_tokens!')
                all_tokens = []

            for tmp_sent in all_tokens:
                # prepare input sentences for LSTM
                input_sent = []
                input_SL_sent = []
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
                        input_SL_sent.append(tmp_token.copy())
                    elif NER_label == 'B-Mat':
                        input_SL_sent.append({
                            'text': '<MAT>', 
                            'start': materials_copy[0]['start'], 
                            'end': materials_copy[0]['end'],
                        })
                    else:
                        pass
                    input_sent.append(tmp_token)
                all_sents.append(input_sent)
                all_SL_sents.append(input_SL_sent)

        for i, input_SL_sent in enumerate(all_SL_sents):
            input_sent = all_sents[i]
            sent_text = input_para[input_sent[0]['start']: input_sent[-1]['end']]
            if self.parameters['has_CHO'] or self.parameters['ele_num']:    
                recognitionResult = self.mat_recognize_sent(input_SL_sent, ori_para_text=input_para)
            else:
                recognitionResult = self.mat_recognize_sent(input_SL_sent)
            result.append({
                'sentence': sent_text,
                'tokens': input_sent,
                'all_materials': mat_to_recognize[i]['materials'],
                'targets': [input_SL_sent[tmp_index] for tmp_index in recognitionResult['targets']],
                'precursors': [input_SL_sent[tmp_index] for tmp_index in recognitionResult['precursors']],
                'other_materials': [input_SL_sent[tmp_index] for tmp_index in recognitionResult['other_materials']],
            })
            # reformated as the exact words in the original paragraph
            for mat_type in {'all_materials', 'precursors', 'targets', 'other_materials'}:
                for tmp_mat in result[-1][mat_type]:
                    find_one_entity_token_ids(input_sent, tmp_mat)
                    tmp_mat['text'] = input_para[tmp_mat['start']: tmp_mat['end']]

        return result


class MatIdentificationBagging(MatIdentification):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, bagging=[]):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        self.identify_models = []
        if bagging:
            for tmp_path in bagging:
                self.identify_models.append(
                    MatIdentification(
                        model_path=tmp_path,
                        )
                    )
        else:
            self.identify_models.append(
                MatIdentification(
                    model_path=model_path,
                    )
                )

    def mat_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        materials = []

        all_y_preds = []
        all_tags_scores = []

        standard_id_to_tag = self.identify_models[0].model.id_to_tag
        standard_tag_to_id = {v:k for (k, v) in standard_id_to_tag.items()}

        words = [tmp_token['text'] for tmp_token in input_sent]
        for tmp_model in self.identify_models:
            # Prepare input
            sentence = prepare_sentence(words, tmp_model.word_to_id, tmp_model.char_to_id,
                                        lower=tmp_model.parameters['lower'])
            input = create_input(sentence, tmp_model.parameters, False)
            # Prediction
            if tmp_model.parameters['crf']:
                tags_scores, output = tmp_model.f_eval(*input)
                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:,None]
                old_ids = sorted(tmp_model.model.id_to_tag.keys())
                new_ids = [
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[
                            tmp_old_id
                        ]
                    ]
                    for tmp_old_id in old_ids
                ]
                tags_scores_normalized[:, new_ids] = tags_scores_normalized[:, old_ids]
                all_tags_scores.append(tags_scores_normalized)
                y_preds = np.array(output)[1:-1]
                # y_preds = np.array(tmp_model.f_eval(*input))[1:-1]
            else:
                y_preds = tmp_model.f_eval(*input).argmax(axis=1)

            y_preds_score = np.zeros((len(y_preds), len(tmp_model.model.id_to_tag)))
            for i, y_pred in enumerate(y_preds):
                y_preds_score[
                    i,
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[y_pred]
                        ]
                ] = 1.0

            all_y_preds.append(y_preds_score)

        # bagging
        bagged_ids = []
        all_y_preds = sum(all_y_preds)
        if all_tags_scores:
            all_tags_scores = sum(all_tags_scores)
        sequence_len = len(all_y_preds)
        for i in range(sequence_len):
            sorted_y_preds = sorted(all_y_preds[i], reverse=True)
            if (sorted_y_preds[0] == sorted_y_preds[1]) \
                and (len(all_tags_scores) > 0):
                bagged_ids.append(np.argmax(all_tags_scores[i]))
            else:
                bagged_ids.append(np.argmax(all_y_preds[i]))
        y_preds = [standard_id_to_tag[tmp_id] for tmp_id in bagged_ids]

        # result
        y_preds = iobes_iob(y_preds)
        mat_begin = False
        for tmp_index, y_pred in enumerate(y_preds):
            if y_pred == 'B-Mat':
                materials.append(input_sent[tmp_index])
                mat_begin = True
            elif y_pred == 'I-Mat' and mat_begin == True:
                materials[-1]['end'] = input_sent[tmp_index]['end']
                materials[-1]['text'] += ' ' + input_sent[tmp_index]['text']
            else:
                mat_begin = False
        return materials


class MatRecognitionBagging(MatRecognition):
    """
    Use LSTM for materials recognition
    """

    def __init__(self, model_path=None,
                    mat_identify_model_path=None,
                    parse_dependency=False, use_topic=False,
                    bagging=[], mat_identify_bagging=[]):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        :param mat_identify_model_path: path to the model for materials identification. If None input, default initialize.
        :param parse_dependency: parse dependency or not. If True, the parsed dependency will be used as the key word feature.
        """
        self.recognition_models = []

        if bagging:
            for tmp_path in bagging:
                self.recognition_models.append(
                    MatRecognition(
                        model_path=tmp_path,
                        mat_identify_model_path=mat_identify_model_path,
                        parse_dependency=parse_dependency,
                        use_topic=use_topic,
                        )
                    )
        else:
            self.recognition_models.append(
                MatRecognition(
                    model_path=model_path,
                    mat_identify_model_path=mat_identify_model_path,
                    parse_dependency=parse_dependency,
                    use_topic=use_topic,
                    )
                )

        self.parameters = self.recognition_models[0].parameters.copy()
        valid_keys = {'has_CHO', 'ele_num'}
        self.parameters = dict(filter(lambda x: x[0] in valid_keys, self.parameters.items()))

        self.identify_model = MatIdentificationBagging(
            model_path=mat_identify_model_path,
            bagging=mat_identify_bagging,
            )


    def mat_recognize_sent(self, input_sent, ori_para_text=''):
        """
        Recognize target/precursor in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return recognitionResult: dict containing keys of precursors, targets, and other materials,
                the value of each one is a list of index of token in the sentence
        """
        # goal
        recognitionResult = {'precursors': [], 'targets': [], 'other_materials': []}
        all_y_preds = []
        all_tags_scores = []

        standard_id_to_tag = self.recognition_models[0].model.id_to_tag
        standard_tag_to_id = {v:k for (k, v) in standard_id_to_tag.items()}

        words = [tmp_token['text'] for tmp_token in input_sent]
        for tmp_model in self.recognition_models:
            # Prepare input
            if tmp_model.parameters['has_CHO'] or tmp_model.parameters['ele_num']:
                sentence = prepare_sentence(words,
                                            tmp_model.word_to_id,
                                            tmp_model.char_to_id,
                                            lower=tmp_model.parameters['lower'],
                                            use_CHO=tmp_model.parameters['has_CHO'],
                                            use_eleNum=tmp_model.parameters['ele_num'],
                                            input_tokens=input_sent,
                                            original_para_text=ori_para_text)
            else:
                sentence = prepare_sentence(words,
                                            tmp_model.word_to_id,
                                            tmp_model.char_to_id,
                                            lower=tmp_model.parameters['lower'],
                                            use_CHO=False,
                                            use_eleNum=False)

            input = create_input(sentence, tmp_model.parameters, False)
            # Prediction
            if tmp_model.parameters['crf']:
                tags_scores, output = tmp_model.f_eval(*input)
                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:,None]
                old_ids = sorted(tmp_model.model.id_to_tag.keys())
                new_ids = [
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[
                            tmp_old_id
                        ]
                    ]
                    for tmp_old_id in old_ids
                ]
                tags_scores_normalized[:, new_ids] = tags_scores_normalized[:, old_ids]
                all_tags_scores.append(tags_scores_normalized)
                y_preds = np.array(output)[1:-1]
                # y_preds = np.array(tmp_model.f_eval(*input))[1:-1]
            else:
                y_preds = tmp_model.f_eval(*input).argmax(axis=1)

            y_preds_score = np.zeros((len(y_preds), len(tmp_model.model.id_to_tag)))
            for i, y_pred in enumerate(y_preds):
                y_preds_score[
                    i,
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[y_pred]
                        ]
                ] = 1.0

            all_y_preds.append(y_preds_score)

        # bagging
        bagged_ids = []
        all_y_preds = sum(all_y_preds)
        if all_tags_scores:
            all_tags_scores = sum(all_tags_scores)
        sequence_len = len(all_y_preds)
        for i in range(sequence_len):
            sorted_y_preds = sorted(all_y_preds[i], reverse=True)
            if (sorted_y_preds[0] == sorted_y_preds[1]) \
                and (len(all_tags_scores) > 0):
                bagged_ids.append(np.argmax(all_tags_scores[i]))
            else:
                bagged_ids.append(np.argmax(all_y_preds[i]))
        y_preds = [standard_id_to_tag[tmp_id] for tmp_id in bagged_ids]

        # result
        y_preds = iobes_iob(y_preds)
        mat_begin = False
        for tmp_index, y_pred in enumerate(y_preds):
            if y_pred == 'B-Pre':
                recognitionResult['precursors'].append(tmp_index)
            if y_pred == 'B-Tar':
                recognitionResult['targets'].append(tmp_index)
            if y_pred == 'B-Mat':
                recognitionResult['other_materials'].append(tmp_index)
        return recognitionResult


class MatTPIdentificationBagging(MatTPIdentification):
    """
    Use LSTM for materials identification
    """

    def __init__(self, model_path=None, bagging=[]):
        """
        :param model_path: path to the model for materials recognition. If None input, default initialize.
        """
        self.matTP_identify_models = []
        if bagging:
            for tmp_path in bagging:
                self.matTP_identify_models.append(
                    MatTPIdentification(
                        model_path=tmp_path,
                        )
                    )
        else:
            self.matTP_identify_models.append(
                MatTPIdentification(
                    model_path=model_path,
                    )
                )

    def matTP_identify_sent(self, input_sent):
        """
        Identify materials in a sentence, which is a list of tokens.

        :param input_sent: list of tokens representing a sentence
        :return materials: list of materials from LSTM
        """
        # goal
        recognitionResult = {'all_materials': [], 'precursors': [], 'targets': [], 'other_materials': []}
        type_to_abbr = {'precursors': 'Pre', 'targets': 'Tar', 'other_materials': 'Mat'}
        abbr_to_type = {v: k for (k, v) in type_to_abbr.items()}

        all_y_preds = []
        all_tags_scores = []

        standard_id_to_tag = self.matTP_identify_models[0].model.id_to_tag
        standard_tag_to_id = {v:k for (k, v) in standard_id_to_tag.items()}

        words = [tmp_token['text'] for tmp_token in input_sent]
        for tmp_model in self.matTP_identify_models:
            # Prepare input
            sentence = prepare_sentence(words, tmp_model.word_to_id, tmp_model.char_to_id,
                                        lower=tmp_model.parameters['lower'])
            input = create_input(sentence, tmp_model.parameters, False)
            # Prediction
            if tmp_model.parameters['crf']:
                tags_scores, output = tmp_model.f_eval(*input)
                tags_scores_exp = np.exp(tags_scores)
                tags_scores_normalized = tags_scores_exp / np.sum(tags_scores_exp, axis=1)[:,None]
                old_ids = sorted(tmp_model.model.id_to_tag.keys())
                new_ids = [
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[
                            tmp_old_id
                        ]
                    ]
                    for tmp_old_id in old_ids
                ]
                tags_scores_normalized[:, new_ids] = tags_scores_normalized[:, old_ids]
                all_tags_scores.append(tags_scores_normalized)
                y_preds = np.array(output)[1:-1]
                # y_preds = np.array(tmp_model.f_eval(*input))[1:-1]
            else:
                y_preds = tmp_model.f_eval(*input).argmax(axis=1)

            y_preds_score = np.zeros((len(y_preds), len(tmp_model.model.id_to_tag)))
            for i, y_pred in enumerate(y_preds):
                y_preds_score[
                    i,
                    standard_tag_to_id[
                        tmp_model.model.id_to_tag[y_pred]
                        ]
                ] = 1.0

            all_y_preds.append(y_preds_score)

        # bagging
        bagged_ids = []
        all_y_preds = sum(all_y_preds)
        if all_tags_scores:
            all_tags_scores = sum(all_tags_scores)
        sequence_len = len(all_y_preds)
        for i in range(sequence_len):
            sorted_y_preds = sorted(all_y_preds[i], reverse=True)
            if (sorted_y_preds[0] == sorted_y_preds[1]) \
                and (len(all_tags_scores) > 0):
                bagged_ids.append(np.argmax(all_tags_scores[i]))
            else:
                bagged_ids.append(np.argmax(all_y_preds[i]))
        y_preds = [standard_id_to_tag[tmp_id] for tmp_id in bagged_ids]

        # result
        y_preds = iobes_iob(y_preds)
        mat_begin = None
        for tmp_index, y_pred in enumerate(y_preds):
            if y_pred.startswith('B-'):
                mat_begin = y_pred[2:]
                recognitionResult['all_materials'].append(input_sent[tmp_index])
                recognitionResult[abbr_to_type[mat_begin]].append(input_sent[tmp_index])
            elif y_pred.startswith('I-') and mat_begin == y_pred[2:]:
                recognitionResult['all_materials'][-1]['end'] = input_sent[tmp_index]['end']
                recognitionResult['all_materials'][-1]['text'] += ' ' + input_sent[tmp_index]['text']
                recognitionResult[abbr_to_type[mat_begin]][-1]['end'] = input_sent[tmp_index]['end']
                recognitionResult[abbr_to_type[mat_begin]][-1]['text'] += ' ' + input_sent[tmp_index]['text']
            else:
                mat_begin = None

        return recognitionResult

