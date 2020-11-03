import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import pickle
import spacy
import logging
theano.theano_logger.setLevel(logging.ERROR)

from .utils import shared, set_values, get_name
from .nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
from .optimization import Optimization

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

nlp = spacy.load('en')

class Model(object):
    """
    Network architecture.
    """

    def __init__(self, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.

        :param model_path: File path to reload the model
        """
        # Model location
        self.model_path = model_path
        self.parameters_path = os.path.join(model_path, 'parameters.pkl')
        self.mappings_path = os.path.join(model_path, 'mappings.pkl')
        # Load the parameters and the mappings from disk
        with open(self.parameters_path, 'rb') as f:
            self.parameters = pickle.load(f)

        with open(self.mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.

        :param id_to_word: mapping from a number (id) to a word in text
        :param id_to_char: mapping from a number (id) to a character in a word
        :param id_to_tag: mapping from a number (id) to a tag of word
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            pickle.dump(mappings, f)

    def add_component(self, param):
        """
        Add a new parameter to the network.

        :param param: a dict of parameter names and parameter values
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in list(self.components.items()):
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in list(self.components.items()):
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              ele_num,
              has_CHO,
              topic_dim,
              keyword_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.

        :param dropout: droupout rate
        :param char_dim: dimension of character feature
        :param char_lstm_dim: dimension of hidden layer for lstm dealing with character feature
        :param char_bidirect: use bidirectional lstm for character feature or not
        :param word_dim: dimension of word feature
        :param word_lstm_dim: dimension of hidden layer for lstm dealing with word embedding
        :param word_bidirect: use bidirectional lstm for word recognition or not
        :param lr_method: learning method
        :param pre_emb: pretrained embedding
        :param crf: use crf or not
        :param cap_dim: use capital character feature or not
        :param keyword_dim: dimension of keyword feature
        :param training: training or not
        :param kwargs: customized parameters of model
        :return f_train: training function
        :return f_eval: evaluation function
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')
        if ele_num:
            ele_nums = T.fmatrix(name='ele_nums')
        if has_CHO:
            has_CHOs = T.fmatrix(name='has_CHOs')
        if topic_dim:
            topics = T.fmatrix(name='topics')
        if keyword_dim:
            key_words = T.imatrix(name='key_words')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print('Loading pretrained embeddings from %s...' % pre_emb)
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print('WARNING: %i invalid lines' % emb_invalid)
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # modified
                c_lemma = 0
                # Lookup table initialization
                for i in range(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                    # modified
                    elif word != '<UNK>':
                        word_lemma = nlp(word)[0].lemma_
                        if word_lemma in pretrained:
                            new_weights[i] = pretrained[
                                word_lemma
                            ]
                            c_lemma += 1
                word_layer.embeddings.set_value(new_weights)
                print('Loaded %i pretrained embeddings.' % len(pretrained))
                print(('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros + c_lemma, n_words,
                            100. * (c_found + c_lower + c_zeros + c_lemma) / n_words
                      ))
                print(('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero. %i after lemma.') % (
                          c_found, c_lower, c_zeros, c_lemma
                      ))

        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        # modified appended
        if ele_num:
            input_dim += 1
            inputs.append(ele_nums)

        # modified appended
        if has_CHO:
            input_dim += 1
            inputs.append(has_CHOs)

        #
        # topic feature
        #
        if topic_dim:
            input_dim += topic_dim
            inputs.append(topics)

        #
        # key words feature
        #
        if keyword_dim:
            input_dim += keyword_dim
            inputs.append(key_words)

        # Prepare final input
        inputs = T.concatenate(inputs, axis=1)
        
        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        if ele_num:
            eval_inputs.append(ele_nums)
        if has_CHO:
            eval_inputs.append(has_CHOs)
        if topic_dim:
            eval_inputs.append(topics)
        if keyword_dim:
            eval_inputs.append(key_words)
            pass
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        # print('Compiling...')
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs= [tags_scores, forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),],
                # outputs=forward(observations, transitions, viterbi=True,
                                # return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
