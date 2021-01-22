import inspect
import os
import numpy as np
import pickle
import regex

import tensorflow as tf
from tensorflow import keras
from transformers import BertConfig, TFBertMainLayer

from .utils import found_package
import transformers
import tensorflow_addons as tfa

from .utils import iobes_iob, parse_lr_method
from .conlleval_perl import evaluate_lines
from .layers import SimpleSequenceScores, RNNSequenceScores, CharacterFeatures
from .losses import PlainLoss
from .loader import get_input_format, dict_to_tf_dataset
from . import bert_optimization

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]

class NERModel(keras.Model):
    """
    Network architecture.
    """

    def __init__(self,
                 model_path=None,
                 model_name=None,
                 to_reload_model=False,
                 tag_scheme='iobes',
                 clean_tag=True,
                 id_to_word=None,
                 id_to_char=None,
                 id_to_tag=None,
                 classifier_type='lstm',
                 bert_path=None,
                 bert_first_trainable_layer=0,
                 word_dim=100,
                 word_lstm_dim=100,
                 word_bidirect=True,
                 word_unroll=False,
                 word_rnn_wrapper=False,
                 char_dim=25,
                 char_lstm_dim=25,
                 char_bidirect=True,
                 char_combine_method='concat',
                 char_unroll=False,
                 char_rnn_wrapper=False,
                 # cap_dim,
                 ele_num=True,
                 only_CHO=True,
                 tar_tag=False,
                 pre_tag=False,
                 # topic_dim,
                 # keyword_dim,
                 rnn_type='gru',
                 lower=False,
                 zeros=False,
                 use_ori_text_char=False,
                 crf=True,
                 crf_begin_end=True,
                 dropout=0.5,
                 pre_embedding=None,
                 lr_method='sgd-lr_.005',
                 loss_per_token=False,
                 batch_size=1,
                 num_epochs=100,
                 steps_per_epoch=3500,
                 **kwargs):
        """
        Initialize the model. We can init a empty model with model_name, or reload
        a pre-trained model from model_path

        :param model_path: File path to reload the model. If specified, model will be 
        reloaded from model_path, and model_name will be discarded.
        :param model_name: Name of the model. If specified, the model will save in a 
        folder called model_name. 
        """
        super().__init__(**kwargs)
        # Model location
        parent_folder = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..'
            )
        )
        if not model_path:
            if model_name:
                self.model_path = os.path.join(parent_folder, 'generated',  model_name)
            else:
                self.model_path = os.path.join(parent_folder, 'generated', 'model_1')
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.parameters_path = os.path.join(self.model_path, 'parameters.pkl')
        else:
            # reload model parameters
            self.model_path = model_path
            self.parameters_path = os.path.join(self.model_path, 'parameters.pkl')

        self.last_cp_dir = os.path.join(self.model_path, 'last_cp')
        if not os.path.exists(self.last_cp_dir):
            os.makedirs(self.last_cp_dir)
        self.last_cp_path = os.path.join(self.last_cp_dir, 'cp.ckpt')
        self.opt_cp_dir = os.path.join(self.model_path, 'opt_cp')
        if not os.path.exists(self.opt_cp_dir):
            os.makedirs(self.opt_cp_dir)
        self.opt_cp_path = os.path.join(self.opt_cp_dir, 'cp.ckpt')

        self.tag_scheme = tag_scheme
        self.clean_tag = clean_tag
        self.classifier_type = classifier_type
        self.bert_path = bert_path
        self.bert_first_trainable_layer = bert_first_trainable_layer
        self.word_dim = word_dim
        self.word_lstm_dim = word_lstm_dim
        self.word_bidirect = word_bidirect
        self.word_unroll = word_unroll
        self.word_rnn_wrapper = word_rnn_wrapper
        self.char_dim = char_dim
        self.char_lstm_dim = char_lstm_dim
        self.char_bidirect = char_bidirect
        self.char_combine_method = char_combine_method
        self.char_unroll = char_unroll
        self.char_rnn_wrapper = char_rnn_wrapper
        self.ele_num = ele_num
        self.only_CHO = only_CHO
        self.tar_tag = tar_tag
        self.pre_tag = pre_tag
        self.rnn_type = rnn_type
        self.lower = lower
        self.zeros = zeros
        self.use_ori_text_char = use_ori_text_char
        self.crf = crf
        self.crf_begin_end = crf_begin_end
        self.dropout = dropout
        self.pre_embedding = pre_embedding
        self.lr_method = lr_method
        self.loss_per_token = loss_per_token
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        assert id_to_tag
        if not self.bert_path:
            assert id_to_word and id_to_char and id_to_tag
        self.create_mappings(
            id_to_word=id_to_word,
            id_to_char=id_to_char,
            id_to_tag=id_to_tag,
        )
        if not to_reload_model:
            # TODO: need to think about how to save bert model
            self.save_model()

        # we will use mask defaultly
        # therefore, id of words and tags should starts from 1
        # add embedding for mask
        if self.pre_embedding is not None:
            self.word_dim = np.shape(self.pre_embedding)[1]
            self.pre_embedding_matrix = np.concatenate(
                (
                    np.zeros((1, self.word_dim)),
                    self.pre_embedding
                )
            )

        lr_method_name, lr_method_parameters = parse_lr_method(self.lr_method)
        print('lr_method_name, lr_method_parameters',
              lr_method_name, lr_method_parameters)
        if lr_method_name == 'sgd':
            self.optimizer = keras.optimizers.SGD(
                learning_rate=lr_method_parameters.get('lr', 0.005),
            )
        elif lr_method_name == 'adam':
            self.optimizer = keras.optimizers.Adam(
                learning_rate=lr_method_parameters.get('lr', 5e-5),
            )
        elif lr_method_name == 'adamdecay':
            num_train_steps = self.steps_per_epoch * self.num_epochs
            num_warmup_steps = int(
                num_train_steps * lr_method_parameters.get('warmup', 0.05)
            )
            print('num_train_steps, num_warmup_steps',
                  num_train_steps, num_warmup_steps)
            self.optimizer = bert_optimization.create_optimizer(
                init_lr=lr_method_parameters.get('lr', 5e-5),
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                epsilon=lr_method_parameters.get('epsilon', 1e-6),
                weight_decay_rate=lr_method_parameters.get('weight_decay_rate', 0.01),
            )
        else:
            raise ValueError(
                "Not implemented learning method: {}".format(lr_method_name)
            )

        # TODO: to be removed after Huggingface is updated and TF>=2.4 is used.
        #  This is added due to some incompatibility between keras and huggingface.
        #  See: https://github.com/huggingface/transformers/issues/6560
        self.optimizer._HAS_AGGREGATE_GRAD = False

        self.loss = PlainLoss(name='loss_layer')

        self.create_constants()
        # add layers
        if self.bert_path:
            if to_reload_model:
                bert_config = BertConfig.from_pretrained(self.bert_path)
                self.bert_layer = TFBertMainLayer(bert_config, name="bert")
            else:
                self.bert_layer = transformers.TFBertModel.from_pretrained(
                    self.bert_path,
                    from_pt=True,
                ).bert
            self.set_bert_trainable_layers(self.bert_layer, self.bert_first_trainable_layer)

            assert self.word_dim == 0
            assert self.char_dim == 0
            assert pre_embedding == None

        if self.pre_embedding is not None:
            # init embedding layer
            # https://stackoverflow.com/questions/47057361/keras-using-tensorflow-backend-masking-on-loss-function
            # mapping ids always start from 1
            # To use mapping directly, mask_zero should always be ture
            self.word_layer = keras.layers.Embedding(
                input_dim=self.n_words+1,
                output_dim=self.word_dim,
                embeddings_initializer=keras.initializers.Constant(
                    self.pre_embedding_matrix
                ),
                mask_zero=True,
                trainable=True,
                name='word_layer',
            )
        elif self.word_dim:
            self.word_layer = keras.layers.Embedding(
                input_dim=self.n_words+1,
                output_dim=self.word_dim,
                embeddings_initializer='glorot_uniform',
                mask_zero=True,
                trainable=True,
                name='word_layer',
            )

        if self.dropout:
            self.dropout_layer = keras.layers.Dropout(self.dropout)

        # TODO: it is better to assign a name to each layer in the future
        if self.char_dim:
            self.char_layer = CharacterFeatures(
                n_chars=self.n_chars,
                char_dim=self.char_dim,
                char_lstm_dim=self.char_lstm_dim,
                char_bidirect=self.char_bidirect,
                char_combine_method=self.char_combine_method,
                batch_size=self.batch_size,
                rnn_type=self.rnn_type,
                unroll=self.char_unroll,
                rnn_wrapper=self.char_rnn_wrapper,
            )

        if self.classifier_type == 'simple':
            print('classifier_type', classifier_type)
            # activation function here is always None
            self.unary_score_layer = SimpleSequenceScores(
                n_tags=self.n_tags,
            )
        elif self.classifier_type == 'lstm':
            print('classifier_type', classifier_type)
            self.unary_score_layer = RNNSequenceScores(
                word_bidirect=self.word_bidirect,
                word_lstm_dim=self.word_lstm_dim,
                n_tags=self.n_tags,
                rnn_type=self.rnn_type,
                unroll=self.word_unroll,
                rnn_wrapper=self.word_rnn_wrapper,
            )
        else:
            raise NotImplementedError('Classifier {} not recognized!'.format(self.classifier_type))

        self.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )

    def create_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.

        :param id_to_word: mapping from a number (id) to a word in text
        :param id_to_tag: mapping from a number (id) to a tag of word
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag

        if self.id_to_word:
            self.n_words = len(self.id_to_word)
        if self.id_to_char:
            self.n_chars = len(self.id_to_char)

        self.tag_to_id = {v: k for (k, v) in self.id_to_tag.items()}
        self.all_tags = np.array(sorted(
            self.tag_to_id.keys(),
            key=lambda x: self.tag_to_id[x]
        ))

        self.all_tag_ids = np.array(
            [self.tag_to_id[t] for t in self.all_tags],
            dtype=np.int32
        )
        self.n_tags = len(self.all_tags)
        self.tag_tag_to_idx = {
            t: i for (i, t) in enumerate(self.all_tags)
        }
        self.tag_id_to_idx = {
            id: i for (i, id) in enumerate(self.all_tag_ids)
        }
        self.tag_id_to_idx_lookup = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.all_tag_ids,
                values=tf.range(self.n_tags, dtype=tf.int32),
            ),
            default_value=tf.constant(0, dtype=tf.int32),
            name="tag_id_to_idx_lookup",
        )
        # attention: string in tensor is bytes
        # need to use x.decode('utf-8') after lookup
        self.tag_id_to_tag_lookup = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.all_tag_ids,
                values=self.all_tags,
            ),
            default_value=tf.constant('<MASK>'),
            name="tag_id_to_tag_lookup",
        )

    def set_bert_trainable_layers(self, bert_main_layer, bert_first_trainable_layer=0):
        # get name of encoder layers
        pattern_bert_layer = regex.compile('.*(layer_._[0-9]+).*')
        bert_encoder_layer_ids = []
        for w in bert_main_layer.encoder.weights:
            tmp_m = pattern_bert_layer.match(w.name)
            if tmp_m and tmp_m.group(1) not in bert_encoder_layer_ids:
                bert_encoder_layer_ids.append(tmp_m.group(1))

       # set bert layers trainable and not trainable
        bert_trainable_layers = bert_encoder_layer_ids[bert_first_trainable_layer:]
        pattern_bert_trainable_layer = regex.compile(
            '.*encoder.+({}).*'.format('|'.join(bert_trainable_layers))
        )
        for w in bert_main_layer.weights:
            if (
                len(bert_trainable_layers) > 0
                and pattern_bert_trainable_layer.match(w.name)
            ):
                w._trainable = True
            else:
                w._trainable = False
            # only for test
            if bert_first_trainable_layer == 100:
                w._trainable = True

    def create_constants(self):
        self.small = -1000.0

        if self.crf and self.crf_begin_end:
            self.b_s = np.ones((1, self.n_tags + 2,), dtype=np.float32) * self.small
            self.b_s[:, -2] = 0.0
            self.e_s = np.ones((1, self.n_tags + 2,), dtype=np.float32) * self.small
            self.e_s[:, -1] = 0.0

            self.b_idx = np.array([self.n_tags], dtype=np.int32)
            self.e_idx = np.array([self.n_tags+1], dtype=np.int32)

    def build(self, input_shape):
        """

        :param input_shape: (None, num_eles)
        :return:
        """
        if self.crf_begin_end:
            self.transitions = self.add_weight(
                shape=(self.n_tags+2, self.n_tags+2),
                initializer='glorot_uniform',
                trainable=True,
                name='transitions'
            )
        else:
            self.transitions = self.add_weight(
                shape=(self.n_tags, self.n_tags),
                initializer='glorot_uniform',
                trainable=True,
                name='transitions'
            )

    def call(self, inputs, training=None):
        # preprocess inputs
        processed_inputs = self.prepare_final_inputs(
            inputs, training=training
        )
        _x = processed_inputs['final_inputs']
        word_mask_bool = processed_inputs['word_mask_bool']
        tags_idxes = processed_inputs['tags_idxes']
        sents_len = processed_inputs['sents_len']
        sents_num_tokens = processed_inputs['sents_num_tokens']
        word_mask_float = tf.cast(word_mask_bool, tf.float32)
        # layers
        # tags_scores: (None, seq_len, n_tags)
        tags_scores = self.get_tags_scores(
            final_inputs=_x,
            word_mask_bool=word_mask_bool,
            word_mask_float=word_mask_float,
        )

        # calculate loss
        if self.crf:
            # TODO: it is better to use scatter_nd to rearrange the scores
            #  to drop out masked elements for bert, which makes crf easier
            if self.crf_begin_end:
                observations = self.add_begin_end_score(
                    tags_scores,
                    sents_len,
                    word_mask_float,
                )
                padded_tags_idxes = self.add_begin_end_idx(
                    tags_idxes,
                    sents_len
                )
                padded_sents_len = sents_len + 2
            else:
                observations = tags_scores
                padded_tags_idxes = tags_idxes
                padded_sents_len = sents_len

            # crf loss
            log_likelihood, transition_params = tfa.text.crf_log_likelihood(
                inputs=observations,
                tag_indices=padded_tags_idxes,
                sequence_lengths=padded_sents_len,
                transition_params=self.transitions,
            )
            loss = -log_likelihood
        else:
            # there is already "-" in keras.losses classes
            loss = keras.losses.sparse_categorical_crossentropy(
                y_true=tags_idxes,
                y_pred=tags_scores,
                from_logits=True,
            )
            loss = loss * word_mask_float
            loss = tf.reduce_sum(loss, axis=-1)

        # # TODO: if mean is better, do we need to weight by sent len?
        if self.loss_per_token:
            loss = loss/tf.cast(sents_num_tokens, tf.float32)

        return loss


    def prepare_final_inputs(self, inputs, training=None):
        # # save parameters
        # if training:
        #     saved_locals = locals()
        #     self.parameters = saved_locals
        #     self.parameters.update(self.parameters['kwargs'])
        #     del self.parameters['kwargs']
        #     del self.parameters['self']
        #     # not save pre_emb because it might be very large
        #     # and duplicates the original embedding file
        #     del self.parameters['pre_emb']
        #     del self.parameters['training']
        #     with open(self.parameters_path, 'wb') as f:
        #         pickle.dump(self.parameters, f)

        # word_ids: (None, max_seq_len, )
        word_ids = inputs['words']
        # word_mask_bool: (None, max_seq_len, )
        word_mask_bool = tf.cast(inputs['score_mask'], tf.bool)

        # Final input (all word features)
        _x = []

        # bert_path
        if self.bert_path:
            # note: return_dict=True will be applied by default
            # it is commeted out because TF says "The parameter `return_dict`
            # cannot be set in graph mode and will always be set to `True`."
            bert_output = self.bert_layer(
                word_ids,
                attention_mask=inputs['bert_attention_mask'],
                token_type_ids=inputs['bert_token_type_ids'],
                # return_dict=True
            )
            _x.append(bert_output.last_hidden_state)

        # word dim
        if self.word_dim:
            # attention! word_ids start from 1, and mask_zero==True
            # thus, they are consistent
            # otherwise, a lookup table is needed to convert id to idx in embeddings
            word_input = self.word_layer(word_ids)
            _x.append(word_input)

        # char dim
        if self.char_dim:
            char_ids = inputs['chars']
            char_input = self.char_layer(char_ids)
            _x.append(char_input)

        if self.ele_num:
            ele_num = tf.expand_dims(inputs['ele_num'], axis=2)
            _x.append(ele_num)

        if self.only_CHO:
            only_CHO = tf.expand_dims(
                tf.cast(inputs['only_CHO'], tf.float32),
                axis=2
            )
            _x.append(only_CHO)

        if self.tar_tag:
            # only for coding test
            tar_tag = tf.expand_dims(
                tf.cast(inputs['tar_tag'], tf.float32),
                axis=2
            )
            _x.append(tar_tag)

        if self.pre_tag:
            # only for coding test
            pre_tag = tf.expand_dims(
                tf.cast(inputs['pre_tag'], tf.float32),
                axis=2
            )
            _x.append(pre_tag)

        # Prepare final input
        if len(_x) == 1:
            final_inputs = _x[0]
        else:
            final_inputs = tf.concat(_x, axis=-1)

        if self.dropout:
            final_inputs = self.dropout_layer(final_inputs, training=training)

        # get final_y, which is tags_idxes used to cal cost
        tag_ids = inputs['tags']
        tags_idxes = self.tag_id_to_idx_lookup.lookup(tag_ids)

        # get sentence length
        sents_len = tf.reshape(inputs['length'], (-1,))

        # get number of tokens in batched sentences
        sents_num_tokens = tf.reshape(inputs['num_tokens'], (-1,))

        processed_inputs = {
            'final_inputs': final_inputs,
            'word_mask_bool': word_mask_bool,
            'tags_idxes': tags_idxes,
            'sents_len': sents_len,
            'sents_num_tokens': sents_num_tokens,
        }
        return processed_inputs

    def get_tags_scores(self, final_inputs, word_mask_bool, word_mask_float):
        """
        return tags_scores
        if crf, scores are logits
        if not crf, scores are normalized with softmax

        :param final_inputs:
        :param word_mask_bool:
        :param word_mask_float:
        :return: (None, max_seq_len, n_tags)
        """
        # layers to get scores before crf
        # the score here is not softmax normalized
        tags_scores = self.unary_score_layer(final_inputs, mask=word_mask_bool)
        tags_scores = tags_scores * tf.expand_dims(word_mask_float, axis=2)
        # If not use crf, normalized tags_scores with softmax to get probability
        # at this time, we should set logits=False in sparse_categorical_crossentropy
        # This is same as not use softmax and set logits=True in
        # sparse_categorical_crossentropy. The loss is not changed
        # ref: https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331ba
        # f93fe77205550f2c2e6c90ee5/tensorflow/python/keras/backend.py#L4508
        return tags_scores

    def pad_with_begin_end(self, seq, sentence_len, begin, end):
        """

        :param seq: (max_seq_len, ) or (max_seq_len, dim)
        :param sentence_len: int, real seq len
        :return:
        """
        padded_seq = tf.concat(
            [
                begin,
                seq[: sentence_len],
                end,
                seq[sentence_len:]
            ],
            axis=0
        )
        return padded_seq

    def add_begin_end_score(self,
                            tags_scores,
                            sents_len,
                            word_mask_float):
        """
        add begin/end tag as the last 2 dimensions in tags_scores

        :param tags_scores: (batch_size, max_seq_len, n_tags)
                    or (batch_size=1, real_seq_len, n_tags)
        :param sents_len: (batch_size, )
                    or (batch_size=1, )
                    consistent with tags_score
        :param word_mask_float: (batch_size, max_seq_len)
                    or (batch_size=1, real_seq_len)
                    consistent with tags_score
        :return: (batch_size, max_seq_len+2, n_tags+2)
        """

        observations = tf.concat(
            [
                tags_scores,
                tf.tile(
                    tf.expand_dims(word_mask_float, 2),
                    multiples=[1, 1, 2]
                )*self.small,
            ],
            axis=-1,
        )

        observations, _ = tf.map_fn(
            lambda x: (
                self.pad_with_begin_end(x[0], x[1], self.b_s, self.e_s),
                x[1]
            ),
            (observations, sents_len)
        )

        return observations

    def add_begin_end_idx(self, tags_idxes, sents_len):
        padded_tags_idxes, _ = tf.map_fn(
            lambda x: (
                self.pad_with_begin_end(x[0], x[1], self.b_idx, self.e_idx),
                x[1]
            ),
            (tags_idxes, sents_len)
        )
        return padded_tags_idxes

    def predict(self, x_batches, return_tensor=False, **kwargs):
        """
        return tags_scores, each sentence would be in real length
        rather than the padded ones
        Scores are logits rather than probabilities

        :param x_batches: all data in multiple batches
        :param kwargs:
        :return:
        """
        # here only scores without crf considered
        # use  predict_idx to get the ids
        processed_inputs = x_batches.map(self.prepare_final_inputs)
        scores = processed_inputs.map(
            lambda x: self.get_tags_scores(
                final_inputs=x['final_inputs'],
                word_mask_bool=x['word_mask_bool'],
                word_mask_float=tf.cast(x['word_mask_bool'], tf.float32),
            )
        ).unbatch()
        word_mask = processed_inputs.map(
            lambda x: x['word_mask_bool']
        ).unbatch()
        scores_mask = tf.data.Dataset.zip((scores, word_mask))
        # restore the length from max_seq_len to real_seq_len
        scores = scores_mask.map(lambda s, mask: s[mask])
        # return tensor for further calculation with tf ops
        # if return tensor and crf==True, we return observations here
        # rather than scores for convenience in furthere calculation
        if not return_tensor:
            scores = [s.numpy() for s in scores]
        return scores


    def predict_idx(self, x_batches, **kwargs):
        # here scores_sents is unbatched data
        scores_sents = self.predict(x_batches, return_tensor=True, **kwargs)

        if self.crf:
            if self.crf_begin_end:
                observations = scores_sents.batch(1)
                observations = observations.map(
                    lambda x: self.add_begin_end_score(
                        tags_scores=x,
                        sents_len=tf.shape(x)[1:2],
                        word_mask_float=tf.ones((1, tf.shape(x)[1])),
                    )
                ).unbatch()
            else:
                observations = scores_sents
            observations = [s.numpy() for s in observations]
            tag_idx_sents = list(map(
                lambda x: (tfa.text.crf.viterbi_decode(
                    x,
                    self.transitions
                ))[0],
                observations
            ))
            if self.crf_begin_end:
                tag_idx_sents = list(map(
                    lambda x: x[1:-1],
                    tag_idx_sents
                ))
        else:
            tag_idx_sents = scores_sents.map(
                lambda x: tf.argmax(x, axis=-1, output_type=tf.int32)
            )
            tag_idx_sents = [s.numpy() for s in tag_idx_sents]
        return tag_idx_sents

    # args and kwargs are the same as original predict function
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    def predict_id(self, x_batches, **kwargs):
        """

        :param x_batches: all data in multiple batches
        :param kwargs:
        :return:
        """

        tag_idx_sents = self.predict_idx(x_batches, **kwargs)
        tag_ids_sents = list(map(
            lambda x: self.all_tag_ids[x],
            tag_idx_sents
        ))
        return tag_ids_sents


    def predict_label(self, x_batches, **kwargs):
        """

        :param x_batches: all data in multiple batches
        :param kwargs:
        :return:
        """
        tag_idx_sents = self.predict_idx(x_batches, **kwargs)
        label_predictions = list(map(
            lambda x: self.all_tags[x],
            tag_idx_sents
        ))
        if self.clean_tag:
            label_predictions = list(map(self.remove_broken_entities, label_predictions))
        return label_predictions

    def remove_broken_entities(self, tag_seq):
        clean_seq = []
        for tag in tag_seq:
            if (
                tag.startswith('B-')
                or tag.startswith('S-')
                or tag.startswith('O')
            ):
                clean_seq.append(tag)
            elif (
                len(clean_seq) > 0
                and clean_seq[-1][2:] == tag[2:]
                and (not clean_seq[-1].startswith('E-'))
            ):
                clean_seq.append(tag)
            else:
                clean_seq.append('O')
        return clean_seq

    def evaluate_id(self, x_batches, y_true, raw_sentences, **kwargs):
        """

        :param x_batches: all data in multiple batches
        :param y_true:
        :param raw_sentences:
        :param kwargs:
        :return: averaged F1 score
        """

        labels_true = y_true.map(
            lambda x: self.tag_id_to_tag_lookup.lookup(x)
        )

        return self.evaluate_label(x_batches, labels_true, raw_sentences, **kwargs)


    def evaluate_label(self, x_batches, labels_true, raw_sentences, **kwargs):
        tags_pred = self.predict_label(x_batches)

        # labels_true was padded, here we recover it
        labels_true = labels_true.unbatch()
        word_mask = x_batches.map(
            lambda x: tf.cast(x['score_mask'], tf.bool)
        ).unbatch()
        labels_mask = tf.data.Dataset.zip((labels_true, word_mask))
        # restore the length from max_seq_len to real_seq_len
        labels_true = labels_mask.map(lambda l, mask: l[mask])
        labels_true = [l.numpy().astype('U') for l in labels_true]

        predictions = []
        count = np.zeros((self.n_tags, self.n_tags), dtype=np.int32)

        for sent_index, raw_sentence in enumerate(raw_sentences):
            r_tags = labels_true[sent_index]
            r_idx = [self.tag_tag_to_idx[t] for t in r_tags]

            p_tags = tags_pred[sent_index]
            p_idx = [self.tag_tag_to_idx[t] for t in p_tags]

            assert len(p_tags) == len(r_tags)
            assert len(r_tags) == len(raw_sentence)

            if self.tag_scheme == 'iobes':
                p_tags = iobes_iob(p_tags)
                r_tags = iobes_iob(r_tags)
            for i, (i_pred, i_real) in enumerate(zip(p_idx, r_idx)):
                new_line = " ".join(
                    [raw_sentence[i]['text'],  r_tags[i], p_tags[i]]
                )
                predictions.append(new_line)
                count[i_real, i_pred] += 1
            predictions.append("")

        eval_lines = evaluate_lines(predictions)
        eval_lines = eval_lines.rstrip()
        eval_lines = eval_lines.split('\n')
        eval_lines = [l.rstrip() for l in eval_lines]
        for line in eval_lines:
            print(line)

        # Confusion matrix with accuracy for each tag
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * self.n_tags)).format(
            "ID", "NE", "Total",
            *([tag for tag in self.all_tags] + ["Percent"])
        ))
        for i in range(self.n_tags):
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * self.n_tags)).format(
                str(i), self.all_tags[i], str(count[i].sum()),
                *([count[i][j] for j in range(self.n_tags)] +
                  ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
            ))

        # Global accuracy
        print("%i/%i (%.5f%%)" % (
            count.trace(), count.sum(),
            100. * count.trace() / max(1, count.sum())
        ))

        return float(eval_lines[1].strip().split()[-1])


    def save_model(self, save_weights=False):
        """
        Write components values to disk.
        """

        signature = inspect.signature(self.__class__.__init__)
        model_config_essential = dict(filter(
            lambda x: x[0] in signature.parameters,
            self.__dict__.items()
        ))
        with open(
            os.path.join(self.model_path, 'model_config.pkl'),
            'wb'
        ) as fw:
            pickle.dump(model_config_essential, fw)
        if save_weights:
            self.save_weights(self.opt_cp_path)

    @classmethod
    def create_scratch_model(cls, model_path, bert_path=None, to_reload_model=True):
        """
        Create model from config file w/o initialization

        :param model_path:
        :return:
        """
        config_path = os.path.join(model_path, 'model_config.pkl')
        with open(config_path, 'rb') as fr:
            model_config = pickle.load(fr)
        model_config['model_path'] = model_path
        if bert_path:
            model_config['bert_path'] = bert_path
        model_config['to_reload_model'] = to_reload_model
        model = cls(**model_config)
        batch_size = model_config['batch_size']
        # run on one sample to build model
        data_dicts = [
            {
                'words': [1, ],
                'chars': [[1,], ],
                'ele_num': [0.0, ],
                'only_CHO': [0, ],
                'tar_tag': [0, ],
                'pre_tag': [0, ],
                'tags': [1, ],
                'length': 1,
                'num_tokens': 1,
                'bert_token_type_ids': [0, ],
                'bert_attention_mask': [1, ],
                'score_mask': [1, ],
            }
        ] * batch_size
        data_type, data_shape, padded_data_shape = get_input_format(model_type='MER')
        data_X, data_Y = dict_to_tf_dataset(
            data_dicts,
            data_type,
            data_shape,
            padded_shape=padded_data_shape,
            column_y=None,
            batch_size=batch_size,
        )
        # the weight has been changed even only fit once
        # need to reload weight to use it
        # TODO: is there anyway to compile the model without calling fit?
        model.fit(
            x=tf.data.Dataset.zip((data_X, data_Y)),
            epochs=1,
            verbose=0,
        )
        return model

    @classmethod
    def reload_model(cls, model_path, cp_path=None, bert_path=None):
        """
        Load components values from disk.
        """
        # TODO: should be able to create scratch bert model with vocab
        #  rather than whole checkpoint file
        model = NERModel.create_scratch_model(
            model_path=model_path,
            bert_path=bert_path,
            to_reload_model=True,
        )
        if cp_path is None:
            cp_path = os.path.join(model_path, 'opt_cp', 'cp.ckpt')
        model.load_weights(cp_path)
        return model


