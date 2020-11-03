import numpy as np
import tensorflow as tf
from tensorflow import keras


from .utils import iob_iobes, iobes_iob
# from .nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
# from .optimization import Optimization
from .conlleval_perl import evaluate_lines

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

all_RNNs = {
    'rnn': keras.layers.RNN,
    'lstm': keras.layers.LSTM,
    'gru': keras.layers.GRU,
}

all_RNN_cells = {
    'rnn': keras.layers.SimpleRNNCell,
    'lstm': keras.layers.LSTMCell,
    'gru': keras.layers.GRUCell,
}
bidirect_merge_mode = {
    'concat': 'concat',
    'sum': 'sum',
}

class SimpleSequenceScores(keras.layers.Layer):
    def __init__(self,
                 n_tags,
                 mask_zero=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_tags = n_tags
        self.mask_zero = mask_zero

        self.final_layer = keras.layers.Dense(
            units=self.n_tags,
            kernel_initializer='glorot_uniform',
            activation=None,
            name='final_layer',
        )

    def call(self, inputs, mask=None):
        """

        :param inputs: (None, seq_len, feature_dim)
        :return:
        """
        _x = inputs
        tags_scores = self.final_layer(_x)
        return tags_scores

    def compute_mask(self, inputs, mask=None):
        """
        masked position is False.
        True represents valid

        :param inputs: (None, seq_len, feature_dim)
        :param mask: not used
        :return mat_mask: (None, seq_len, )
        """
        if not self.mask_zero:
            return None
        mask = tf.not_equal(inputs, 0)
        mask = tf.reduce_any(mask, axis=-1)
        return mask


class RNNSequenceScores(keras.layers.Layer):
    def __init__(self,
                 word_bidirect,
                 word_lstm_dim,
                 n_tags,
                 mask_zero=True,
                 rnn_type='gru',
                 unroll=False,
                 rnn_wrapper=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.word_bidirect = word_bidirect
        self.word_lstm_dim = word_lstm_dim
        self.n_tags = n_tags
        self.mask_zero = mask_zero
        self.unroll = unroll
        self.rnn_wrapper = rnn_wrapper
        self.rnn_type = rnn_type.lower()

        if self.word_bidirect:
            # attention, when Bidirectional is used, mask must be input
            # otherwise, the padding would change cell states
            if not self.rnn_wrapper:
                self.word_bi_lstm = keras.layers.Bidirectional(
                    all_RNNs[self.rnn_type](
                        units=self.word_lstm_dim,
                        return_sequences=True,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        unroll=self.unroll,
                    ),
                    merge_mode='concat',
                    name='word_bi_lstm'
                )
            else:
                self.word_bi_lstm = keras.layers.Bidirectional(
                    keras.layers.RNN(
                        all_RNN_cells[self.rnn_type](
                            units=self.word_lstm_dim,
                            kernel_initializer='glorot_uniform',
                            recurrent_initializer='glorot_uniform',
                        ),
                        return_sequences=True,
                        unroll=self.unroll,
                    ),
                    merge_mode='concat',
                    name='word_bi_lstm'
                )
            self.tanh_layer = keras.layers.Dense(
                units=self.word_lstm_dim,
                activation='tanh',
                kernel_initializer='glorot_uniform',
                name='tanh_layer',
            )
        else:
            if not self.rnn_wrapper:
                self.word_lstm_for = all_RNNs[self.rnn_type](
                    units=self.word_lstm_dim,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='glorot_uniform',
                    unroll=self.unroll,
                    name='word_lstm_for',
                )
            else:
                # LSTM use cudarnn and gpu by default, which is faster than RNN wrapped lstmCell
                # this version can be used when cudarnn is not available
                self.word_bi_lstm = keras.layers.RNN(
                    all_RNN_cells[self.rnn_type](
                        units=self.word_lstm_dim,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                    ),
                    return_sequences=True,
                    unroll=self.unroll,
                    name='word_lstm_for'
                )

        # activation function here is always None
        self.final_layer = keras.layers.Dense(
            units=self.n_tags,
            kernel_initializer='glorot_uniform',
            activation=None,
            name='final_layer',
        )

    def call(self, inputs, mask=None):
        """

        :param inputs: (None, seq_len, feature_dim)
        :return:
        """
        _x = inputs
        if mask is None:
            mask = self.compute_mask(_x)
        if self.word_bidirect:
            _x = self.word_bi_lstm(_x, mask=mask)
            _x = self.tanh_layer(_x)
        else:
            _x = self.word_lstm_for(_x, mask=mask)
        tags_scores = self.final_layer(_x)
        return tags_scores

    def compute_mask(self, inputs, mask=None):
        """
        masked position is False.
        True represents valid

        :param inputs: (None, seq_len, feature_dim)
        :param mask: not used
        :return mat_mask: (None, seq_len, )
        """
        if not self.mask_zero:
            return None
        mask = tf.not_equal(inputs, 0)
        mask = tf.reduce_any(mask, axis=-1)
        return mask


class CharacterFeatures(keras.layers.Layer):
    def __init__(self,
                 n_chars,
                 char_dim,
                 char_lstm_dim,
                 char_bidirect,
                 batch_size,
                 mask_zero=True,
                 char_combine_method='concat',
                 rnn_type='gru',
                 unroll=False,
                 rnn_wrapper=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_chars = n_chars
        self.char_dim = char_dim
        self.char_lstm_dim = char_lstm_dim
        self.char_bidirect = char_bidirect
        self.batch_size = batch_size
        self.mask_zero = mask_zero
        self.char_combine_method = char_combine_method.lower()
        self.rnn_type = rnn_type.lower()
        self.unroll = unroll
        self.rnn_wrapper = rnn_wrapper

        self.char_emb_layer = keras.layers.Embedding(
                input_dim=self.n_chars+1,
                output_dim=self.char_dim,
                embeddings_initializer='glorot_uniform',
                mask_zero=True,
                trainable=True,
                name='char_emb_layer',
            )

        if self.char_bidirect:
            if self.batch_size == 1 and (not self.rnn_wrapper):
                self.char_bi_lstm = keras.layers.Bidirectional(
                    all_RNNs[self.rnn_type](
                        units=self.char_lstm_dim,
                        return_sequences=False,
                        # TODO: use stateful or not?
                        #     characters might be connected to each other
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                        unroll=self.unroll,
                    ),
                    merge_mode=bidirect_merge_mode.get(
                        self.char_combine_method, 'concat'
                    ),
                    name='char_bi_lstm'
                )
            else:
                # there seems to be a bug in tf lstm code for
                # nested lstm and batch_size > 1
                self.char_bi_lstm = keras.layers.Bidirectional(
                    keras.layers.RNN(
                        all_RNN_cells[self.rnn_type](
                            units=self.char_lstm_dim,
                            kernel_initializer='glorot_uniform',
                            recurrent_initializer='glorot_uniform',
                        ),
                        return_sequences=False,
                        unroll=self.unroll,
                    ),
                    merge_mode=bidirect_merge_mode.get(
                        self.char_combine_method, 'concat'
                    ),
                    name='char_bi_lstm'
                )
            if self.char_combine_method == 'tanh':
                self.char_tanh_layer = keras.layers.Dense(
                    units=self.char_lstm_dim,
                    activation='tanh',
                    kernel_initializer='glorot_uniform',
                    name='char_tanh_layer',
                )
        else:
            if self.batch_size == 1 and (not self.rnn_wrapper):
                # LSTM use cudarnn and gpu by default,
                # which is faster than RNN wrapped lstmCell
                # the results are the same
                # note that the RNN wrapped lstmCell version
                # only support tensor-type mask
                # both version support numpy-type inputs
                self.char_lstm_for = all_RNNs[self.rnn_type](
                    units=self.char_lstm_dim,
                    return_sequences=False,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='glorot_uniform',
                    name='char_lstm_for',
                    unroll=self.unroll,
                )
            else:
                self.char_lstm_for = keras.layers.RNN(
                    all_RNN_cells[self.rnn_type](
                        units=self.char_lstm_dim,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='glorot_uniform',
                    ),
                    return_sequences=False,
                    unroll=self.unroll,
                )

    def call(self, inputs, mask=None):
        """

        :param inputs: (None, seq_len, max_word_len)
        :return:
        """
        input_shape = tf.shape(inputs)
        _x = inputs

        # _x: (batch_size*seq_len, max_word_len)
        _x = tf.reshape(
            _x,
            tf.concat(
                [[-1, ], input_shape[2:]],
                axis=0
            )
        )
        # mask: (batch_size*seq_len, max_word_len)
        if mask is None:
            mask = self.compute_mask(_x)
        else:
            mask = tf.reshape(
                mask,
                tf.concat(
                    [[-1, ], input_shape[2:]],
                    axis=0
                )
            )

        # _x: (batch_size*seq_len, max_word_len, char_dim)
        _x = self.char_emb_layer(_x)

        if self.char_bidirect:
            # _x: (batch_size*seq_len, 2*char_lstm_dim)
            _x = self.char_bi_lstm(_x, mask=mask)
            if self.char_combine_method == 'tanh':
                # _x: (batch_size*seq_len, char_lstm_dim)
                _x = self.char_tanh_layer(_x)
        else:
            # _x: (batch_size*seq_len, char_lstm_dim)
            _x = self.char_lstm_for(_x, mask=mask)

        if self.char_bidirect and self.char_combine_method == 'concat':
            # _x: (batch_size, seq_len, 2*char_lstm_dim)
            _x = tf.reshape(
                _x,
                tf.concat(
                    [input_shape[:2], [2*self.char_lstm_dim, ]],
                    axis=0
                )
            )
        else:
            # _x: (batch_size, seq_len, char_lstm_dim)
            _x = tf.reshape(
                _x,
                tf.concat(
                    [input_shape[:2], [self.char_lstm_dim, ]],
                    axis=0
                )
            )

        return _x

    def compute_mask(self, inputs, mask=None):
        """
        masked position is False.
        True represents valid

        :param inputs: (None, seq_len, max_word_len) or (None, max_word_len)
        :param mask: not used
        :return mat_mask: (None, seq_len, ) or (None,)
        """
        if not self.mask_zero:
            return None
        mask = tf.not_equal(inputs, 0)
        return mask


