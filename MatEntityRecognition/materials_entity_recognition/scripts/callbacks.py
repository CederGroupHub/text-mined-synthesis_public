import datetime

import tensorflow as tf
from tensorflow import keras
import numpy as np


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

# TODO: add early stop call back

class GetScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 data_x,
                 data_y,
                 data_raw_sentences,
                 save_better_model=True,
                 best_val_score=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_x = data_x
        self.data_y = data_y
        self.data_raw_sentences = data_raw_sentences
        self.save_better_model = save_better_model
        self.best_val_score = best_val_score

    def on_epoch_end(self, epoch, logs=None):
        print('the {}th epoch ends'.format(epoch))
        print('logs', logs, logs.get('loss'))

        time_1 = datetime.datetime.now()
        val_score = self.model.evaluate_id(
            x_batches=self.data_x,
            y_true=self.data_y,
            raw_sentences=self.data_raw_sentences,
        )
        print('score', val_score)

        if (val_score > self.best_val_score and self.save_better_model):
            # find a better model
            # save model
            print('better model found in epoch {}!'.format(epoch))
            print('last score: ', self.best_val_score)
            print('new score: ', val_score)
            self.best_val_score = val_score
            self.model.save_weights(self.model.opt_cp_path)

        time_2 = datetime.datetime.now()
        print('time cost in validation: ', time_2 - time_1)
        print()

