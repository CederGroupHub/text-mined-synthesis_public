import tensorflow as tf
from tensorflow import keras
import numpy as np

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

class PlainLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        loss = y_pred
        return loss


