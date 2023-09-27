from abc import abstractmethod
import tensorflow as tf
from tensorflow import keras


class BaseMT(keras.Model):

    @abstractmethod
    def reset_array(self):
        pass

    @abstractmethod
    def encoder_output(self, input_ids: tf.Tensor):
        pass

    @abstractmethod
    def next_ids(self, target_in_ids: tf.Tensor, enc_output: tf.Tensor, v_mask=None):
        pass

    @abstractmethod
    def update_target_in_ids(self, target_out_ids: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def result(self):
        pass
