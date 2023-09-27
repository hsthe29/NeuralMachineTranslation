import tensorflow as tf
from tensorflow import keras


class TheAttention(keras.layers.Layer):
    def __init__(self, units):
        super(TheAttention, self).__init__()
        self.Wq = keras.layers.Dense(units)
        self.Wk = keras.layers.Dense(units)
        self.layer_norm = keras.layers.LayerNormalization()
        self.scale_factor = tf.math.sqrt(tf.cast(units, dtype=tf.float32))

    def call(self, query, key, q_mask, v_mask):
        """
        :param query: tf.Tensor, shape=[N, Tt, Ht]
        :param key: tf.Tensor, shape=[N, Ts, Hs]
        :param q_mask: tf.Tensor, shape=[N, Tt]
        :param v_mask: tf.Tensor, shape=[N, Ts]
        :return:
            attention tensor: tf.Tensor, shape=[N, Tt, Ht]
            attention weight: tf.Tensor, shape=[N, Ts, Tt]
        """
        value = key

        # apply projection
        q = self.Wq(query)
        k = self.Wk(key)

        qk = tf.matmul(q, k, transpose_b=True)  # [N, Tt, Ts]
        qk = qk / self.scale_factor

        q_mask = tf.expand_dims(q_mask, axis=-1)
        v_mask = tf.expand_dims(v_mask, axis=-1)

        mask = 1 - tf.matmul(q_mask, v_mask, transpose_b=True)  # [N, Tt, Ts]
        logits = qk - mask * 1e9

        weights = tf.nn.softmax(logits, axis=-1)

        output = tf.matmul(weights, value)  # [N, Tt, Hs]
        output = self.layer_norm(output)  # [N, Tt, Hs]
        return output, weights
