import tensorflow as tf
from tensorflow import keras


class Attention(keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.Wq = keras.layers.Dense(units)
        self.Wk = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
        self.norm = keras.layers.LayerNormalization()

    def _compute_scores(self, q, v, v_mask):
        query_attn = self.Wq(q)
        key_attn = self.Wk(v)

        q_reshaped = tf.expand_dims(query_attn, axis=-2)
        k_reshaped = tf.expand_dims(key_attn, axis=-3)

        scores = self.V(tf.nn.tanh(q_reshaped + k_reshaped))
        scores = tf.squeeze(scores, axis=-1)
        v_mask = tf.expand_dims(v_mask, axis=-2)
        scores -= 1.0e9*tf.cast(tf.logical_not(v_mask), dtype=scores.dtype)
        weights = tf.nn.softmax(scores)

        result = tf.matmul(weights, v)
        q_mask = tf.ones(tf.shape(q)[:-1], dtype=bool)
        q_mask = tf.expand_dims(q_mask, axis=-1)
        result *= tf.cast(q_mask, dtype=result.dtype)
        return result, weights

    def call(self, query, value, v_mask):
        attn_output, attn_scores = self._compute_scores(query, value, v_mask)

        result = keras.layers.add([query, attn_output])
        result = self.norm(result)

        return result, attn_scores
