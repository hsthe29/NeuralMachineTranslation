import tensorflow as tf
from tensorflow import keras


class Attention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        # query: batch, t, units
        # value: batch, s, units

        # `W1@ht`.
        w1_query = self.W1(query)

        # `W2@hs`.
        w2_value = self.W2(value)
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        return context_vector, attention_weights


class CrossAtention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
