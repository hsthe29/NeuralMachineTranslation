import tensorflow as tf
from tensorflow import keras


class Attention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(key_dim=units, num_heads=3, **kwargs)
        self.norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, query, value):
        attn_output, attn_scores = self.mha(
            query=query,
            value=value,
            return_attention_scores=True)

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        x = self.add([query, attn_output])
        x = self.norm(x)

        return x, attn_scores
