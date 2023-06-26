import tensorflow as tf
from tensorflow import keras
from encoder import Encoder
from decoder import Decoder
from src.utils import create_look_ahead_mask


def make_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

    dec_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = tf.cast(tf.math.equal(tar, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.output_fc = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = make_masks(inp, tar)

        context = self.encoder(inp, training, enc_padding_mask)

        x = self.decoder(tar, context, training, look_ahead_mask, dec_padding_mask)

        logits = self.output(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
