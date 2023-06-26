import tensorflow as tf
from tensorflow import keras


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

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.make_masks(inp, tar)

        context = self.encoder(inp, training, enc_padding_mask)  # (batch_size, context_len, d_model)

        x = self.decoder(tar, context, training, look_ahead_mask, dec_padding_mask)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits

    def make_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
        enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
        dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = tf.cast(tf.math.equal(tar, 0), tf.float32)
        dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask
