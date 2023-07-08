import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.model.rnn.decoder import Decoder
from src.model.rnn.encoder import Encoder


class NMT(keras.Model):
    def __init__(self, source_processor,
                 target_processor, embedding_size, units):
        super(NMT, self).__init__()

        self.encoder = Encoder(source_processor.vocab_size, embedding_size, units)
        self.decoder = Decoder(target_processor.vocab_size, embedding_size, units)

        self.index_from_string = tf.keras.layers.StringLookup(
            vocabulary=target_processor.vocab, mask_token='')
        token_mask_ids = self.index_from_string(['', '[sos]']).numpy()

        self.token_mask = np.zeros([self.index_from_string.vocabulary_size()], bool)
        self.token_mask[np.array(token_mask_ids)] = True
        self.token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        self.start_token = self.index_from_string(tf.constant(['[sos]']))
        self.end_token = self.index_from_string(tf.constant('[eos]'))

    def call(self, inputs):
        in_tok, tar_in = inputs
        input_mask = in_tok != 0

        batch_size = tf.shape(in_tok)[0]
        first_state = self.encoder.init_state(batch_size)
        encoder_context, state = self.encoder(in_tok, first_state)

        logits = self.decoder(tar_in, encoder_context, v_mask=input_mask, state=state)
        return logits

    def predict_tokens(self, inputs, max_len, temperature=0.0):
        input_mask = inputs != 0
        batch_size = tf.shape(inputs)[0]

        in_tokens = tf.reshape(tf.cast([self.start_token.numpy()] * batch_size.numpy(), tf.int64), shape=(-1, 1))

        done = tf.zeros([batch_size, 1], dtype=bool)

        result_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        attn_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        first_state = self.encoder.init_state(batch_size)
        enc_output, enc_state = self.encoder(inputs, first_state)
        dec_state = [enc_state, enc_state]

        for i in tf.range(max_len):
            logits, dec_state = self.decoder.next_tokens(in_tokens, enc_output, v_mask=input_mask, state=dec_state)
            attn_array = attn_array.write(i, tf.squeeze(self.decoder.attention_weights))

            # Set the logits for all masked tokens to -inf, so they are never chosen.
            logits = tf.where(self.token_mask, -np.inf, logits)

            if temperature == 0.0:
                new_tokens = tf.argmax(logits, axis=-1)
            else:
                logits = tf.squeeze(logits, axis=1)
                new_tokens = tf.random.categorical(logits / temperature,
                                                   num_samples=1)
            done = done | (new_tokens == self.end_token)
            in_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            result_array = result_array.write(i, in_tokens[0])

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        result_tokens = tf.transpose(result_array.stack())
        attention_stack = attn_array.stack()
        return result_tokens, attention_stack
