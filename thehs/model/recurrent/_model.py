import tensorflow as tf
from tensorflow import keras
import numpy as np
from ._decoder import Decoder
from ._encoder import Encoder
from thehs.model.base_model import BaseMT


class RnnMT(BaseMT):
    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 embedding_size: int = 256,
                 hidden_units: int = 128,
                 dropout_rate: float = 0.1):
        super(RnnMT, self).__init__()

        self.__ids_array = None
        self.__ids_index = 0
        self.encoder = Encoder(num_layers=num_layers,
                               vocab_size=vocab_size,
                               hidden_units=hidden_units,
                               embedding_size=embedding_size,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers,
                               vocab_size=vocab_size,
                               hidden_units=hidden_units,
                               embedding_size=embedding_size)

        self.build(input_shape=[tf.TensorShape([1, 1]), tf.TensorShape([1, 11])])

    @classmethod
    def from_dict(cls, params):
        instance = cls(**params)
        return instance

    def call(self, inputs):
        # destructuring inputs
        input_ids, target_ids = inputs
        input_mask = tf.cast(input_ids != 0, dtype=tf.float32)
        target_mask = tf.cast(target_ids != 0, dtype=tf.float32)

        batch_size = tf.shape(input_ids)[0]
        first_state = self.encoder.zero_state(batch_size)
        enc_seq, states = self.encoder(input_ids, first_state)

        logits = self.decoder(target_ids, enc_seq, q_mask=target_mask, v_mask=input_mask, states=states)

        return logits

    def predict(self, input_ids, input_mask=None, max_length=256):
        batch_size = tf.shape(input_ids)[0]

        target_ids = tf.reshape(tf.cast([self.start_token] * batch_size.numpy(), tf.int64), shape=(-1, 1))

        done = tf.zeros([batch_size, 1], dtype=bool)

        result_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        attn_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        first_state = self.encoder.zero_state(batch_size)
        enc_seq, enc_state = self.encoder(input_ids, first_state)
        dec_state = [enc_state, enc_state]
        # tf.print(tf.convert_to_tensor(dec_state).shape)
        for i in tf.range(max_length):
            # tf.print(i)
            logits, dec_state, attn_weight = self.decoder.next_tokens(target_ids, enc_seq, v_mask=input_mask,
                                                                      in_states=dec_state)
            # tf.print(dec_state.shape)
            # attn_array = attn_array.write(i, tf.squeeze(self.decoder.attention_weights))

            # Set the logits for all masked tokens to -inf, so they are never chosen.
            logits = tf.where(self.token_mask, -np.inf, logits)

            new_tokens = tf.argmax(logits, axis=-1)
            done = done | (new_tokens == self.end_token)
            in_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            result_array = result_array.write(i, in_tokens[0])

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        result_tokens = tf.transpose(result_array.stack())
        # attention_stack = attn_array.stack()
        return result_tokens  # , attention_stack

    def reset_array(self):
        self.__ids_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        self.__ids_index = 0

    def encode_input(self, input_ids):
        batch_size = tf.shape(input_ids)[0]
        first_state = self.encoder.zero_state(batch_size)
        enc_seq, states = self.encoder(input_ids, first_state)
        return enc_seq, states

    def next_ids(self, target_in_ids, prev_output, v_mask=None, attention=False):
        enc_seq, states = prev_output
        q_mask = tf.cast(target_in_ids != 0, dtype=tf.float32)
        logits, dec_state, attn_weight = self.decoder.next_ids(target_in_ids,
                                                               enc_seq,
                                                               q_mask=q_mask,
                                                               v_mask=v_mask,
                                                               in_states=states)
        dec_output = (enc_seq, dec_state)
        if attention:
            return logits, dec_output, attn_weight
        else:
            return logits, dec_output

    def update_target_in_ids(self, target_out_ids):
        self.__ids_array = self.__ids_array.write(self.__ids_index, target_out_ids)
        self.__ids_index += 1
        return target_out_ids

    def result(self):
        value = tf.transpose(self.__ids_array.stack())
        return value[0]
