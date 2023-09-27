import tensorflow as tf
from tensorflow import keras
from ._attention import TheAttention


class DecoderStage(keras.layers.Layer):
    def __init__(self, hidden_units):
        super(DecoderStage, self).__init__()

        self.rnn = keras.layers.LSTM(hidden_units,
                                     return_sequences=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform')

        self.attention = TheAttention(units=hidden_units)

    def call(self, dec_inputs, enc_seq, q_mask=None, v_mask=None, state=None):
        dec_seq, hs, cs = self.rnn(dec_inputs, initial_state=state)
        dec_context, attention_weights = self.attention(dec_seq, enc_seq, q_mask, v_mask)

        return dec_context, (hs, cs), attention_weights


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, vocab_size, hidden_units=128, embedding_size=256):
        super(Decoder, self).__init__()
        self.stages = []
        self.embedding = keras.layers.Embedding(vocab_size,
                                                embedding_size, mask_zero=True)
        for _ in range(num_layers):
            self.stages.append(DecoderStage(hidden_units))

        self.fc = keras.layers.Dense(vocab_size)

    def call(self, inputs, enc_seq, q_mask=None, v_mask=None, state=None):
        x = self.embedding(inputs)

        for stage in self.stages:
            x, _, _ = stage(x, enc_seq, q_mask, v_mask, state)

        logits = self.fc(x)

        return logits

    def next_ids(self, inputs, enc_seq, q_mask=None, v_mask=None, in_states=None):
        x = self.embedding(inputs)
        out_states = []
        attention_weights = []
        for stage, state in zip(self.stages, in_states):
            x, out_state, attention_weight = stage(x, enc_seq, q_mask, v_mask, state)
            out_states.append(out_state)
            attention_weights.append(attention_weight)
        logits = self.fc(x)

        return logits, out_states, tf.convert_to_tensor(attention_weights)
