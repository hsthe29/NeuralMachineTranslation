import tensorflow as tf
from tensorflow import keras
from ._feedforward import FeedForward


class EncoderStage(keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super(EncoderStage, self).__init__()

        self.rnn = keras.layers.Bidirectional(
            layer=keras.layers.LSTM(hidden_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform'),
            merge_mode='concat')

        self.feedforward = FeedForward(size=hidden_units*2, dropout_rate=dropout_rate)

    def call(self, inputs, state):
        enc_seq, f_h, f_c, b_h, b_c = self.rnn(inputs, initial_state=state)
        enc_seq = self.feedforward(enc_seq)
        return enc_seq, (f_h, f_c, b_h, b_c)


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, vocab_size, hidden_units=128, embedding_size=256, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.hidden_units = hidden_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
        self.stages = []
        for _ in range(num_layers):
            self.stages.append(EncoderStage(hidden_units, dropout_rate))

        self.merge_hidden_state = keras.layers.Dense(hidden_units)
        self.merge_cell_state = keras.layers.Dense(hidden_units)

    def call(self, input_ids, state=None):
        x = self.embedding(input_ids)
        for stage in self.stages:
            x, state = stage(x, state=state)
        fh, fc, bh, bc = state
        fhs = keras.layers.concatenate([fh, bh])
        fcs = keras.layers.concatenate([fc, bc])
        hs = self.merge_hidden_state(fhs)
        cs = self.merge_cell_state(fcs)
        return x, (hs, cs)

    def zero_state(self, batch_size):
        return [
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units])
        ]

    def random_state(self, batch_size):
        return [
            tf.random.normal((batch_size, self.hidden_units)),
            tf.random.normal([batch_size, self.hidden_units]),
            tf.random.normal([batch_size, self.hidden_units]),
            tf.random.normal([batch_size, self.hidden_units])
        ]
