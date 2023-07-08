import tensorflow as tf
from tensorflow import keras


class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.embedding = keras.layers.Embedding(self.vocab_size, embedding_size, mask_zero=True)
        self.bi_lstm_1 = keras.layers.Bidirectional(
            layer=keras.layers.LSTM(hidden_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform'),
            merge_mode='concat')
        self.dense_1 = keras.layers.Dense(hidden_units)
        self.bi_lstm_2 = keras.layers.Bidirectional(
            layer=keras.layers.LSTM(hidden_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform'),
            merge_mode='concat')
        self.dense_2 = keras.layers.Dense(hidden_units)
        self.hidden_dense = keras.layers.Dense(hidden_units)
        self.cell_dense = keras.layers.Dense(hidden_units)

    def call(self, inputs, state):
        x = self.embedding(inputs)
        enc_seq, f_h, f_c, b_h, b_c = self.bi_lstm_1(x, initial_state=state)
        enc_context = self.dense_1(enc_seq)
        enc_seq, f_h, f_c, b_h, b_c = self.bi_lstm_2(enc_context, initial_state=[f_h, f_c, b_h, b_c])
        enc_context = self.dense_2(enc_seq)
        hidden_state = keras.layers.concatenate([f_h, b_h])
        hidden_state = self.hidden_dense(hidden_state)
        cell_state = keras.layers.concatenate([f_c, b_c])
        cell_state = self.cell_dense(cell_state)
        return enc_context, [hidden_state, cell_state]

    def init_state(self, batch_size):
        return [
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units])
        ]
