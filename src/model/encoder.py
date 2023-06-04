import tensorflow as tf
from tensorflow import keras


class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.embedding = keras.layers.Embedding(self.vocab_size, embedding_size, mask_zero=True)
        self.bi_gru = keras.layers.Bidirectional(
            layer=keras.layers.GRU(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform'),
            merge_mode='concat')
        self.dense = keras.layers.Dense(hidden_units)

    def call(self, x, state):
        x = self.embedding(x)
        enc_seq, fs, bs = self.bi_gru(x, state)
        enc_seq = self.dense(enc_seq)
        enc_state = keras.layers.Add()([fs, bs])
        return enc_seq, enc_state

    def init_state(self, batch_size):
        return [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
