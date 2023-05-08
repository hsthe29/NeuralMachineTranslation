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
            merge_mode='sum')

    def call(self, x):
        x = self.embedding(x)
        enc_seq, fs, bs = self.bi_gru(x)

        state = keras.layers.Add()([fs, bs])
        return enc_seq, state
