from tensorflow import keras
from src.model.rnn.attention import Attention


class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, units):
        super(Decoder, self).__init__()
        self.attention_weights = None
        self.vocab_size = vocab_size

        self.embedding_size = embedding_size
        self.units = units

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                embedding_size, mask_zero=True)

        self.lstm_1 = keras.layers.LSTM(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        self.lstm_2 = keras.layers.LSTM(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        self.attention_1 = Attention(units)
        self.attention_2 = Attention(units)

        self.dense = keras.layers.Dense(self.vocab_size)

    def call(self, inputs, encoder_context, v_mask, state):
        x = self.embedding(inputs)

        query, _, _ = self.lstm_1(x, initial_state=state)
        context_vector, _ = self.attention_1(query, encoder_context, v_mask)
        query, _, _ = self.lstm_2(context_vector, initial_state=state)
        context_vector, _ = self.attention_2(query, encoder_context, v_mask)
        logits = self.dense(context_vector)

        return logits

    def next_tokens(self, x, encoder_context, v_mask, state):
        x = self.embedding(x)

        query, hidden_state, cell_state = self.lstm_1(x, initial_state=state[0])
        context_vector, attention_weights = self.attention_1(query, encoder_context, v_mask)
        return_state_1 = [hidden_state, cell_state]
        query, hidden_state, cell_state = self.lstm_2(context_vector, initial_state=state[1])
        context_vector, attention_weights = self.attention_2(query, encoder_context, v_mask)
        return_state_2 = [hidden_state, cell_state]
        self.attention_weights = attention_weights
        logits = self.dense(context_vector)

        return logits, (return_state_1, return_state_2)
