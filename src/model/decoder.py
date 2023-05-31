from tensorflow import keras
from src.model.attention import Attention


class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, units):
        super(Decoder, self).__init__()
        self.attention_weights = None
        self.vocab_size = vocab_size

        self.embedding_size = embedding_size
        self.units = units

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                embedding_size, mask_zero=True)
        # 2. The GRU keeps track of what's been generated so far.
        self.gru = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        # 3. The RNN output will be the query for the attention layer.
        self.attention = Attention(units)
        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = keras.layers.Dense(self.vocab_size)

    def call(self, x, enc_output, state=None):
        # 1. Lookup the embeddings
        x = self.embedding(x)
        # 2. Process the target sequence.
        query, state = self.gru(x, initial_state=state)
        # 3. Use the GRU output as the query for the attention over the context.
        context_vector, attention_weights = self.attention(query=query, value=enc_output)
        self.attention_weights = attention_weights
        # Step 5. Generate logit predictions:
        logits = self.output_layer(context_vector)

        return logits, state

