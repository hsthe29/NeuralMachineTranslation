import tensorflow as tf
from tensorflow import keras
from src.model.Attention import Attention


class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, units):
        super(Decoder, self).__init__()
        # self.target_processor = target_processor
        self.attention_weights = None
        self.vocab_size = vocab_size
        # self.word_to_id = tf.keras.layers.StringLookup(
        #     vocabulary=target_processor.get_vocabulary(),
        #     mask_token='', oov_token='[UNK]')
        # self.id_to_word = tf.keras.layers.StringLookup(
        #     vocabulary=target_processor.get_vocabulary(),
        #     mask_token='', oov_token='[UNK]',
        #     invert=True)
        # self.start_token = self.word_to_id('[START]')
        # self.end_token = self.word_to_id('[END]')

        self.embedding_size = embedding_size
        self.units = units

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                embedding_size, mask_zero=True)

        # 2. The RNN keeps track of what's been generated so far.
        self.gru = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        # 3. The RNN output will be the query for the attention layer.
        self.attention = Attention(units)

        self.Wc = keras.layers.Dense(units, activation='tanh', use_bias=False)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = keras.layers.Dense(self.vocab_size)

    def call(self,
             x, value,
             state,
             input_mask):
        # 1. Lookup the embeddings
        x = self.embedding(x)

        # 2. Process the target sequence.
        query, state = self.gru(x, initial_state=state)
        # 3. Use the GRU output as the query for the attention over the context.
        context_vector, attention_weights = self.attention(
            query=query, value=value, mask=input_mask)
        self.attention_weights = attention_weights
        ct_ht = tf.concat([context_vector, query], axis=-1)
        attention_vector = self.Wc(ct_ht)
        # attention_vector: ('batch', 't', 'dec_units'))

        # Step 5. Generate logit predictions:
        logits = self.output_layer(attention_vector)
        # logits: ('batch', 't', 'output_vocab_size'))

        return logits, state
