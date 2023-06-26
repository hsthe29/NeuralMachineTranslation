from tensorflow import keras
from feedforward import FeedForward
from attention import SelfAttention
from embedding import PositionalEmbedding


class EncoderStage(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderStage, self).__init__()

        self.self_attention = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, training, mask):
        x = self.self_attention(x, mask, training=training)
        x = self.ffn(x, training=training)
        return x


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_stages = [
            EncoderStage(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x, training=training)
        for stage in self.enc_stages:
            x = stage(x, training, mask)

        return x  # Shape `(batch_size, seq_len, d_model)`.
