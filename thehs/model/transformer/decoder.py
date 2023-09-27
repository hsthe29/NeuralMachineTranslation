from tensorflow import keras
from attention import SelfAttention, CrossAttention
from feedforward import FeedForward
from embedding import PositionalEmbedding


class DecoderStage(keras.layers.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderStage, self).__init__()

        self.self_attention = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context, training, look_ahead_mask, mask):
        x = self.self_attention(x, look_ahead_mask, training=training)
        x = self.cross_attention(x, context, mask, training=training)

        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x, training=training)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dec_stages = [
            DecoderStage(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, training, look_ahead_mask, mask):
        x = self.pos_embedding(x)

        x = self.dropout(x, training=training)

        for stage in self.dec_stages:
            x = stage(x, context, training, look_ahead_mask, mask)

        self.last_attn_scores = self.dec_stages[-1].last_attn_scores

        return x
