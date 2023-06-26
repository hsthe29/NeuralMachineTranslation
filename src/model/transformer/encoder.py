from tensorflow import keras


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
