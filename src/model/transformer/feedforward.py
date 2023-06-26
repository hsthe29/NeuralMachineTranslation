from tensorflow import keras


class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(dff, activation='relu')
        self.dense2 = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.add = keras.layers.Add()
        self.norm = keras.layers.LayerNormalization()

    def call(self, x, training):
        x_d = self.dense1(x)
        x_d = self.dense2(x_d)
        x_d = self.dropout(x_d, training=training)
        x = self.add([x, x_d])
        x = self.norm(x)
        return x
