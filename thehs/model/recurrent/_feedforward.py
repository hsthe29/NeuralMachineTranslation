from tensorflow import keras


class FeedForward(keras.layers.Layer):
    def __init__(self, size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(size, activation='relu')
        self.dense2 = keras.layers.Dense(size)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.add = keras.layers.Add()
        self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        x_d = self.dense1(x)
        x_d = self.dense2(x_d)
        x_d = self.dropout(x_d)
        x = self.add([x, x_d])
        x = self.norm(x)
        return x
