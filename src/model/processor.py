import tensorflow as tf

from src.utils import to_lower_normalize


class LanguageProcessor:
    def __init__(self, vocabulary):
        self.processor = tf.keras.layers.TextVectorization(
            standardize=to_lower_normalize,
            vocabulary=vocabulary,
            ragged=True)

    def convert_to_tensor(self, text):
        return self.processor(text).to_tensor()

    @property
    def vocab_size(self):
        return self.processor.vocabulary_size()

    @property
    def vocab(self):
        return self.processor.get_vocabulary()
