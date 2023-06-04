import tensorflow as tf

from src.utils import text_normalize_en, text_normalize_vi


class Language:
    def __init__(self, vocabulary, special_tokens, lang):
        mask_token, oov_token, start_token, end_token = special_tokens
        vocab = [start_token, end_token] + vocabulary

        if lang == 'en':
            self.preprocess = text_normalize_en
        elif lang == 'vi':
            self.preprocess = text_normalize_vi
        else:
            raise ValueError("Language is not currently supported!")

        self.__word_to_index = tf.keras.layers.TextVectorization(
            standardize=self.preprocess,
            vocabulary=vocab,
            ragged=True)

        self.__index_to_word = tf.keras.layers.StringLookup(
            vocabulary=vocab,
            mask_token=mask_token,
            invert=True)

    def convert_to_tensor(self, text):
        text = tf.convert_to_tensor(text)
        if len(text.shape) == 0:
            text = tf.convert_to_tensor(text)[tf.newaxis]

        return self.__word_to_index(text).to_tensor()

    def convert_to_text(self, tokens):
        result_text_tokens = self.__index_to_word(tokens)

        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')

        result_text = tf.strings.strip(result_text)
        return result_text

    @property
    def vocab_size(self):
        return self.__word_to_index.vocabulary_size()

    @property
    def vocab(self):
        return self.__word_to_index.get_vocabulary()
