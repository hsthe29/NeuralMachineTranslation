import tensorflow as tf
import numpy as np
from src.utils import preprocess_en, preprocess_vi, add_tokens


class Language:
    def __init__(self, vocabulary, special_tokens, is_english=True):
        mask_token, oov_token, start_token, end_token = special_tokens
        vocab = [start_token, end_token] + vocabulary
        self.is_english = is_english
        if is_english:
            self.clean = preprocess_en
        else:
            self.clean = preprocess_vi

        self.__word_to_index = tf.keras.layers.TextVectorization(
            standardize=add_tokens,
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

    def tokenize(self, text):
        return tf.strings.split(' '.join(['[sos]', self.clean(text), '[eos]']))

    def convert_to_text(self, tokens, attn_maps, src_words):
        oovs = tf.where(tokens == 1)
        word_tokens = self.__index_to_word(tokens)
        if len(oovs) > 0:
            a = tf.Variable(np.zeros(tokens.shape, dtype=bool))
            b = tf.Variable(np.zeros(tokens.shape, dtype=str))
            for (x) in oovs:
                index = tf.squeeze(x)
                a[index].assign(True)
                bind_id = tf.math.argmax(attn_maps[index])
                b[index].assign(src_words[bind_id])
            word_tokens = tf.where(a, b, word_tokens)
        result_text = tf.strings.reduce_join(word_tokens, separator=' ')
        result_text = tf.strings.strip(result_text)
        return result_text

    @property
    def vocab_size(self):
        return self.__word_to_index.vocabulary_size()

    @property
    def vocab(self):
        return self.__word_to_index.get_vocabulary()

    @property
    def lang(self):
        return "english" if self.is_english else "vietnamese"
