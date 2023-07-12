import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, src_language, tar_language, model):
        super(Translator, self).__init__()
        self.src_language = src_language
        self.tar_language = tar_language
        self.model = model

    def __call__(self, sentence, max_length=100):
        sentence = self.src_language.clean(sentence)
        if not isinstance(sentence, tf.Tensor):
            sentence = tf.convert_to_tensor(sentence)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        inputs = self.src_language.convert_to_tensor(sentence)

        result_tokens, attention = self.model.predict_tokens(inputs, max_len=max_length)
        result_text = self.tar_language.convert_to_text(result_tokens)
        return result_text
