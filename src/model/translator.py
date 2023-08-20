import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, src_language, tar_language, model):
        super(Translator, self).__init__()
        self.src_language = src_language
        self.tar_language = tar_language
        self.model = model

    def __call__(self, text, max_length=100):
        cleaned_text = self.src_language.clean(text)
        if not isinstance(cleaned_text, tf.Tensor):
            cleaned_text = tf.convert_to_tensor(cleaned_text)
        if len(cleaned_text.shape) == 0:
            cleaned_text = cleaned_text[tf.newaxis]

        inputs = self.src_language.convert_to_tensor(cleaned_text)

        result_tokens, attention = self.model.predict_tokens(inputs, max_len=max_length)
        result_text = self.tar_language.convert_to_text(result_tokens[0], attention, self.src_language.tokenize(text))
        return result_text
