from abc import ABC

import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, src_language, tar_language, model):
        super(Translator, self).__init__()
        self.src_language = src_language
        self.tar_language = tar_language
        self.model = model

    def __call__(self, sentence, max_length=100):

        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.src_tokenizer(sentence).to_tensor()
        print(sentence)

        encoder_input = sentence

        start_end = self.tar_tokenizer([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        return output  # Shape: `()`.

        # tokens = tokenizers.en.lookup(output)[0]

        # # `tf.function` prevents us from using the attention_weights that were
        # # calculated on the last iteration of the loop.
        # # So, recalculate them outside the loop.
        # self.transformer([encoder_input, output[:,:-1]], training=False)
        # attention_weights = self.transformer.decoder.last_attn_scores

        # return text, tokens, attention_weights
