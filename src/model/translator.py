import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.model.decoder import Decoder
from src.model.encoder import Encoder


class Translator(keras.Model):
    def __init__(self, source_processor,
                 target_processor, config):
        super(Translator, self).__init__()

        embedding_size = config.embedding_size
        units = config.recurrent_units
        # Build the encoder and decoder
        self.encoder = Encoder(source_processor.vocab_size, embedding_size, units)
        self.decoder = Decoder(target_processor.vocab_size, embedding_size, units)

        self.source_processor = source_processor
        self.target_processor = target_processor

        self.index_from_string = tf.keras.layers.StringLookup(
            vocabulary=self.target_processor.vocab, mask_token=config.mask_token)
        token_mask_ids = self.index_from_string([config.mask_token,
                                                 config.oov_token,
                                                 config.start_token]).numpy()

        self.token_mask = np.zeros([self.index_from_string.vocabulary_size()], bool)
        self.token_mask[np.array(token_mask_ids)] = True

        self.start_token = self.index_from_string(tf.constant(config.start_token))
        self.end_token = self.index_from_string(tf.constant(config.end_token))

    def call(self, inputs):
        in_tok, tar_in = inputs
        batch_size = tf.shape(in_tok)[0]
        first_state = self.encoder.init_state(batch_size)
        enc_output, enc_state = self.encoder(in_tok, first_state)

        logits, dec_state = self.decoder(tar_in, enc_output, state=enc_state)
        return logits, dec_state

    @tf.function
    def train_step(self, inputs):
        in_tok, out_tok = inputs
        tar_in = out_tok[:, :-1]
        tar_out = out_tok[:, 1:]
        tokens = (in_tok, tar_in)
        y_true = tar_out

        with tf.GradientTape() as tape:
            logits, _ = self(tokens, training=True)
            y_pred = logits
            step_loss = self.loss(y_true, y_pred)

        variables = self.trainable_variables
        gradients = tape.gradient(step_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        eval_result = {self.loss.name: step_loss}
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        eval_result.update({m.name: m.result() for m in self.metrics})
        return eval_result

    @tf.function
    def test_step(self, inputs):
        in_tok, out_tok = inputs
        tar_in = out_tok[:, :-1]
        tar_out = out_tok[:, 1:]
        tokens = (in_tok, tar_in)
        y_true = tar_out

        logits, _ = self(tokens, training=False)
        y_pred = logits
        step_loss = self.loss(y_true, y_pred)

        eval_result = {self.loss.name: step_loss}
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        eval_result.update({m.name: m.result() for m in self.metrics})
        return eval_result

    def translate(self, input_texts, max_len, temperature=0.0, return_attention=True):
        print(input_texts)
        input_tokens = self.source_processor.convert_to_tensor(input_texts)
        print('inp', input_tokens)
        batch_size = tf.shape(input_tokens)[0]

        first_state = self.encoder.init_state(batch_size)
        enc_output, enc_state = self.encoder(input_tokens, first_state)
        dec_state = enc_state

        in_tokens = tf.reshape(tf.cast([self.start_token.numpy()] * batch_size.numpy(), tf.int64), shape=(-1, 1))

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=bool)

        for _ in tf.range(max_len):
            # Pass in two tokens from the target sequence:
            # 1. The current input to the decoder.
            # 2. The target for the decoder's next prediction.

            logits, dec_state = self.decoder(in_tokens, enc_output, state=dec_state)
            attention.append(self.decoder.attention_weights)
            token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]

            # Set the logits for all masked tokens to -inf, so they are never chosen.
            logits = tf.where(token_mask, -np.inf, logits)

            if temperature == 0.0:
                new_tokens = tf.argmax(logits, axis=-1)
            else:
                logits = tf.squeeze(logits, axis=1)
                new_tokens = tf.random.categorical(logits / temperature,
                                                   num_samples=1)
            done = done | (new_tokens == self.end_token)
            in_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            result_tokens.append(in_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        print( tf.concat(result_tokens, axis=-1))
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.target_processor.convert_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}
