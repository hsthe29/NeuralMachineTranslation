import tensorflow as tf
from tensorflow import keras
from src.model.Decoder import Decoder
from src.model.Encoder import Encoder


class Translator(keras.Model):
    def __init__(self, units, source_embedding_size,
                 target_embedding_size,
                 source_processor,
                 target_processor):
        super(Translator, self).__init__()

        # Build the encoder and decoder
        self.encoder = Encoder(source_processor.vocabulary_size(), source_embedding_size, units)
        self.decoder = Decoder(target_processor.vocabulary_size(), target_embedding_size, units)

        self.source_processor = source_processor
        self.target_processor = target_processor

    def call(self, inputs):
        in_tok, tar_in = inputs
        input_mask = in_tok != 0
        enc_output, enc_state = self.encoder(in_tok)

        logits, dec_state = self.decoder(tar_in, enc_output, input_mask=input_mask, state=enc_state)
        return logits, dec_state

    @tf.function
    def train_step(self, inputs):
        in_tok, out_tok = inputs
        tar_in = out_tok[:, :-1]
        tar_out = out_tok[:, 1:]
        tokens = (in_tok, tar_in)
        target_mask = tar_out != 0

        with tf.GradientTape() as tape:
            logits, _ = self.call(tokens)
            y_true = tar_out
            y_pred = logits
            step_loss = self.loss(y_true, y_pred)
            step_loss /= tf.reduce_sum(tf.cast(target_mask, tf.float32))

        variables = self.trainable_variables
        gradients = tape.gradient(step_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'loss': step_loss}

    def translate(self, input_text, max_output_len):
        pass
