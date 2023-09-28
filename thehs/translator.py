import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .tokenizer import Tokenizer
from .model import AutoModel
from .model import BaseMT, BaseConfig


class Translator(tf.Module):

    def __init__(self, config: BaseConfig, model: str | BaseMT):
        super(Translator, self).__init__()
        self.tokenizer = Tokenizer(processor=config.processor)
        if isinstance(model, str):
            self.translator = AutoModel.from_pretrained(model)
        elif isinstance(model, BaseMT):
            self.translator = model
        else:
            raise TypeError(f"type {type(model)} is not in supported type: keras.Model")

        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id

        token_mask = np.zeros(self.tokenizer.vocab_size, bool)
        token_mask[[self.pad_id, self.bos_id]] = True
        self.token_mask = tf.convert_to_tensor(token_mask[tf.newaxis, tf.newaxis, :])

    def __call__(self, input_ids):
        enc_out = self.translator.encode_input(input_ids)
        pass

    def __greedy_search(self, input_ids):
        prev_out = self.translator.encode_input(input_ids)
        batch_size = tf.shape(input_ids)[0]
        first_ids = tf.reshape(tf.cast([self.bos_id] * batch_size.numpy(), tf.int64), shape=(-1, 1))
        target_in_ids = self.translator.update_target_in_ids(first_ids)
        input_mask = tf.cast(input_ids != 0, dtype=tf.float32)
        done = tf.zeros([batch_size, 1], dtype=bool)
        for _ in tf.range(100):
            logits, prev_out = self.translator.next_ids(target_in_ids, prev_out, input_mask)
            # attn_array = attn_array.write(i, tf.squeeze(self.decoder.attention_weights))

            # Set the logits for all masked tokens to -inf, so they are never chosen.
            logits = tf.where(self.token_mask, -np.inf, logits)

            new_ids = tf.argmax(logits, axis=-1)

            done = done | (new_ids == self.eos_id)

            target_out_ids = tf.where(done, tf.constant(0, dtype=tf.int64), new_ids)

            target_in_ids = self.translator.update_target_in_ids(target_out_ids)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        return self.translator.result()

    def __beam_search(self):
        pass

    def translate(self, texts, greedy: bool = True):
        self.translator.reset_array()
        encoded_data = self.tokenizer(texts, return_tensors=True)
        input_ids = encoded_data["input_ids"]
        input_ids = input_ids.to_tensor()

        output_ids = self.__greedy_search(input_ids)
        list_output_ids = output_ids.numpy().tolist()
        output_str = self.tokenizer.decode_ids(list_output_ids)

        return output_str
