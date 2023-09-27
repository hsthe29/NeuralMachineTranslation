import tensorflow as tf
from sentencepiece import SentencePieceProcessor
import os

available_processors = {"bpe.8k", "bpe.16k", "bpe.32k", "bpe.64k"}


def __retrieve_processor(name):
    processor_path = os.path.join("assets", "tokenizer", f"{name}.model")
    if name in available_processors:
        return processor_path
    else:
        raise ValueError(f"Unavailable processor path: {processor_path}!")


def load_processor(name):
    model_file = __retrieve_processor(name)
    processor = SentencePieceProcessor(model_file=model_file)

    print(
        f"\033[92mLoaded processor\033[00m {name} \033[92mand found\033[00m {processor.piece_size()} \033[92munique "
        f"tokens.\033[00m")
    return processor


class Tokenizer(tf.Module):
    def __init__(self, processor):
        super(Tokenizer, self).__init__()
        self.processor = load_processor(processor)
        self.prefix = processor

    def __call__(self, text: str | list[str],
                 return_tensors: bool = False,
                 padding: bool = False,
                 max_length: int = None,
                 pad_id: int = None):
        """
        :param text: str | list[str]
        :param return_tensors:
        :param padding:
        :param max_length:
        :param pad_id:
        :return:
        """
        if pad_id is None:
            pad_id = self.processor.pad_id()

        if isinstance(text, str):
            input_ids = self.encode(text,
                                    return_tensors=False,
                                    padding=padding,
                                    max_length=max_length,
                                    pad_id=pad_id)

            attention_mask = []
            for i in input_ids:
                attention_mask.append(int(i != 0))
            if return_tensors:
                input_ids = tf.ragged.constant([input_ids])
                attention_mask = tf.ragged.constant([attention_mask])
        elif isinstance(text, list):
            input_ids = []
            attention_mask = []
            for txt in text:
                sub_ids = self.encode(txt,
                                      return_tensors=False,
                                      padding=padding,
                                      max_length=max_length,
                                      pad_id=pad_id)
                sub_mask = []
                for i in sub_ids:
                    sub_mask.append(int(i != 0))
                input_ids.append(sub_ids)
                attention_mask.append(sub_mask)

            if return_tensors:
                input_ids = tf.ragged.constant(input_ids)
                attention_mask = tf.ragged.constant(attention_mask)

        else:
            raise TypeError("... ")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def tokenize(self, text: str):
        if not isinstance(text, str):
            raise TypeError(f"Text input must be a string!")
        return self.processor.EncodeAsPieces(text)

    def encode(self, text: str | list[str],
               return_tensors: bool = False,
               padding: bool = False,
               max_length: int = None,
               pad_id=None):

        bos_id = self.processor.bos_id()
        eos_id = self.processor.eos_id()
        if pad_id is None:
            pad_id = self.processor.pad_id()

        if isinstance(text, str):
            ids = self.processor.EncodeAsIds(text)
        elif isinstance(text, list):
            ids = self.processor.PieceToId(text)
        else:
            raise ValueError("input must be str or list[str]!")
        if max_length is not None:
            max_possible_length = max_length - 2
            if len(ids) >= max_possible_length:
                ids = ids[:max_possible_length]
            ids = [bos_id] + ids + [eos_id]

            if padding:
                pad_size = max_length - len(ids)
                ids = ids + [pad_id] * pad_size
        else:
            ids = [bos_id] + ids + [eos_id]

        if return_tensors:
            ids = tf.convert_to_tensor(ids)[tf.newaxis, :]

        return ids

    def convert_tokens_to_ids(self, tokens):
        return self.processor.PieceToId(tokens)

    def decode_ids(self, input_ids):
        return self.processor.DecodeIds(input_ids)

    def decode_tokens(self, tokens):
        return self.processor.DecodePieces(tokens)

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    @property
    def pad_id(self):
        return self.processor.pad_id()

    @property
    def bos_id(self):
        return self.processor.bos_id()

    @property
    def eos_id(self):
        return self.processor.eos_id()
