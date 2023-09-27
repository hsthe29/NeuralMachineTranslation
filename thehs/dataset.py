import tensorflow as tf
from .tokenizer import Tokenizer
from tqdm import tqdm


def read_text_from_file(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())

    return lines


def transform(data: list[str] | tuple[list[str], list[str]] | list[list[str]],
              tokenizer,
              max_length=None):
    if max_length is None:
        max_length = 512

    num_removed = 0

    if len(data) == 0:
        raise TypeError("Provide list or tuple are required!")
    elif isinstance(data[0], str):
        iter_obj = data
        source_ids = []

        for src_sent in tqdm(iter_obj):
            src_ids = tokenizer.encode(src_sent)
            if len(src_ids) > max_length:
                num_removed += 1
                continue
            source_ids.append(src_ids)

        return_data = tf.ragged.constant(source_ids)

    else:
        iter_obj = zip(*data)
        source_ids = []
        target_ids = []

        num_removed = 0
        for src_sent, tar_sent in tqdm(iter_obj):
            src_ids = tokenizer.encode(src_sent)
            tar_ids = tokenizer.encode(tar_sent)
            if len(src_ids) > max_length or len(tar_ids) > max_length:
                num_removed += 1
                continue
            source_ids.append(src_ids)
            target_ids.append(tar_ids)
        return_data = (tf.ragged.constant(source_ids),
                       tf.ragged.constant(target_ids))
    print(f"Fount {num_removed} ids pairs with length greater than {max_length}. And it was removed from the dataset!")
    del data
    return len(source_ids), return_data


def make_dataset(files: str | list[str] | tuple[str, str],
                 tokenizer: Tokenizer,
                 batch_size: int = 16,
                 max_length: int | None = None,
                 train: bool = True):
    if isinstance(files, str):
        source_file = files
        target_file = None
    elif isinstance(files, list) or isinstance(files, tuple):
        source_file, target_file = files
    else:
        raise ValueError("Argument files not passed with right value!")
    source_sentences = read_text_from_file(source_file)
    if train:
        target_sentences = read_text_from_file(target_file)

        buffer_size, data = transform((source_sentences, target_sentences), tokenizer, max_length)
        del source_sentences, target_sentences
        return (tf.data.Dataset
                .from_tensor_slices(data)
                .shuffle(buffer_size)
                .batch(batch_size)
                .map(lambda x, y: (
                    (x.to_tensor(default_value=0), y[:, :-1].to_tensor(default_value=0)),
                    y[:, 1:].to_tensor(default_value=0)
                ))
                .prefetch(tf.data.AUTOTUNE))

    else:
        buffer_size, data = transform(source_sentences, tokenizer, max_length)
        del source_sentences
        return (tf.data.Dataset
                .from_tensor_slices(data)
                .map(lambda x: x.to_tensor(default_value=0))
                .prefetch(tf.data.AUTOTUNE))
