import numpy as np
import tensorflow as tf

__all__ = ['load_dataset', 'normalize', 'take_dataset', 'split_dataset']


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    print(f"Loaded from {path}: {len(sentences)} sentences")
    return sentences


def normalize(sentences):
    n = len(sentences)
    for i in range(n):
        sentences[i] = sentences[i].strip().lower()


def take_dataset(source_sentences, target_sentences, corpus_size=None, threshold=40):
    train_src_raw = []
    train_tar_raw = []
    max_sentences = len(source_sentences)

    n = 0
    for i in range(max_sentences):
        if len(source_sentences[i].split()) <= threshold:
            train_src_raw.append(source_sentences[i])
            train_tar_raw.append(target_sentences[i])
            n += 1
            if n == corpus_size:
                break

    return train_src_raw, train_tar_raw


def split_dataset(sentence_pairs, batch_size, ratio=0.8):
    sentence_pairs = tuple(sentence_pairs)
    ratio += 0.01
    size = len(sentence_pairs)
    is_train = np.random.uniform(size=(size,)) < ratio

    train_en_selection = []
    val_en_selection = []
    train_vi_selection = []
    val_vi_selection = []

    for i, pair in enumerate(sentence_pairs):
        if is_train[i]:
            train_en_selection.append(pair[0])
            train_vi_selection.append(pair[1])
        else:
            val_en_selection.append(pair[0])
            val_vi_selection.append(pair[1])

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((train_en_selection, train_vi_selection))
        .shuffle(size)
        .batch(batch_size))
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((val_en_selection, val_vi_selection))
        .shuffle(size)
        .batch(batch_size))

    return train_raw, val_raw
