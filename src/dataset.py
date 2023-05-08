import numpy as np
import tensorflow as tf
from src import config

__all__ = ['load_dataset', 'normalize_dataset', 'take_dataset', 'split_dataset']

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    print(f"Loaded from {path}: {len(sentences)} sentences")
    return sentences


def normalize_dataset(source_sentences, target_sentences):
    n = len(source_sentences)
    for i in range(n):
        source_sentences[i] = source_sentences[i].strip().lower()
        target_sentences[i] = target_sentences[i].strip().lower()


def take_dataset(source_sentences, target_sentences, corpus_size, threshold=40):
    train_src_raw = []
    train_tar_raw = []

    n = 0
    for i in range(n):
        if len(source_sentences[i].split()) <= threshold:
            train_src_raw.append(source_sentences[i])
            train_tar_raw.append(target_sentences[i])
            n += 1
            if n == corpus_size:
                break

    return train_src_raw, train_tar_raw


def split_dataset(source_sentences, target_sentences, ratio=0.8):
    ratio += 0.01
    is_train = np.random.uniform(size=(config.TRAINING_SIZE,)) < ratio

    train_en_selection = []
    val_en_selection = []
    train_vi_selection = []
    val_vi_selection = []

    for i in range(config.TRAINING_SIZE):
        if is_train[i]:
            train_en_selection.append(source_sentences[i])
            train_vi_selection.append(target_sentences[i])
        else:
            val_en_selection.append(source_sentences[i])
            val_vi_selection.append(target_sentences[i])

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((train_en_selection, train_vi_selection))
        .shuffle(config.TRAINING_SIZE)
        .batch(config.BATCH_SIZE))
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((val_en_selection, val_vi_selection))
        .shuffle(config.TRAINING_SIZE)
        .batch(config.BATCH_SIZE))

    return train_raw, val_raw
