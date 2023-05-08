import os
import tensorflow as tf
import numpy as np


def get_all_file(path):
    all_files = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            all_files.append(os.path.join(path, name))

    return all_files


def make_vocabulary(sentences, vocab_size):
    word_dict = {}
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 0
    print(f"Found {len(word_dict.keys())} words")
    sorted_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    return list(map(lambda x: x[0], sorted_dict))


def load_vocabulary(paths):
    with open(paths['source'], encoding='utf-8') as f:
        source_vocab = f.readlines()

    with open(paths['target'], encoding='utf-8') as f:
        target_vocab = f.readlines()

    source_vocab = list(map(lambda x: x.strip(), source_vocab))
    target_vocab = list(map(lambda x: x.strip(), target_vocab))

    return source_vocab, target_vocab


def text_to_tokens(processor, text):
    text = tf.convert_to_tensor(text)
    if len(text.shape) == 0:
        text = tf.convert_to_tensor(text)[tf.newaxis]
    context = processor(text).to_tensor()

    return context


def tokens_to_text(processor, tokens):
    vocab = np.asarray(processor.get_vocabulary())
    result_text_tokens = tf.map_fn(fn=lambda x: vocab[x.numpy()],
                                   elems=tokens.to_tensor(), dtype=tf.string)
    #  = list(map(lambda x: ' '.join(vocab[x]), tokens))

    result_text = tf.strings.reduce_join(result_text_tokens,
                                         axis=1, separator=' ')

    result_text = tf.strings.strip(result_text)
    return result_text
