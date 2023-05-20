import os
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

__all__ = ['get_all_file', 'make_vocabulary', 'load_vocabulary', 'to_lower_normalize',
           'get_special_tokens', 'visualize_attention']


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


def to_lower_normalize(text):
    # Split accented characters.
    text = tf.strings.lower(text, encoding='utf-8')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[sos]', text, '[eos]'], separator=' ')
    return text


def get_special_tokens(config):
    return config['mask_token'], config['oov_token'], config['start_token'], config['end_token']


def visualize_attention(attention, sentence, predicted_sentence):
    sentence = to_lower_normalize(sentence).numpy().decode().split()
    predicted_sentence = predicted_sentence.numpy().decode().split()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    attention = attention[:len(predicted_sentence), :len(sentence)]

    ax.matshow(attention, cmap='viridis', vmin=0.0)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')
    plt.suptitle('Attention weights')

    plt.savefig('result/attention/attention1.png')

    # plt.show()
