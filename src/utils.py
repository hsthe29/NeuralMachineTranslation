import re
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import ticker
from underthesea import word_tokenize
import config


def make_vocabulary(sentences, vocab_size):
    word_dict = {}

    with open(config.reversed_name, 'r', encoding='utf-8') as f:
        reserved_names = f.readlines()

    for i in range(len(reserved_names)):
        reserved_names[i] = reserved_names[i].strip().casefold()
    reserved_names = set(reserved_names)

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in reserved_names:
                continue
            if bool(re.search(r'\d', word)):
                continue
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 0

    print(f"Found {len(word_dict.keys())} words")
    print(f"Making dictionary of {vocab_size} words...")
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


def normalize_en(text):
    text = text.casefold()
    text = text.strip()
    text = re.sub(r'''[^ a-z0-9.?!,'":-]''', '', text)

    normalized_text = add_spaces(text)
    return normalized_text


def normalize_vi(text):
    text = text.casefold()
    text = text.strip()
    text = re.sub(r'''[^ aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z0-9.?!,'":-]''',
                  '', text)
    text = word_tokenize(text, format='text')
    normalized_text = add_spaces(text)
    return normalized_text


def text_normalize_en(texts):
    texts = tf.strings.lower(texts, encoding='utf-8')
    texts = tf.strings.regex_replace(texts, r'''[^ a-z0-9.?!,'":-]''', '')
    texts = tf.strings.regex_replace(texts, r'(\d+\.\d+|\d+)([a-z])', r'\1 \2')
    texts = tf.strings.regex_replace(texts, r'([a-z])(\d+\.\d+|\d+)', r'\1 \2')
    texts = tf.strings.regex_replace(texts, r'(\d+\.\d+|\d+?|[a-z])([.,])', r'\1 \2 ')

    texts = tf.strings.strip(texts)

    texts = tf.strings.join(['[sos]', texts, '[eos]'], separator=' ')
    return texts


def text_normalize_vi(texts):
    texts = tf.strings.lower(texts, encoding='utf-8')
    texts = tf.strings.regex_replace(
        texts,
        r'''[^ aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z0-9.?!,'":-_]''',
        '')
    texts = tf.strings.regex_replace(
        texts,
        r'''(\d+\.\d+|\d+)([aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z])''',
        r'\1 \2')
    texts = tf.strings.regex_replace(
        texts,
        r'''([aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z])(\d+\.\d+|\d+)''',
        r'\1 \2')
    texts = tf.strings.regex_replace(
        texts,
        r'''(\d+\.\d+|\d+?|[aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z])([.,])''',
        r'\1 \2 ')
    texts = tf.strings.strip(texts)

    texts = tf.strings.join(['[sos]', texts, '[eos]'], separator=' ')
    return texts


def add_spaces(text):
    text = re.sub(r'(\d+\.\d+|\d+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d+\.\d+|\d+)', r'\1 \2', text)
    text = re.sub(r'(?<!\d\.\d)([.,])(?!\d)', r' \1 ', text)
    return text


def get_special_tokens():
    return config.mask_token, config.oov_token, config.start_token, config.end_token


def visualize_attention(attention, sentence, predicted_sentence):
    sentence = tf.squeeze(text_normalize_en(sentence)).numpy().decode().split()
    predicted_sentence = tf.squeeze(text_normalize_vi(predicted_sentence)).numpy().decode().split()[1:]
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

    # plt.savefig('result/attention/attention2.png')

    plt.show()


def plot_history(history, save_img=False):
    history = history.history
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(ncols=2, hspace=0.0, wspace=0.2)
    axes = gs.subplots()

    axes[0].plot(history['masked_loss'])
    axes[0].plot(history['val_masked_loss'])
    axes[0].set_title('training masked loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    axes[1].plot(history['masked_acc'])
    axes[1].plot(history['val_masked_acc'])
    axes[1].set_title('training masked acc')
    axes[1].set_ylabel('accuracy')
    axes[1].set_xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save_img:
        plt.savefig('result/train/train_loss.png')
    else:
        plt.show()


def tokenize_vi(text):
    for i in range(len(text)):
        text[i] = word_tokenize(text[i], format='text')
    return text


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def save_vocabulary(vocab, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')


def load_vocab(file_path):
    vocab = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())

    return vocab


def update_vocab(corpus, vocab_size, path):
    vocab = make_vocabulary(corpus, vocab_size)
    save_vocabulary(vocab, path)
