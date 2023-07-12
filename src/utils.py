import re
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import ticker
from underthesea import word_tokenize

__all__ = ['preprocess_en', 'preprocess_vi', 'get_special_tokens', 'plot_history',
           'create_look_ahead_mask', 'load_vocab', 'add_tokens', 'build']


def preprocess_en(text):
    text = text.casefold()
    text = text.strip()
    text = re.sub(r'''[^ a-z0-9.?!,'":-]''', '', text)
    text = re.sub(r'(\d+)([a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d+)', r'\1 \2', text)
    text = re.sub(r'''([.,?:!-'"])''', r' \1 ', text)
    text = re.sub(r'(\d+) (\.) (\d+)', r'\1.\3', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"' m", 'am', text)
    text = re.sub(r"' ve", 'have', text)
    text = re.sub(r"' re", 'are', text)
    text = re.sub(r"won ' t", 'will not', text)
    text = re.sub(r"can ' t", 'can not', text)
    # text = re.sub(r"ain ' t", 'aint', text)
    text = re.sub(r"n ' t", ' not', text)
    text = re.sub(r"(') (s|d|ll|n)", r'\1\2', text)
    return text


def preprocess_vi(text):
    text = text.casefold()
    text = text.strip()
    text = re.sub(r'''[^ aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z0-9.?!,'":-]''',
                  '', text)
    text = re.sub(r'(\d+)([aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z])',
                  r'\1 \2', text)
    text = re.sub(r'([aăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z])(\d+)',
                  r'\1 \2', text)
    text = re.sub(r'''([.,?:!-'"])''', r' \1 ', text)
    text = re.sub(r'(\d+) (\.) (\d+)', r'\1.\3', text)
    text = re.sub(r'\s+', ' ', text)
    text = word_tokenize(text, format='text')
    return text


def get_special_tokens():
    return '', '[UNK]', '[sos]', '[eos]'


def visualize_attention(attention, src_sent, pred_sent):
    src_sent_normalized = add_tokens(preprocess_en(src_sent)).numpy().decode().split()
    pred_sent_normaized = add_tokens(pred_sent).numpy().deocde().split()[1:]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    attention = attention[:len(pred_sent_normaized), :len(src_sent_normalized)]

    ax.matshow(attention, cmap='viridis', vmin=0.0)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + src_sent, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + pred_sent_normaized, fontdict=fontdict)

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


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def load_vocab(file_path):
    vocab = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip().casefold())
    return vocab


def add_tokens(texts):
    texts = tf.strings.join(['[sos]', texts, '[eos]'], separator=' ')
    return texts


def build(model, shape):
    dummy_inputs = tf.random.uniform(shape)
    model((dummy_inputs, dummy_inputs), training=False)
