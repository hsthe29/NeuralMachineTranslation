import json
import os.path
import re
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import ticker


# def visualize_attention(attention, src_sent, pred_sent):
#     src_sent_normalized = add_tokens(preprocess_en(src_sent)).numpy().decode().split()
#     pred_sent_normaized = add_tokens(pred_sent).numpy().deocde().split()[1:]
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(1, 1, 1)
#
#     attention = attention[:len(pred_sent_normaized), :len(src_sent_normalized)]
#
#     ax.matshow(attention, cmap='viridis', vmin=0.0)
#
#     fontdict = {'fontsize': 14}
#
#     ax.set_xticklabels([''] + src_sent, fontdict=fontdict, rotation=90)
#     ax.set_yticklabels([''] + pred_sent_normaized, fontdict=fontdict)
#
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     ax.set_xlabel('Input text')
#     ax.set_ylabel('Output text')
#     plt.suptitle('Attention weights')
#
#     # plt.savefig('result/attention/attention2.png')
#
#     plt.show()


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


def save_model(model, name: str):
    model_json = model.to_json()
    weights_component = ["save", "weights", name]
    weights_path = os.path.join(*weights_component)

    saved_models_path = os.path.join("save", "saved_models.json")
    with open(saved_models_path, "r") as f:
        saved_list = json.load(f)

    model.save_weights(weights_path)

    model_json = json.loads(model_json)

    saved_list[name] = [model_json, weights_component]

    with open(saved_models_path, "w") as f:
        json.dump(saved_list, f)
