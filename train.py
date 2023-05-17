from src.dataset import *
from src.losses import *
from src.metrics import *
from src.model.processor import LanguageProcessor
from src.model.translator import Translator
import tensorflow as tf
from src.utils import *
import yaml


def make_checkpoint():
    checkpoint_filepath = 'checkpoint/translator.tf'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_masked_acc',
        mode='max',
        save_best_only=True
    )
    return checkpoint_callback


def main(config):
    en_sentences = load_dataset(config['en_data'])
    vi_sentences = load_dataset(config['vi_data'])

    en_vocab = make_vocabulary(en_sentences, config['vocab_size'])
    vi_vocab = make_vocabulary(vi_sentences, config['vocab_size'])

    normalize_dataset(en_sentences, vi_sentences)
    train_en_raw, train_vi_raw = take_dataset(en_sentences, vi_sentences, config['sentences'])

    train_raw, val_raw = split_dataset(zip(train_en_raw, train_vi_raw),
                                       batch_size=config['batch_size'], ratio=config['train_ratio'])

    en_processor = LanguageProcessor(en_vocab)
    vi_processor = LanguageProcessor(vi_vocab)

    def convert_dataset(source, target):
        return en_processor.convert_to_tensor(source), vi_processor.convert_to_tensor(target)

    train_ds = train_raw.map(convert_dataset, tf.data.AUTOTUNE)
    val_ds = val_raw.map(convert_dataset, tf.data.AUTOTUNE)

    model = Translator(en_processor, vi_processor, config)

    model.compile(optimizer=tf.optimizers.Adam(config['adam']['lr'],
                                               config['adam']['beta_1'],
                                               config['adam']['beta_2']),
                  loss=MaskedLoss(),
                  metrics=[masked_acc])
    checkpoint = make_checkpoint()
    history = model.fit(train_ds, epochs=config['epochs'], validation_data=val_ds, callbacks=[checkpoint])


if __name__ == '__main__':
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(tf.__version__)
    main(config)
