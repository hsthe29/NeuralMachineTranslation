from src.dataset import *
from src.losses import *
from src.metrics import *
from src.model.LanguageProcessor import LanguageProcessor
from src.model.Translator import Translator
import tensorflow as tf
from src import config
from src.utils import *


def make_checkpoint():
    checkpoint_filepath = 'checkpoint/translator.tf'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        # monitor='val_accuracy',
        # mode='max',
        # save_best_only=True
    )


def main():
    en_sentences = load_dataset(config.DATA_PATH + r'\train\train.en')
    vi_sentences = load_dataset(config.DATA_PATH + r'\train\train.vi')

    en_vocab = make_vocabulary(en_sentences, config.VOCAB_SIZE)
    vi_vocab = make_vocabulary(vi_sentences, config.VOCAB_SIZE)

    normalize_dataset(en_sentences, vi_sentences)
    train_en_raw, train_vi_raw = take_dataset(en_sentences, vi_sentences, config.TRAINING_SIZE)

    train_raw, val_raw = split_dataset(train_en_raw, train_vi_raw)

    en_processor = LanguageProcessor(en_vocab)
    vi_processor = LanguageProcessor(vi_vocab)
    convert_dataset = \
        lambda source, target: \
            (en_processor.convert_to_tensor(source),
             vi_processor.convert_to_tensor(target))
    train_ds = train_raw.map(convert_dataset, tf.data.AUTOTUNE)
    val_ds = val_raw.map(convert_dataset, tf.data.AUTOTUNE)

    UNITS = 256

    model = Translator(UNITS, UNITS, UNITS, en_processor, vi_processor)

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=masked_loss,
                  metrics=[masked_acc, masked_loss])

    history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds, callbacks=[make_checkpoint()])


if __name__ == '__main__':
    # main()
    with open('checkpoint/h.txt', 'r') as f:
        print(f.readlines())
