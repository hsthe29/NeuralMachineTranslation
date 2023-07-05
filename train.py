from src.dataset import *
from src.losses import *
from src.metrics import *
from src.language import Language
from src.model.translator import Translator
from src.utils import *
import config


def get_checkpoint(path, save_weights_only=True,
                   monitor='val_masked_acc',
                   mode='max'):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=save_weights_only,
        monitor=monitor,
        mode=mode,
        save_best_only=True
    )
    return checkpoint_callback


def convert_dataset(source, target):
    return en_lang.convert_to_tensor(source), vi_lang.convert_to_tensor(target)


if __name__ == '__main__':

    signal_tokens = get_special_tokens()

    train_en = load_dataset('./data/normalized/train/train.en')
    train_vi = load_dataset('./data/normalized/train/train.vi')

    val_en = load_dataset('./data/normalized/dev/dev.en')
    val_vi = load_dataset('./data/normalized/dev/dev.vi')

    if config.update_vocab:
        update_vocab(train_en, config.vocab_size, 'vocab/vocab.en')
        update_vocab(train_vi, config.vocab_size, 'vocab/vocab.vi')

    en_vocab = load_vocab('vocab/vocab.en')
    vi_vocab = load_vocab('vocab/vocab.vi')

    en_lang = Language(en_vocab, signal_tokens, 'en')
    vi_lang = Language(vi_vocab, signal_tokens, 'vi')
    ~~>  +++ 012332▁ ▂ ▃ ▄ ▅ ▆ ▇ █ ▀ ▔ ▏ ▎ ▍ ▌ ▋ ▊ ▉ ▐ ▕ ▖ ▗ ▘ ▙ ▚ ▛ ▜ ▝ ▞ ▟ ░ ▒ ▓      
    Hồ Sỹ Thế

    train_en, train_vi = filter_long_pairs(train_en, train_vi, config.train_size)
    val_en, val_vi = filter_long_pairs(val_en, val_vi, config.val_size)

    train_raw = make_dataset(train_en, train_vi, batch_size=config.batch_size)
    val_raw = make_dataset(val_en, val_vi, batch_size=config.batch_size)

    train_ds = train_raw.map(convert_dataset, tf.data.AUTOTUNE)
    val_ds = val_raw.map(convert_dataset, tf.data.AUTOTUNE)

    model = Translator(en_lang, vi_lang, config)

    model.compile(optimizer=tf.optimizers.Adam(config.optimizer['adam']['lr'],
                                               config.optimizer['adam']['beta_1'],
                                               config.optimizer['adam']['beta_2']),
                  loss=MaskedLoss(),
                  metrics=[masked_acc])
    checkpoint = get_checkpoint(config.ckpt_dir)
    history = model.fit(train_ds,
                        epochs=config.epochs,
                        validation_data=val_ds,
                        callbacks=[checkpoint,
                                   tf.keras.callbacks.EarlyStopping(patience=5)])

    plot_history(history)
