model = "Neural Machine Translator"
# "./data/normalized/train/train_normalized.en"
train_data = "path to train folder"
dev_data = "path to dev folder"
test_data = "path to test folder"

reversed_name = "./data/normalized/reserved_names.txt"
save_checkpoint_per_epochs: 1
ckpt_dir = 'checkpoint/translator.tf'
# log_dir: logs

update_vocab = False

src_language = "en"
tar_LANGUAGE = "vi"
recurrent_units = 256
embedding_size = 256
train_size = 400_000
val_size = 15_000
test_size = 15_000
vocab_size = 8000
batch_size = 64
epochs = 50

start_token = '[sos]'
end_token = '[eos]'
mask_token = ''
oov_token = '[UNK]'

optimizer = {
    'adam': {
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
    }
}
