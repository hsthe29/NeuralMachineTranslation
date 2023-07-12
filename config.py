model = "Neural Machine Translator"
train_data = "path to train folder"
dev_data = "path to dev folder"
test_data = "path to test folder"

ckpt_dir = 'saved/translator.tf'

src_language = "en"
tar_LANGUAGE = "vi"
max_length = 80
recurrent_units = 512
embedding_size = 256
train_size = 1_000_000
val_size = 15_000
test_size = 15_000
batch_size = 128
epochs = 50
