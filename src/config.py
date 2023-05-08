
DATA_PATH = r"D:\Dataset\PhoMT_small"
VOCAB_PATH = {
    'source': r"D:\Dataset\vocab\en.txt",
    'target': r"D:\Dataset\vocab\vi.txt"
}
SOURCE_LANGUAGE = 'en'
TARGET_LANGUAGE = 'vi'
TRAINING_SIZE = 5000
VALIDATION_SIZE = 500
VOCAB_SIZE = 1000
BATCH_SIZE = 64
EPOCHS = 50

START_TOKEN = '[START]'
END_TOKEN = '[END]'
MASK_TOKEN = ''
OOV_TOKEN = '[UNK]'
