import config
from src.dataset import load_dataset, preprocess_data


if __name__ == "__main__":
    train_en = load_dataset(config.train_data + 'train.en')
    train_vi = load_dataset(config.train_data + 'train.vi')
    preprocess_data(train_en, 'en')
    preprocess_data(train_vi, 'vi')

    dev_en = load_dataset(config.train_data + 'dev.en')
    dev_vi = load_dataset(config.train_data + 'dev.vi')
    preprocess_data(dev_en, 'en')
    preprocess_data(dev_vi, 'vi')

    test_en = load_dataset(config.train_data + 'test.en')
    test_vi = load_dataset(config.train_data + 'test.vi')
    preprocess_data(test_en, 'en')
    preprocess_data(test_vi, 'vi')

    with open('./data/normalized/train/train.en', 'w') as f:
        for sent in train_en:
            f.write(sent+"\n")
    with open('./data/normalized/train/train.vi', 'w') as f:
        for sent in train_vi:
            f.write(sent+"\n")

    with open('./data/normalized/dev/dev.en', 'w') as f:
        for sent in dev_en:
            f.write(sent+"\n")
    with open('./data/normalized/dev/dev.vi', 'w') as f:
        for sent in dev_vi:
            f.write(sent+"\n")

    with open('./data/normalized/test/test.en', 'w') as f:
        for sent in test_en:
            f.write(sent+"\n")
    with open('./data/normalized/test/test.vi', 'w') as f:
        for sent in test_vi:
            f.write(sent+"\n")
