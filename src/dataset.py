def load_dataset(path):
    with open(path, 'r') as f:
        sentences = f.readlines()
    print(f"Loaded from {path}: {len(sentences)} sentences")
    return sentences


def normalize_dataset(source_sentences, target_sentences):
    n = len(source_sentences)
    for i in range(n):
        source_sentences[i] = source_sentences[i].strip().lower()
        target_sentences[i] = target_sentences[i].strip().lower()


def take_dataset(source_sentences, target_sentences, corpus_size, threshold=40):

    train_src_raw = []
    train_tar_raw = []

    n = 0
    for i in range(n):
        if len(source_sentences[i].split()) <= threshold:
            train_src_raw.append(source_sentences[i])
            train_tar_raw.append(target_sentences[i])
            n += 1
            if n == corpus_size:
                break

    return train_src_raw, train_tar_raw
