from src.model.processor import Language
from src.model.translator import Translator
from src.utils import *
import tensorflow as tf
import yaml

if __name__ == "__main__":
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    checkpoint_filepath = 'checkpoint/translator_v1.tf'

    en_vocab = load_vocab('vocab/vocab.en')
    vi_vocab = load_vocab('vocab/vocab.vi')

    special_tokens = get_special_tokens(config)

    en_processor = Language(en_vocab, special_tokens)
    vi_processor = Language(vi_vocab, special_tokens)

    pre_model = Translator(en_processor, vi_processor, config)
    pre_model.load_weights(checkpoint_filepath)

    # dev = load_dataset(config['dev_en'])
    # normalize(dev)
    # samples = []
    #
    # indexes = np.random.choice(np.arange(len(dev)), size=10)
    #
    # for index in indexes:
    #     samples.append(dev[index])

    # samples = ["check off the things you accomplish each day , and reflect on how you feel afterwards ."]
    #
    samples = ["we do n't talk anymore like we used to do ."]
    # samples = tokenize_vi(samples)
    for sentence in samples:
        print(' -> Source: ', sentence)
        for temp in [0.0, 0.9]:
            result = pre_model.translate(sentence, max_len=40, temperature=temp)
            result_texts = result['text']
            print('temperature:', temp)
            for bt in result_texts:
                pred_sentence = bt.numpy().decode()
                print(' -> Target: ', pred_sentence)
        print()

    result = pre_model.translate(sentence, max_len=40)
    result_texts = result['text']
    attentions = result['attention']
    visualize_attention(attentions[-1], samples[0], result_texts[0])
