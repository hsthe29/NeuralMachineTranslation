from src.model.Translator import Translator
import os
from src import config
from src.utils import load_vocabulary


# def main():
#     translator = Translator()

def main():
    en, vi = load_vocabulary(config.VOCAB_PATH)
    print(en)
    print(vi)

if __name__ == '__main__':
    main()
