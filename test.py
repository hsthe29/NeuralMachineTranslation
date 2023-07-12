import json
import os
import tensorflow as tf
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import config
from src.language import Language
from src.model.translator import Translator
from src.model.rnn.nmt import NMT
from src.utils import *
import sys
import webbrowser


def load_translator():
    print("Loading vocabulary... ", end='')
    en_vocab = load_vocab('vocab/vocab.en')
    vi_vocab = load_vocab('vocab/vocab.vi')
    print("Done")

    special_tokens = get_special_tokens()

    english = Language(en_vocab, special_tokens, is_english=True)
    vietnamese = Language(vi_vocab, special_tokens, is_english=False)
    return english, vietnamese


if __name__ == "__main__":
    en, vi = load_translator()
    dd = "it ain't me"
    tt = en.clean(dd)
    print(en.clean(dd))
    print(en.convert_to_tensor(tt))
