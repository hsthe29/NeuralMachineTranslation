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
from nltk.translate.bleu_score import corpus_bleu


def load_translator(model_path):
    print("Loading vocabulary... ", end='')
    en_vocab = load_vocab('vocab/vocab.en')
    vi_vocab = load_vocab('vocab/vocab.vi')
    print("Done")

    special_tokens = get_special_tokens()

    english = Language(en_vocab, special_tokens, is_english=True)
    vietnamese = Language(vi_vocab, special_tokens, is_english=False)
    print("Building model...")
    model = NMT(english, vietnamese, config.embedding_size, config.recurrent_units)
    build(model, shape=(1, config.max_length))
    print("Loading weights...")
    model.load_weights(model_path)
    translator = Translator(english, vietnamese, model)
    print('Done! Translator has been created.')
    return translator


def calculate_bleu_score(references, predictions):
    pass


if __name__ == "__main__":
    num_evals = 5000
    with open("data/normalized/test/test.en", "r", encoding="utf-8") as f:
        en_sents = f.readlines()[:num_evals]
    with open("data/normalized/test/test.vi", "r", encoding="utf-8") as f:
        vi_ref_sents = f.readlines()[:num_evals]


    vi_references = []

    for line in vi_ref_sents:
        vi_references.append([line.strip().split()])

    translator = load_translator("saved/model_weights.ckpt")
    start_clock = time.perf_counter()
    hypotheses = []
    for i, sent in enumerate(en_sents):
        result = translator(sent, max_length=config.max_length)
        result = tf.strings.regex_replace(result, '_', ' ')
        # result = tf.strings.regex_replace(result, r'(\s+)([.,?!])', r'\2')
        hypotheses.append(result.numpy().decode().strip().split())
        # print(hypotheses[i])

    bleu = corpus_bleu(vi_references, hypotheses)
    end_clock = time.perf_counter()
    print("BLEU score: ", bleu)
    print("Took:", end_clock - start_clock, "s")
