from thehs import Translator
from thehs.model import AutoModel, AutoConfig
from thehs import Tokenizer
import tensorflow as tf
import os
import time


if __name__ == "__main__":

    config = AutoConfig.from_file("config.json")
    model = AutoModel.from_pretrained(config.name())
    translator = Translator(config, model)
    en_text = "He was not given permission to be represented by defense lawyers until early July"
    en_text = "This TensorFlow binary is optimized to use available CPU instructions"
    st = time.perf_counter()
    result = translator.translate(en_text)
    end = time.perf_counter()
    vi_text = result[0]
    print("Source: ", en_text)
    print("Translation: ", vi_text)
    print("Utilization: ", end - st, "s")
