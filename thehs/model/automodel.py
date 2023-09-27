import json
import os.path

import tensorflow as tf
from .recurrent import RnnMT
from .transformer import TransformerMT
from .graph import GraphMT
from .config import BaseConfig


class AutoModel:
    __registered_models = {
        "RnnMT": RnnMT,
        "TransformerMT": TransformerMT,
        "GraphMT": GraphMT
    }

    @classmethod
    def from_config(cls, config: BaseConfig):
        model_json = config.to_json()
        class_name = config.model_class

        return tf.keras.models.model_from_json(
            model_json,
            custom_objects={
                class_name: cls.__registered_models[class_name]
            }
        )

    @classmethod
    def from_pretrained(cls, name, show_warnings=False):
        class_name, model_json, weights_path = cls.__retrieve(name)
        model = tf.keras.models.model_from_json(
            model_json,
            custom_objects={
                class_name: cls.__registered_models[class_name]
            }
        )
        if show_warnings:
            model.load_weights(weights_path)
        else:
            model.load_weights(weights_path).expect_partial()

        return model

    @classmethod
    def __retrieve(cls, name):
        with open(os.path.join("save", "saved_models.json"), "r") as f:
            saved_list = json.load(f)
        if name not in saved_list.keys():
            raise ValueError(f"Model {name} is not saved. Remember training and save model first!")
        config, weights_component = saved_list[name]
        class_name = config["class_name"]
        model_json = json.dumps(config)
        weights_path = os.path.join(*weights_component)
        return class_name, model_json, weights_path
