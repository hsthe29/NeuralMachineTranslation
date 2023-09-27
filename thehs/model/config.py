import json
import os.path
from abc import ABC, abstractmethod


class BaseConfig(ABC):
    prefix: str | None = None
    model_class: str | None = None

    @abstractmethod
    def params_dict(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def to_json(self):
        pass


class RnnConfig(BaseConfig):
    __prefix: str | None = "RNN"
    model_class: str | None = "RnnMT"

    def __init__(self,
                 num_layers=2,
                 vocab_size=16000,
                 embedding_size=256,
                 hidden_units=128,
                 dropout_rate=0.1,
                 max_length=128,
                 processor: str = "bpe.16k"):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.processor = processor

    def params_dict(self):
        return {
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate
        }

    def name(self):
        return "-".join([
            self.__prefix + str(self.max_length),
            f"N{self.num_layers}",
            f"H{self.hidden_units}",
            f"E{self.embedding_size}",
            self.processor.replace(".", "-").upper()
        ])

    def to_json(self):
        json_str = ('{'
                    f'"class_name": "{self.model_class}", '
                    f'"config": {str(self.params_dict())}'
                    '}')
        return json_str.replace("'", '"')


class AutoConfig:
    __registered_configs = {
        "RNN": RnnConfig
    }

    @classmethod
    def from_file(cls, config_file: str) -> BaseConfig:
        """
        Create model config from file.
        Structure:
            "type": one of ["RNN", "TFM", "GNN"]
                    where RNN stand for RnnMT, TFM for Transformer, GNN for Graph Neural Network
            "params": named parameter of specific config class. Read documentation of those class for more

        :param config_file: config file
        :return:
            Config instance of specific Config class (subclass of BaseConfig)
        """
        file_path = os.path.join("config", config_file)
        with open(file_path, "r") as f:
            obj = json.load(f)

        config_type = cls.__registered_configs[obj["type"]]
        params = obj["params"]
        return config_type(**params)
