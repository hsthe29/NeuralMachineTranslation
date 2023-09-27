import tensorflow as tf

from thehs import Parser
from thehs.model import AutoModel
from thehs.model import AutoConfig
from thehs.utils import save_model
from thehs.metrics import masked_acc
from thehs.losses import MaskedLoss
from thehs.dataset import make_dataset
from thehs.schedule import StairReductionSchedule
from thehs import Tokenizer


parser = Parser()

parser.DEFINE_string("name", None, "The name represents the training model")
parser.DEFINE_bool("use-warmup", False)
parser.DEFINE_float("init-lr", 1e-3)

flags = parser.parse()


if __name__ == "__main__":
    config = AutoConfig.from_file("config.json")
    tokenizer = Tokenizer(config.processor)
    model = AutoModel.from_pretrained(config.name())

    val_ds = make_dataset(("data/test/test.en", "data/test/test.vi"), tokenizer, max_length=config.max_length)

    optimizer = tf.keras.optimizers.Adam(StairReductionSchedule(1e-3, 3000, 800, reduction_proportion=0.95))
    model.compile(optimizer=optimizer,
                  loss=MaskedLoss(),
                  metrics=[masked_acc])

    model.summary()

    result = model.evaluate(val_ds, return_dict=True)
    print(result)

    # save_model(model, config.name())
