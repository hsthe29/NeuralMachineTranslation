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
    #
    train_ds = make_dataset(("data/dev/dev.en", "data/dev/dev.vi"), tokenizer, max_length=config.max_length)
    val_ds = make_dataset(("data/test/test.en", "data/test/test.vi"), tokenizer, max_length=config.max_length)
    steps_per_ds = int(train_ds.cardinality())
    base_steps = int(2.5*steps_per_ds)
    reduction_steps = int(steps_per_ds/8)

    model = AutoModel.from_config(config)
    optimizer = tf.keras.optimizers.Adam(StairReductionSchedule(1e-3,
                                                                base_steps,
                                                                reduction_steps,
                                                                reduction_proportion=0.96))
    model.compile(optimizer=optimizer,
                  loss=MaskedLoss(),
                  metrics=[masked_acc])

    model.summary()

    model.fit(train_ds, epochs=8, validation_data=val_ds)

    save_model(model, config.name())
