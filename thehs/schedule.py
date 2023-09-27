import tensorflow as tf
from tensorflow import keras


class WarmupLinearSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, num_train_steps, num_warmup_steps):
        super(WarmupLinearSchedule, self).__init__()
        self.init_lr = init_lr
        self.lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            init_lr,
            num_train_steps - num_warmup_steps,
            0.0,
            power=1)
        self.num_train_steps = tf.cast(num_train_steps, dtype=tf.float32)
        self.num_warmup_steps = tf.cast(num_warmup_steps, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        is_warmup = tf.cast(step < self.num_warmup_steps, tf.float32)

        decay_step = (1.0 - is_warmup) * (step - self.num_warmup_steps) + is_warmup * step

        learning_rate = self.lr_fn(decay_step)

        warmup_percent_done = step / self.num_warmup_steps
        warmup_learning_rate = self.init_lr * warmup_percent_done
        final_learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        return final_learning_rate


class TransformerWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(TransformerWarmupSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class StairReductionSchedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, base_steps, reduction_steps, reduction_proportion=0.9):
        super(StairReductionSchedule, self).__init__()

        self.init_lr = init_lr
        self.base_steps = tf.cast(base_steps, dtype=tf.int64)
        self.reduction_steps = tf.cast(reduction_steps, dtype=tf.float32)
        self.reduction_proportion = tf.cast(reduction_proportion, dtype=tf.float32)

    def __call__(self, step):
        is_base = tf.cast(step < self.base_steps, dtype=tf.float32)
        reduce_steps = tf.cast(step - self.base_steps, dtype=tf.float32)
        num_exp = reduce_steps // self.reduction_steps + 1.0
        reduction_rate = tf.math.pow(self.reduction_proportion, num_exp)
        return self.init_lr * (is_base + (1 - is_base) * reduction_rate)
