import tensorflow as tf


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name="masked_loss"):
        super(MaskedLoss, self).__init__(name=name)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.loss(y_true, y_pred, sample_weight=sample_weight)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)
