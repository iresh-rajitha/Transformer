import tensorflow as tf


def loss_function(y_true, y_pred, max_length):
    y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)
