import tensorflow as tf
from tensorflow.keras import layers

def res_block(x, num_hidden):
    shortcut = x
    x = layers.Conv2D(num_hidden, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_hidden, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape, action_size, num_resBlocks, num_hidden):
    inputs = tf.keras.Input(shape=input_shape) 

    x = layers.Conv2D(num_hidden, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(num_resBlocks):
        shortcut = x
        x = layers.Conv2D(num_hidden, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_hidden, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

    p = layers.Conv2D(32, 3, padding="same")(x)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Flatten()(p)
    policy_output = layers.Dense(action_size, name="policy")(p)

    v = layers.Conv2D(3, 3, padding="same")(x)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(1)(v)
    value_output = layers.Activation("tanh", name="value")(v)

    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output])
