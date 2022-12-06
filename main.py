import numpy as np
import tensorflow as tf
from tensorflow import Tensor, keras
from keras import layers
from keras import models

x_train = []
y_train = []
x_test = []
y_test = []


def init_sets():
    return np.load('x_train.npy'), np.load('y_train.npy'), np.load('x_test.npy'), np.load('y_test.npy')


def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2D(kernel_size=kernel_size,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv2D(kernel_size=kernel_size,
                      strides=1,
                      filters=filters,
                      padding="same")(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1,
                          strides=2,
                          filters=filters,
                          padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net():
    inputs = layers.Input(shape=(300, 300, 3))
    num_filters = 32

    t = layers.BatchNormalization()(inputs)
    t = layers.Conv2D(kernel_size=3,
                      strides=1,
                      filters=num_filters,
                      padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = layers.AveragePooling2D(4)(t)
    t = layers.Flatten()(t)

    t = layers.Dense(1272, activation="relu")(t)
    t = layers.Dropout(0.5)(t)
    t = layers.Dense(228, activation="relu")(t)
    t = layers.Dropout(0.5)(t)
    outputs = layers.Dense(6, activation='softmax')(t)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def first_start():
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = init_sets()
    return create_res_net()


with tf.device("/gpu:0"):
    model = first_start()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer="sgd", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=15)

model.save("model.hdf5")
