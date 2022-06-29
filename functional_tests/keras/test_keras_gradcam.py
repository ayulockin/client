import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


np.random.seed(42)
x = np.random.randint(255, size=(100, 28, 28, 1))
y = np.random.randint(10, size=(100,))

# Simple dataset
simple_dataset = (x, y)

# tf.data.Dataset
def parse_data(x, y):
    x = tf.image.convert_image_dtype(x, dtype=tf.float32)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x, y))
trainloader = (
    train_ds
    .shuffle(32)
    .map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

run = wandb.init(project="keras")


def get_functional_model():
    inputs = tf.keras.layers.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(3, 3, activation="relu")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = get_functional_model()
# Simple dataset
# _ = model.fit(x, y, epochs=2, callbacks=[WandbCallback(validation_data=simple_dataset, log_evaluation=True)])

# tf.data.Dataset
_ = model.fit(trainloader,
             epochs=2, callbacks=[WandbCallback(generator=iter(trainloader), log_evaluation=True, validation_steps=2, log_evaluation_frequency=1)])
