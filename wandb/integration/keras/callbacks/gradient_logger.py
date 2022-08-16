import wandb
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, callbacks

from ..keras import patch_tf_keras, _GradAccumulatorCallback, _CustomOptimizer


tf_logger = tf.get_logger()

patch_tf_keras()


class WandbGradientLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        training_data,
        input_shape: Optional[Sequence[int]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self._set_training_data(training_data)

    def _set_training_data(self, training_data):
        self.training_data = training_data
        if int(tf.__version__.split(".")[0]) < 2:
            raise Exception("Gradient logging requires tensorflow 2.0 or higher.")
        if self.training_data is None:
            raise ValueError("training_data argument is required for gradient logging.")
        if isinstance(self.training_data, (list, tuple)):
            if len(self.training_data) != 2:
                raise ValueError("training data must be a tuple of length two")
            self._training_data_x, self._training_data_y = self.training_data
        else:
            self._training_data_x = self.training_data  # generator, tf.data.Dataset etc
            self._training_data_y = None

    def _build_grad_accumulator_model(self):
        inputs = (
            keras.Input(shape=self.input_shape)
            if self.input_shape is not None
            else self.model.inputs
        )
        outputs = self.model(inputs)
        grad_acc_model = models.Model(inputs, outputs)
        grad_acc_model.compile(loss=self.model.loss, optimizer=_CustomOptimizer())

        # make sure magic doesn't think this is a user model
        grad_acc_model._wandb_internal_model = True

        self._grad_accumulator_model = grad_acc_model
        self._grad_accumulator_callback = _GradAccumulatorCallback()

    def set_model(self, model):
        self.model = model
        self._build_grad_accumulator_model()

    def _log_gradients(self):
        # Suppress callback warnings grad accumulator
        og_level = tf_logger.level
        tf_logger.setLevel("ERROR")

        self._grad_accumulator_model.fit(
            self._training_data_x,
            self._training_data_y,
            verbose=0,
            callbacks=[self._grad_accumulator_callback],
        )
        tf_logger.setLevel(og_level)
        weights = self.model.trainable_weights
        grads = self._grad_accumulator_callback.grads
        metrics = {}
        for (weight, grad) in zip(weights, grads):
            metrics[
                "gradients/" + weight.name.split(":")[0] + ".gradient"
            ] = wandb.Histogram(grad)
        return metrics

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(self._log_gradients(), commit=False)
