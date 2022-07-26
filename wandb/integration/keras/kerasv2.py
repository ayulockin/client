"""
keras init
"""

import wandb

import tensorflow as tf


class BaseWandbCallback(tf.keras.callbacks.Callback):
    """
    doc string
    """
    def __init__(
        self,
        monitor="val_loss",
        mode="auto",
    ):

        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wandb.wandb_lib.telemetry.context(run=wandb.run) as tel:
            tel.feature.keras = True
        