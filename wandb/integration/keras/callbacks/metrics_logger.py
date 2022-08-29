import wandb
from typing import Optional

from tensorflow.keras import callbacks

from ..keras import patch_tf_keras


patch_tf_keras()


class WandBMetricsLogger(callbacks.Callback):
    def __init__(self, log_batch_frequency: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_batch_frequency = log_batch_frequency

    def on_epoch_end(self, epoch, logs={}):
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

    def on_batch_end(self, batch, logs={}):
        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)

    def on_train_batch_end(self, batch, logs={}):
        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)
