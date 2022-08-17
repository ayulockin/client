import wandb

import tensorflow as tf
from tensorflow.keras import callbacks

from ..keras import patch_tf_keras

patch_tf_keras()


class WandbModelLogger(callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_rendered = False

    def _render_graph(self):
        if not self._graph_rendered:
            wandb.run.summary["graph"] = wandb.Graph.from_keras(self.model)
            self._graph_rendered = True

    def on_batch_end(self, batch, logs=None):
        # Couldn't do this in train_begin because keras may still not be built
        self._render_graph()

    def on_train_batch_end(self, batch, logs=None):
        # Couldn't do this in train_begin because keras may still not be built
        self._render_graph()
