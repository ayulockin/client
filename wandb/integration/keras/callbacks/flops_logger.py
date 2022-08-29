import wandb

import tensorflow as tf
from keras import callbacks

from ..keras import _can_compute_flops


class FLOPsLogger(callbacks.Callback):
    def __init__(self, log_flops_on_workspace: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_flops_on_workspace = log_flops_on_workspace

    def get_flops(self) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        if not hasattr(self, "model"):
            raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            self.model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in self.model.inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(self.model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9) / 2

    def on_train_begin(self, logs=None):
        if _can_compute_flops():
            try:
                gflops = self.get_flops()
                wandb.summary["GFLOPs"] = gflops
                if self.log_flops_on_workspace:
                    wandb.log({"GFLOPs": gflops}, commit=True)
            except Exception as e:
                wandb.termwarn("Unable to compute FLOPs for this model.")
