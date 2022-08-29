import os
import wandb
import shutil

from tensorflow.keras import callbacks

from ..keras import patch_tf_keras


patch_tf_keras()


class WandbModelCheckpointCallback(callbacks.Callback):
    def __init__(
        self,
        save_weights_only: bool = False,
        save_model_as_artifact: bool = False,
        verbose: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_weights_only = save_weights_only
        self.save_model_as_artifact = save_model_as_artifact
        self.verbose = verbose
        self.filepath = os.path.join(wandb.run.dir, "model-best.h5")

    def _save_model(self, epoch):
        if wandb.run.disabled:
            return
        if self.verbose > 0:
            print(
                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                " saving model to %s"
                % (epoch, self.monitor, self.best, self.current, self.filepath)
            )

        try:
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
        # Was getting `RuntimeError: Unable to create link` in TF 1.13.1
        # also saw `TypeError: can't pickle _thread.RLock objects`
        except (ImportError, RuntimeError, TypeError) as e:
            wandb.termerror(
                "Can't save model in the h5py format. The model will be saved as "
                "as an W&B Artifact in the 'tf' format."
            )

    def _save_model_as_artifact(self, epoch):
        if wandb.run.disabled:
            return

        # Save the model in the SavedModel format.
        # TODO: Replace this manual artifact creation with the `log_model` method
        # after `log_model` is released from beta.
        if self.save_weights_only:
            self.model.save_weights(
                self.filepath[:-3], overwrite=True, save_format="tf", options=None
            )
        else:
            self.model.save(self.filepath[:-3], overwrite=True, save_format="tf")

        # Log the model as artifact.
        name = wandb.util.make_artifact_name_safe(f"model-{wandb.run.name}")
        model_artifact = wandb.Artifact(name, type="model")
        model_artifact.add_dir(self.filepath[:-3])
        wandb.run.log_artifact(model_artifact, aliases=["latest", f"epoch_{epoch}"])

        # Remove the SavedModel from wandb dir as we don't want to log it to save memory.
        shutil.rmtree(self.filepath[:-3])

    def on_epoch_end(self, epoch, logs=None):
        self._save_model(epoch)
        if self.save_model_as_artifact:
            self._save_model_as_artifact(epoch)
