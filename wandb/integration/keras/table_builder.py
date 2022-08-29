import wandb
from typing import Tuple, List
from abc import ABC, abstractmethod

import tensorflow as tf


class TablesBuilder(ABC):
    """
    dd
    """

    @abstractmethod
    def add_ground_truth(self):
        """Use this method to write the logic for adding validation/training
        data to `data_table` initialized using `init_data_table` method.

        Example:
            ```
            for idx, data in enumerate(dataloader):
                self.data_table.add_data(
                    idx,
                    data
                )
            ```

        This method is called once `on_train_begin` or equivalent hook.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.add_ground_truth")


    @abstractmethod
    def add_model_predictions(self):
        """Use this method to write the logic for adding model prediction for
        validation/training data to `pred_table` initialized using
        `init_pred_table` method.

        Example:
            ```
            # Assuming the dataloader is not shuffling the samples.
            for idx, data in enumerate(dataloader):
                preds = model.predict(data)

                self.pred_table.add_data(
                    self.data_table_ref.data[idx][0],
                    self.data_table_ref.data[idx][1],
                    preds
                )
            ```

        This method is called `on_epoch_end` or equivalent hook.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.add_model_predictions")

    def init_data_table(self, column_names: list):
        """Initialize the W&B Tables for validation data.
        Call this method `on_train_begin` or equivalent hook. This is followed by
        adding data to the table row or column wise.

        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

    def init_pred_table(self, column_names: list):
        """Initialize the W&B Tables for model evaluation.
        Call this method `on_epoch_end` or equivalent hook. This is followed by
        adding data to the table row or column wise.

        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.pred_table = wandb.Table(columns=column_names)

    def log_data_table(self, 
                       name: str='val',
                       type: str='dataset',
                       table_name: str='val_data'):
        """Log the `data_table` as W&B artifact and call
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded data (images, text, scalar, etc.).
        This allows the data to be uploaded just once.

        Args:
            name (str):  A human-readable name for this artifact, which is how 
                you can identify this artifact in the UI or reference 
                it in use_artifact calls. (default is 'val')
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        data_artifact = wandb.Artifact(name, type=type)
        data_artifact.add(self.data_table, table_name)

        # Calling `use_artifact` uploads the data to W&B.
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get(table_name)

    def log_pred_table(self,
                       type: str='evaluation'
                       table_name: str='eval_data'):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.

        Args:
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        pred_artifact = wandb.Artifact(
            f'run_{wandb.run.id}_pred', type=type)
        pred_artifact.add(self.pred_table, table_name)
        # TODO: Add aliases
        wandb.run.log_artifact(pred_artifact)