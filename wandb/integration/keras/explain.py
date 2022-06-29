"""init explain"""
import wandb
import tensorflow as tf


class GradCAM:
    def __init__(self, model, layer_name=None, pred_index=None):
        self.model = model
        self.layer_name = layer_name
        self.pred_index = pred_index

        if layer_name is None:
            self.layer_name = self._infer_target_layer()
            if self.layer_name is None:
                wandb.termwarn(
                    "Model does not seem to contain a 4D layer. Grad CAM cannot be applied."
                )

        if self.layer_name is not None:
            self.gradcam_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(self.layer_name).output, self.model.output]
            )

    def get_gradcam(image):
        with tf.GradientTape() as tape:
            conv_outputs, preds = self.gradcam_model(image)
            if self.pred_index is None:
                self.pred_index = tf.argmax(preds[0])
            class_loss = preds[:, pred_index]

        grads = tape.gradient(class_loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = last_conv_layer_output[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    def _infer_target_layer(self):
        """
        Search for the last convolutional layer to perform Grad CAM.

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

    

