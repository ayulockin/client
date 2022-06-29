"""init explain"""
import wandb
import tensorflow as tf


class GradCAM:
    def __init__(self,
                 model,
                 layer_name=None,
                 pred_index=None,
                 use_guided_grads=True):
        self.model = model
        self.layer_name = layer_name
        self.pred_index = pred_index
        self.use_guided_grads = use_guided_grads

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

    def get_gradcam(self, image):
        with tf.GradientTape() as tape:
            conv_outputs, preds = self.gradcam_model(image)
            if self.pred_index is None:
                self.pred_index = tf.argmax(preds[0])
            class_loss = preds[:, self.pred_index]

        grads = tape.gradient(class_loss, conv_outputs)

        if self.use_guided_grads:
            grads = self._get_guided_grads(conv_outputs, grads)

        cams = []
        for output, grad in zip(conv_outputs, grads):
            weights = tf.reduce_mean(grad, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
            cams.append(cam)

        heatmaps = []
        _, w, h, _ = image.shape
        for cam in cams:
            cam = cam[tf.newaxis, ..., tf.newaxis]
            heatmap = tf.image.resize(cam, (w,h))
            heatmap = tf.squeeze(heatmap, axis=[0, -1])
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmaps.append(heatmap.numpy())

        return heatmaps


    def _infer_target_layer(self):
        """
        Search for the last convolutional layer to perform Grad CAM.

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

    def _get_guided_grads(self, outputs, grads):
        cast_conv_outputs = tf.cast(outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        return guided_grads

    

