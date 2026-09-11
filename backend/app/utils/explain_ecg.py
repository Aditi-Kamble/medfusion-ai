"""
Grad-CAM explainability for the ECG branch (ECGNet).
Produces a heatmap showing which regions of the ECG image most influenced
the model's prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from backend.app.utils.preprocess_ecg import get_train_test_transforms, IDX_TO_CLASS


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook the last conv layer in conv_block (before the AdaptiveAvgPool2d)
        target_layer = self.model.conv_block[-2]  # the last ReLU before pooling
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, 3, 224, 224) preprocessed image tensor
        Returns: heatmap as a (224, 224) numpy array, values 0-1
        """
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to match input image size (224x224)
        cam = cv2.resize(cam, (224, 224))

        return cam, target_class


def generate_gradcam_overlay(model, image_path, save_path):
    """
    Runs Grad-CAM on a real ECG image and saves a heatmap overlay to save_path.
    Returns the predicted class name.
    """
    _, transform = get_train_test_transforms()

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    input_tensor.requires_grad_(True)

    gradcam = GradCAM(model)
    heatmap, predicted_idx = gradcam.generate(input_tensor)

    # Build the overlay
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_np = np.array(img_resized)
    overlay = np.uint8(0.5 * original_np + 0.5 * heatmap_color)

    Image.fromarray(overlay).save(save_path)

    return IDX_TO_CLASS[predicted_idx]