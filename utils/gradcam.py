"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping)
für die visuelle Erklärung der Modellentscheidungen.

Erzeugt Heatmaps, die zeigen, welche Bildregionen für die
Frakturerkennung relevant waren.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Denormalisierung für Visualisierung
DENORM = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class GradCAM:
    """
    Grad-CAM für beliebige PyTorch-Modelle.

    Verwendung:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(image_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Berechnet Grad-CAM Heatmap.

        Args:
            image_tensor: (1, 3, 224, 224) auf CPU oder GPU

        Returns:
            heatmap: (224, 224) numpy array, Werte in [0, 1]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x      = image_tensor.to(device)
        x.requires_grad_(True)

        # Forward pass
        logits = self.model(x)
        prob   = torch.sigmoid(logits)

        # Backward pass
        self.model.zero_grad()
        logits.backward()

        # Grad-CAM Berechnung
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global Avg Pooling
        cam      = (weights * self.activations).sum(dim=1, keepdim=True)
        cam      = F.relu(cam)                                      # nur positive Einflüsse

        # Auf Bildgröße interpolieren
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalisieren
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, float(prob.item())


def visualize_gradcam(model, target_layer, image_path: str = None,
                      image_tensor: torch.Tensor = None,
                      save_path: str = None,
                      class_names=("No Fracture", "Fracture"),
                      title: str = "Grad-CAM"):
    """
    Erzeugt und zeigt Grad-CAM Visualisierung.

    Entweder image_path oder image_tensor muss angegeben werden.
    """
    # Bild laden
    if image_path is not None:
        pil_img = Image.open(image_path).convert("RGB")
        tensor  = VAL_TRANSFORMS(pil_img).unsqueeze(0)
    elif image_tensor is not None:
        tensor  = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        # Für Anzeige denormalisieren
        pil_arr = DENORM(tensor.squeeze()).permute(1, 2, 0).clamp(0, 1).numpy()
        pil_img = Image.fromarray((pil_arr * 255).astype(np.uint8))
    else:
        raise ValueError("image_path oder image_tensor muss angegeben werden.")

    grad_cam   = GradCAM(model, target_layer)
    heatmap, prob = grad_cam(tensor)
    grad_cam.remove_hooks()

    pred_class = 1 if prob >= 0.5 else 0
    pred_label = class_names[pred_class]

    # Original-Bild
    orig_arr = np.array(pil_img.resize((224, 224))) / 255.0

    # Heatmap einfärben und überlagern
    colored_heatmap = cm.jet(heatmap)[..., :3]
    overlay         = 0.5 * orig_arr + 0.5 * colored_heatmap

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"{title}  |  Vorhersage: {pred_label} ({prob:.1%})",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(orig_arr, cmap="gray" if orig_arr.ndim == 2 else None)
    axes[0].set_title("Original Röntgenbild")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(cm.ScalarMappable(cmap="jet"), ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay.clip(0, 1))
    axes[2].set_title("Überlagerung")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grad-CAM gespeichert: {save_path}")
    plt.show()
    return heatmap, prob


def batch_gradcam(model, target_layer, val_loader, device,
                  n_samples: int = 6, class_names=("No Fracture", "Fracture"),
                  save_path: str = None):
    """
    Erstellt Grad-CAM Visualisierungen für mehrere Bilder aus dem Val-Set.
    """
    model.eval()
    images_list, labels_list = [], []

    for images, labels in val_loader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(len(b) for b in images_list) >= n_samples:
            break

    images = torch.cat(images_list)[:n_samples]
    labels = torch.cat(labels_list)[:n_samples]

    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.5))
    fig.suptitle("Grad-CAM Batch-Visualisierung", fontsize=14, fontweight="bold")

    for i in range(n_samples):
        img_t  = images[i]
        heatmap, prob = grad_cam(img_t.unsqueeze(0))

        # Bild denormalisieren
        orig = DENORM(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
        colored = cm.jet(heatmap)[..., :3]
        overlay = (0.5 * orig + 0.5 * colored).clip(0, 1)

        pred  = 1 if prob >= 0.5 else 0
        truth = int(labels[i].item())
        color = "green" if pred == truth else "red"

        axes[i, 0].imshow(orig, cmap="gray")
        axes[i, 0].set_title(f"GT: {class_names[truth]}", color=color, fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Pred: {class_names[pred]} ({prob:.0%})", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis("off")

    grad_cam.remove_hooks()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Batch Grad-CAM gespeichert: {save_path}")
    plt.show()