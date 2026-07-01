import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logger = logging.getLogger("StarSight.GradCAM")

class GradCAM1D:
    """
    Grad-CAM implementation for 1D Convolutional networks.
    Hooks into a target convolutional layer to record activations and gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.f_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.b_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0]

    def release(self):
        """Removes the forward and backward hooks."""
        self.f_hook.remove()
        self.b_hook.remove()


def find_last_conv_layer(module: nn.Module) -> nn.Module:
    """
    Dynamically traverses a module from the end to locate the last nn.Conv1d layer.
    """
    for name, sub_module in reversed(list(module.named_modules())):
        if isinstance(sub_module, nn.Conv1d):
            return sub_module
    raise ValueError("Could not find any nn.Conv1d layer in the module.")


def min_max_normalize(cam: torch.Tensor) -> torch.Tensor:
    """
    Applies min-max scaling to normalise the activation maps to [0, 1] per sample.
    """
    batch_min = cam.min(dim=1, keepdim=True)[0]
    batch_max = cam.max(dim=1, keepdim=True)[0]
    denom = batch_max - batch_min + 1e-8
    return (cam - batch_min) / denom


def interpolate_cam(cam: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Linearly interpolates 1D activation maps to match the original input dimension.
    """
    x = cam.unsqueeze(1)  # Shape: (Batch, 1, Time)
    x_interp = nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x_interp.squeeze(1)  # Shape: (Batch, target_len)


def generate_gradcam(
    model: nn.Module,
    x_global: torch.Tensor,
    x_local: torch.Tensor,
    x_stellar: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Generates class activation maps (Grad-CAM) for both the Global and Local
    1D CNN branches of the AstroNet model.
    
    Args:
        model: Trained AstroNet instance.
        x_global: Input global view tensor.
        x_local: Input local view tensor.
        x_stellar: Tabular stellar parameters.
        device: Active PyTorch compute device.
        
    Returns:
        tuple: (global_cam, local_cam) as normalised numpy arrays.
    """
    # Force evaluation on CPU to avoid MPS hooks deadlocks
    cpu_device = torch.device("cpu")
    orig_device = next(model.parameters()).device
    model.to(cpu_device)
    model.eval()
    
    # 1. Locate target convolutional layers
    global_target = find_last_conv_layer(model.global_branch)
    local_target = find_last_conv_layer(model.local_branch)
    
    # 2. Initialize Grad-CAM hooks
    cam_global_helper = GradCAM1D(model, global_target)
    cam_local_helper = GradCAM1D(model, local_target)
    
    # Ensure gradients are tracked
    x_global_cpu = x_global.clone().detach().to(cpu_device)
    x_local_cpu = x_local.clone().detach().to(cpu_device)
    x_stellar_cpu = x_stellar.clone().detach().to(cpu_device)
    
    # 3. Forward Pass
    logits = model(x_global_cpu, x_local_cpu, x_stellar_cpu)
    
    # 4. Backward Pass (Sums logits to compute independent sample gradients in batch)
    model.zero_grad()
    logits.sum().backward(retain_graph=True)
    
    # 5. Extract activations and gradients
    g_act = cam_global_helper.activations.detach()
    g_grad = cam_global_helper.gradients.detach()
    
    l_act = cam_local_helper.activations.detach()
    l_grad = cam_local_helper.gradients.detach()
    
    # 6. Release hooks
    cam_global_helper.release()
    cam_local_helper.release()
    
    # Restore model to its original device
    model.to(orig_device)
    
    # 7. Compute Global Grad-CAM
    g_weights = g_grad.mean(dim=2, keepdim=True)
    g_cam = (g_weights * g_act).sum(dim=1)
    g_cam = torch.relu(g_cam)
    g_cam_norm = min_max_normalize(g_cam)
    glob_len = x_global_cpu.size(2) if x_global_cpu.dim() == 3 else x_global_cpu.size(1)
    g_cam_interp = interpolate_cam(g_cam_norm, glob_len)
    
    # 8. Compute Local Grad-CAM
    l_weights = l_grad.mean(dim=2, keepdim=True)
    l_cam = (l_weights * l_act).sum(dim=1)
    l_cam = torch.relu(l_cam)
    l_cam_norm = min_max_normalize(l_cam)
    loc_len = x_local_cpu.size(2) if x_local_cpu.dim() == 3 else x_local_cpu.size(1)
    l_cam_interp = interpolate_cam(l_cam_norm, loc_len)
    
    return g_cam_interp.numpy(), l_cam_interp.numpy()


def save_gradcam_figures(
    x_global: np.ndarray,
    x_local: np.ndarray,
    g_cam: np.ndarray,
    l_cam: np.ndarray,
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    config
) -> None:
    """
    Saves visual heatmaps, overlays, and publication-quality comparisons
    for each test sample in PNG and SVG formats, along with arrays and validation logs.
    """
    cmap_name = getattr(config, 'GRADCAM_COLORMAP', 'jet')
    dpi = getattr(config, 'GRADCAM_DPI', 300)
    save_arrays = getattr(config, 'SAVE_GRADCAM_ARRAYS', True)
    
    # Check bounds
    num_samples = len(y_true)
    logger.info(f"Generating Grad-CAM visualizations for {num_samples} samples...")
    
    for i in range(num_samples):
        prob = float(y_pred_probs[i])
        pred_label = 1 if prob >= 0.5 else 0
        true_label = int(y_true[i])
        
        glob_signal = x_global[i].flatten()
        loc_signal = x_local[i].flatten()
        
        glob_cam = g_cam[i].flatten()
        loc_cam = l_cam[i].flatten()
        
        # --- 1. Save Raw Global and Local Signals ---
        # Global
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(glob_signal, color='blue', label='Binned Flux')
        ax.set_title(f"Global View Raw Light Curve (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.GRADCAM_GLOBAL_DIR / f"raw_global_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_GLOBAL_DIR / f"raw_global_sample_{i}.svg")
        plt.close()
        
        # Local
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(loc_signal, color='blue', label='Binned Flux')
        ax.set_title(f"Local View Raw Light Curve (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.GRADCAM_LOCAL_DIR / f"raw_local_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_LOCAL_DIR / f"raw_local_sample_{i}.svg")
        plt.close()
        
        # --- 2. Save Grad-CAM Heatmaps ---
        # Global
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        im = ax.imshow(glob_cam.reshape(1, -1), aspect='auto', cmap=cmap_name, extent=[0, len(glob_signal), 0, 1])
        ax.set_yticks([])
        ax.set_title(f"Grad-CAM Heatmap - Global View (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
        plt.tight_layout()
        plt.savefig(config.GRADCAM_GLOBAL_DIR / f"heatmap_global_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_GLOBAL_DIR / f"heatmap_global_sample_{i}.svg")
        plt.close()
        
        # Local
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        im = ax.imshow(loc_cam.reshape(1, -1), aspect='auto', cmap=cmap_name, extent=[0, len(loc_signal), 0, 1])
        ax.set_yticks([])
        ax.set_title(f"Grad-CAM Heatmap - Local View (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
        plt.tight_layout()
        plt.savefig(config.GRADCAM_LOCAL_DIR / f"heatmap_local_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_LOCAL_DIR / f"heatmap_local_sample_{i}.svg")
        plt.close()
        
        # --- 3. Save Overlays ---
        # Global
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        # Plot curve
        ax.plot(glob_signal, color='black', alpha=0.3, label='Flux')
        # Use colored scatter plot to represent overlays
        sc = ax.scatter(np.arange(len(glob_signal)), glob_signal, c=glob_cam, cmap=cmap_name, s=4, label='Grad-CAM Attribution')
        ax.set_title(f"Grad-CAM Overlay - Global View (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Normalized Flux")
        fig.colorbar(sc, ax=ax, label="Attribution Intensity")
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.GRADCAM_OVERLAYS_DIR / f"overlay_global_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_OVERLAYS_DIR / f"overlay_global_sample_{i}.svg")
        plt.close()
        
        # Local
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(loc_signal, color='black', alpha=0.3, label='Flux')
        sc = ax.scatter(np.arange(len(loc_signal)), loc_signal, c=loc_cam, cmap=cmap_name, s=6, label='Grad-CAM Attribution')
        ax.set_title(f"Grad-CAM Overlay - Local View (Sample {i})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Normalized Flux")
        fig.colorbar(sc, ax=ax, label="Attribution Intensity")
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.GRADCAM_OVERLAYS_DIR / f"overlay_local_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_OVERLAYS_DIR / f"overlay_local_sample_{i}.svg")
        plt.close()
        
        # --- 4. Publication-Quality Comparison Figure ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi)
        fig.suptitle(
            f"StarSight CNN Grad-CAM Explanations (Sample {i})\n"
            f"True: {true_label} | Pred: {pred_label} | Prob: {prob:.4f}",
            fontsize=14, fontweight='bold'
        )
        
        # Global view plots (Row 0)
        axes[0, 0].plot(glob_signal, color='blue')
        axes[0, 0].set_title("Global View - Raw Light Curve")
        axes[0, 0].set_xlabel("Bin Index")
        axes[0, 0].set_ylabel("Normalized Flux")
        
        im_g = axes[0, 1].imshow(glob_cam.reshape(1, -1), aspect='auto', cmap=cmap_name, extent=[0, len(glob_signal), 0, 1])
        axes[0, 1].set_title("Global View - Grad-CAM Heatmap")
        axes[0, 1].set_yticks([])
        axes[0, 1].set_xlabel("Bin Index")
        fig.colorbar(im_g, ax=axes[0, 1], orientation='horizontal', pad=0.15)
        
        axes[0, 2].plot(glob_signal, color='black', alpha=0.3)
        sc_g = axes[0, 2].scatter(np.arange(len(glob_signal)), glob_signal, c=glob_cam, cmap=cmap_name, s=3)
        axes[0, 2].set_title("Global View - Attribution Overlay")
        axes[0, 2].set_xlabel("Bin Index")
        axes[0, 2].set_ylabel("Normalized Flux")
        fig.colorbar(sc_g, ax=axes[0, 2])
        
        # Local view plots (Row 1)
        axes[1, 0].plot(loc_signal, color='blue')
        axes[1, 0].set_title("Local View - Raw Light Curve")
        axes[1, 0].set_xlabel("Bin Index")
        axes[1, 0].set_ylabel("Normalized Flux")
        
        im_l = axes[1, 1].imshow(loc_cam.reshape(1, -1), aspect='auto', cmap=cmap_name, extent=[0, len(loc_signal), 0, 1])
        axes[1, 1].set_title("Local View - Grad-CAM Heatmap")
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xlabel("Bin Index")
        fig.colorbar(im_l, ax=axes[1, 1], orientation='horizontal', pad=0.15)
        
        axes[1, 2].plot(loc_signal, color='black', alpha=0.3)
        sc_l = axes[1, 2].scatter(np.arange(len(loc_signal)), loc_signal, c=loc_cam, cmap=cmap_name, s=5)
        axes[1, 2].set_title("Local View - Attribution Overlay")
        axes[1, 2].set_xlabel("Bin Index")
        axes[1, 2].set_ylabel("Normalized Flux")
        fig.colorbar(sc_l, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(config.GRADCAM_COMPARISON_DIR / f"comparison_grid_sample_{i}.png", dpi=dpi)
        plt.savefig(config.GRADCAM_COMPARISON_DIR / f"comparison_grid_sample_{i}.svg")
        plt.close()
        
        # --- 5. Save Raw Arrays & Metadata JSON ---
        if save_arrays:
            np.savez_compressed(
                config.GRADCAM_NUMPY_DIR / f"gradcam_arrays_sample_{i}.npz",
                glob_signal=glob_signal,
                loc_signal=loc_signal,
                glob_cam=glob_cam,
                loc_cam=loc_cam
            )
            
            metadata = {
                "sample_index": i,
                "prediction_probability": prob,
                "predicted_label": pred_label,
                "true_label": true_label,
                "global_cam_min": float(glob_cam.min()),
                "global_cam_max": float(glob_cam.max()),
                "global_cam_mean": float(glob_cam.mean()),
                "local_cam_min": float(loc_cam.min()),
                "local_cam_max": float(loc_cam.max()),
                "local_cam_mean": float(loc_cam.mean()),
            }
            with open(config.GRADCAM_NUMPY_DIR / f"metadata_sample_{i}.json", "w") as f:
                json.dump(metadata, f, indent=4)
                
    logger.info("Grad-CAM figures, arrays, and metadata saved successfully.")
