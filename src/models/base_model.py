import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger("StarSight.BaseModel")

class StarSightBaseModel(nn.Module):
    """
    Base model class providing shared functionalities for checkpoint saving,
    loading, parameter counting, and summaries.
    """
    def __init__(self):
        super(StarSightBaseModel, self).__init__()

    def count_parameters(self) -> tuple:
        """
        Count the total and trainable parameters of the model.
        
        Returns:
            tuple: (total_parameters, trainable_parameters)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_summary_text(self, global_shape: tuple, local_shape: tuple, stellar_shape: tuple, device: torch.device) -> str:
        """
        Generates a standardized textual summary of model parameters, inputs, and device config.
        """
        total, trainable = self.count_parameters()
        summary_lines = [
            "=" * 60,
            "                    ASTRONET MODEL SUMMARY",
            "=" * 60,
            f"Device Configuration           : {device}",
            f"Global Input Feature Shape     : {global_shape}",
            f"Local Input Feature Shape      : {local_shape}",
            f"Stellar Metadata Feature Shape : {stellar_shape}",
            f"Output Activation Shape        : (Batch, 1) (Raw Logits)",
            f"Total Model Parameters         : {total:,}",
            f"Trainable Model Parameters     : {trainable:,}",
            "=" * 60
        ]
        return "\n".join(summary_lines)

    def print_summary(self, global_shape: tuple, local_shape: tuple, stellar_shape: tuple, device: torch.device) -> None:
        """
        Print a standardized summary of the model inputs, outputs, and parameters.
        """
        print(self.get_summary_text(global_shape, local_shape, stellar_shape, device))

    def save_checkpoint(self, path: Path) -> None:
        """
        Save the model weights state dictionary to the specified path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Successfully saved checkpoint to: {path.resolve()}")

    def load_checkpoint(self, path: Path, device: torch.device) -> None:
        """
        Load model weights state dictionary from the specified path.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        logger.info(f"Successfully loaded checkpoint from: {path.resolve()}")
