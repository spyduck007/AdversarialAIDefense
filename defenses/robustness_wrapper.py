import numpy as np
import torch
from typing import Tuple

from defenses.anomaly_detector import AnomalyDetector


class RobustnessWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        detector_path: str,
        device: torch.device = torch.device("cpu"),
        block_adversarial: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.block_adversarial = block_adversarial

        self.detector = AnomalyDetector()
        self.detector.load(detector_path)

    def inspect(self, x: np.ndarray) -> np.ndarray:
        return self.detector.predict(x)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flags = self.inspect(x)

        if self.block_adversarial and np.any(flags == 1):
            raise RuntimeError("Adversarial input detected. Prediction blocked.")

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            outputs = self.model(x_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()

        return preds, flags