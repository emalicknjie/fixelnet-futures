"""FixelNet Conv2d model definition, loading, and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FixelNet(nn.Module):
    """
    Convolutional network for fixel-based trading signal classification.
    Input: (batch, 1, 6, 15) tensor — one channel, 6 indicator rows, 15 time cols.
    Output: log-softmax over 3 classes (Long=0, Heavy Short=1, Short=2).
    """

    def __init__(
        self,
        cv1_in: int, cv1_out: int,
        cv2_in: int, cv2_out: int,
        fc1_in: int, fc1_out: int,
        fc2_in: int, fc2_out: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cv1_in, cv1_out, kernel_size=1)
        self.conv2 = nn.Conv2d(cv2_in, cv2_out, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self._fc1_in = fc1_in

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self._fc1_in)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_model(model_dir: str | Path) -> FixelNet:
    """
    Load a pre-trained FixelNet from model_dir.

    Steps:
    1. Load state dict from test_model.pth.
    2. Derive architecture hyperparameters directly from the state dict shapes.
    3. Instantiate FixelNet and load the state dict.
    """
    model_dir = Path(model_dir)
    weights_path = model_dir / 'test_model.pth'
    if not weights_path.exists():
        raise FileNotFoundError(f'test_model.pth not found in {model_dir}')

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Derive architecture from weight tensor shapes — no separate config file needed
    cv1_in, cv1_out = int(state_dict['conv1.weight'].shape[1]), int(state_dict['conv1.weight'].shape[0])
    cv2_in, cv2_out = int(state_dict['conv2.weight'].shape[1]), int(state_dict['conv2.weight'].shape[0])
    fc1_in, fc1_out = int(state_dict['fc1.weight'].shape[1]), int(state_dict['fc1.weight'].shape[0])
    fc2_in, fc2_out = int(state_dict['fc2.weight'].shape[1]), int(state_dict['fc2.weight'].shape[0])

    model = FixelNet(
        cv1_in=cv1_in, cv1_out=cv1_out,
        cv2_in=cv2_in, cv2_out=cv2_out,
        fc1_in=fc1_in, fc1_out=fc1_out,
        fc2_in=fc2_in, fc2_out=fc2_out,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: FixelNet, fixel: np.ndarray) -> tuple[int, float, float, float]:
    """
    Run inference on a single (6, 15) fixel.

    Returns:
        predicted_class: argmax class index (0=Long, 1=Heavy Short, 2=Short)
        long_score: softmax probability for Long
        heavy_short_score: softmax probability for Heavy Short
        short_score: softmax probability for Short
    """
    t = Tensor(fixel).unsqueeze(0).unsqueeze(0)  # → (1, 1, 6, 15)
    with torch.no_grad():
        logits = model(t)
    probs = Tensor.softmax(logits, dim=1).detach().numpy()[0]
    return int(np.argmax(probs)), float(probs[0]), float(probs[1]), float(probs[2])
