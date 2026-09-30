"""Small helper used by the ResNet-18 baseline notebook.

The original course folder referenced a local ``model.py`` that was no longer
present in the archived project. This reconstructed helper restores that
dependency without changing the notebook's model definition.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True):
    """Create a ResNet-18 classifier with a task-specific output layer."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
