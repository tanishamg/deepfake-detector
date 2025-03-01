import torch
import torchvision.models as models

# Load pretrained ResNeXt model
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# Save the model weights
torch.save(model.state_dict(), "resnext_model.pth")

print("✅ Model saved as resnext_model.pth")