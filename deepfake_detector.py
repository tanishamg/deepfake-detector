import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image

# Load pretrained ResNeXt model
def load_resnext():
    model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Binary classification (Deepfake vs Real)
    model.load_state_dict(torch.load("resnext_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load pretrained Xception (EfficientNet) model
def load_xception():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("xception_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
def predict(image_path):
    resnext_model = load_resnext()
    xception_model = load_xception()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        resnext_pred = resnext_model(image_tensor)
        xception_pred = xception_model(image_tensor)

    final_prediction = (resnext_pred + xception_pred) / 2  # Ensemble averaging
    final_label = torch.argmax(final_prediction, dim=1).item()

    return "Deepfake" if final_label == 1 else "Real"

# Test with an image
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Change this to your test image
    prediction = predict(image_path)
    print(f"Prediction: {prediction}")
