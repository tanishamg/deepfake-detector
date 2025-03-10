import streamlit as st
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
from tqdm import tqdm

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

# Preprocess each frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)  # Add batch dimension

# Analyze video for deepfakes
def analyze_video(video_path):
    resnext_model = load_resnext()
    xception_model = load_xception()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    deepfake_frames, real_frames = 0, 0
    processed_frames = 0

    st.write(f"Processing video: {frame_count} frames...")

    progress_bar = st.progress(0)

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_frame(frame)

        with torch.no_grad():
            resnext_pred = resnext_model(image_tensor)
            xception_pred = xception_model(image_tensor)

        final_prediction = (resnext_pred + xception_pred) / 2
        final_label = torch.argmax(final_prediction, dim=1).item()

        if final_label == 1:
            deepfake_frames += 1
        else:
            real_frames += 1

        processed_frames += 1
        progress_bar.progress(int((processed_frames / frame_count) * 100))

    cap.release()

    if deepfake_frames > real_frames:
        return "🚨 Deepfake Video Detected!"
    else:
        return "✅ Real Video"

# Streamlit UI
st.title("🎭 Deepfake Video Detector")
st.write("Upload a video to check if it is **Real** or **Deepfake**.")

# Upload video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    if st.button("Analyze Video"):
        result = analyze_video(tfile.name)
        st.success(result)
