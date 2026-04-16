import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, datasets
from pathlib import Path

from model import ResNet18Classifier

st.title("Traffic Sign Demo")

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "cropped_belgiumts_classid" / "train"
MODEL_PATH = BASE_DIR / "outputs_ResNet18_augTrue" / "best_ResNet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    dataset = datasets.ImageFolder(root=str(DATA_ROOT))
    return dataset.classes


CLASS_NAMES = load_class_names()

CLASS_DISPLAY_NAMES = {name: name.replace("_", " ").title() for name in CLASS_NAMES}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@st.cache_resource
def load_model():
    model = ResNet18Classifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    class_id = CLASS_NAMES[pred_idx]
    display_name = CLASS_DISPLAY_NAMES.get(class_id, class_id)
    return class_id, display_name, confidence


model = load_model()

uploaded_file = st.file_uploader(
    "Upload a road image or cropped sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, caption="Uploaded image", use_container_width=True)

    st.markdown("### Step 1: Crop the traffic sign region")
    st.markdown("Adjust the sliders so that only the main traffic sign is inside the crop.")

    width, height = image.size

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.slider("Left (x1)", 0, width - 1, 0)
        x2 = st.slider("Right (x2)", x1 + 1, width, width)

    with col2:
        y1 = st.slider("Top (y1)", 0, height - 1, 0)
        y2 = st.slider("Bottom (y2)", y1 + 1, height, height)

    cropped = image.crop((x1, y1, x2, y2))

    st.subheader("Cropped Sign Region")
    st.image(cropped, caption="Cropped region for classification", width=220)

    if st.button("Predict"):
        class_id, display_name, confidence = predict_image(model, cropped)

        st.subheader("Prediction")
        st.write(f"Predicted label: **{display_name}**")
        st.write(f"Internal class id: `{class_id}`")
        st.write(f"Confidence: **{confidence:.4f}**")