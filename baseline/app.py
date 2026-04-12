import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

from model import SimpleCNN

st.title("Traffic Sign Demo")

CLASS_NAMES = [
    "class_118", "class_12", "class_124", "class_130", "class_149",
    "class_151", "class_153", "class_155", "class_194", "class_31",
    "class_34", "class_35", "class_41", "class_57", "class_59",
    "class_61", "class_65", "class_76", "class_80", "class_81",
    "class_82", "class_87", "class_94"
]

CLASS_DISPLAY_NAMES = {
    "class_118": "Traffic Sign 118",
    "class_12": "Traffic Sign 12",
    "class_124": "Traffic Sign 124",
    "class_130": "Traffic Sign 130",
    "class_149": "Traffic Sign 149",
    "class_151": "Traffic Sign 151",
    "class_153": "Traffic Sign 153",
    "class_155": "Traffic Sign 155",
    "class_194": "Traffic Sign 194",
    "class_31": "Traffic Sign 31",
    "class_34": "Traffic Sign 34",
    "class_35": "Traffic Sign 35",
    "class_41": "Traffic Sign 41",
    "class_57": "Traffic Sign 57",
    "class_59": "Traffic Sign 59",
    "class_61": "Traffic Sign 61",
    "class_65": "Traffic Sign 65",
    "class_76": "Traffic Sign 76",
    "class_80": "Traffic Sign 80",
    "class_81": "Traffic Sign 81",
    "class_82": "Traffic Sign 82",
    "class_87": "Traffic Sign 87",
    "class_94": "Traffic Sign 94",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("outputs_belgiumts_classid/best_simplecnn_belgiumts.pth", map_location=DEVICE)
    )
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

uploaded_file = st.file_uploader("Upload a road image or cropped sign image", type=["jpg", "jpeg", "png"])

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