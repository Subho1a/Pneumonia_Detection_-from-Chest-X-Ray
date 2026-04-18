import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import CNN
import os

# App configuration
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

@st.cache_resource
def load_model():
    model = CNN()
    model_path = os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}. Please train the model and save it to this path.")
        return None
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Ensure image is grayscale as expected by the model
    image = ImageOps.grayscale(image)
    img_tensor = transformation(image).unsqueeze(0)
    return img_tensor

def main():
    st.title("🫁 Chest X-Ray Pneumonia Detector")
    st.markdown("""
        Upload a chest X-Ray image to determine if there are signs of **Pneumonia**.
    """)

    model = load_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            # Show original uploaded image
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)

            # Classify automatically
            with st.spinner("Analyzing..."):
                img_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred = torch.argmax(probs).item()
                    
                    # Notebook class names
                    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
                    
                    if pred in class_names:
                        label = class_names[pred]
                        confidence = probs[0][pred].item() * 100
                        
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Prediction", value=label)
                        with col2:
                            st.metric(label="Confidence", value=f"{confidence:.2f}%")
                            
                        if label == 'PNEUMONIA':
                            st.warning("⚠️ **Warning:** The model detected signs of pneumonia. Please consult a healthcare professional for an accurate diagnosis.")
                        else:
                            st.info("✅ **Note:** The model did not detect obvious signs of pneumonia. If symptoms persist, please consult a doctor.")
                    else:
                        st.error(f"Unexpected prediction output class: {pred}. Please check model architecture vs dataset classes.")
                            
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")

if __name__ == "__main__":
    main()
