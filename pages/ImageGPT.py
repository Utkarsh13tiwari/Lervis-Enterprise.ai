import streamlit as st
import requests
import io
from PIL import Image
import base64

API_URL = "https://api-inference.huggingface.co/models/ZB-Tech/Text-to-Image"
headers = {"Authorization": "Bearer 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"}

col1, col2, col3 = st.columns([1,2,1])
with col1:
        st.image(r'C:\Users\utkar\OneDrive\Desktop\Lervis Enterprise\transperent_logo.png', width=200)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def encode_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

with col2:
    st.title("Image and Text-based Design Generator")

    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

    user_prompt = st.text_area("Enter your text prompt", height=150)

    if st.button("Generate Image"):
        encoded_image = None
        if uploaded_image:
            encoded_image = encode_image_to_base64(uploaded_image)
        
        payload = {
            "inputs": user_prompt
        }
        
        if encoded_image:
            payload["image"] = encoded_image

        image_bytes = query(payload)

        try:
            generated_image = Image.open(io.BytesIO(image_bytes))
            st.image(generated_image, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
