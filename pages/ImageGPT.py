import streamlit as st
import requests
import io
from PIL import Image
import base64
from streamlit_pills import pills
import os
from groq import Groq


st.set_page_config(
    page_title="Lervis Enterprise",
    layout="wide",
    page_icon = 'transperent_logo.png'
)  


groq_tcw = st.secrets.db_credentials.groq_tcw
huggingfc_token = st.secrets.db_credentials.huggingfc_token

client = Groq(api_key = groq_tcw)


REFINER_xl_API_URL = "https://api-inference.huggingface.co/models/ZB-Tech/Text-to-Image"

REFINER_1_5_API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

REFINER_2_1_API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

REFINER_1_4_API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

headers = {"Authorization": huggingfc_token}

col1, col2, col3 = st.columns([1,2,1])
with col1:
        st.image('transperent_logo.png', width=200)

def query(payload, api_url):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.content

def encode_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

def stream_parser(stream):
    for chunk in stream:
        yield chunk

def llava_model(client, model, base64_image, prompt):
    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=1,
            model=model,
            max_tokens=1024,
            top_p=1,
            stream=False,
            api_key = groq_tcw,
            stop=None,
        )

        print(completion.choices[0].message)
        return  completion.choices[0].message


    except Exception as e:
        print(f"An error occurred: {e}")
        return None



with col2:
    st.title("Image and Text-based Design Generator")

    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

    prompt_sel = pills("Prompt Engineering Models",["Llava", "Llama 3.2 vision"],["🍀", "🍀"])
    st.subheader("Do Prompt Engineering before you query the Stable Diffusion model: ")

    query_prompt = """Please suggest me the best propmt that I can feed into the stable diffusion model for generating the image provided. 
    
    Analysie the UI of the image properly and respond"""

    if  uploaded_image is not None:

        if prompt_sel == 'Llava':
            model = "llava-v1.5-7b-4096-preview"
            MLLM_reponse = llava_model(client, model=model, base64_image = encode_image_to_base64(uploaded_image), prompt=query_prompt)
        
        else:
            model = "llama-3.2-11b-vision-preview"
            MLLM_reponse = llava_model(client, model=model, base64_image = encode_image_to_base64(uploaded_image), prompt=query_prompt)

        st.write(MLLM_reponse.content)

    st.divider()
    selected = pills("Stable diffusion Models", ["Stable Diffusion xl", "Stable Diffusion v1.5", "Stable Diffusion v2.1", "Flax Stable Diffusion v1.4"], ["🎈", "🎈", "🎈", "🎈"])

    user_prompt = st.text_area("Enter your text prompt", height=150)

    if st.button("Generate Image"):
        with st.status(":red[Processing image file. DON'T LEAVE THIS PAGE WHILE IMAGE FILE IS BEING ANALYZED...]", expanded=True) as status:
            st.write(":orange[Analyzing Image File...]")

        encoded_image = None
        if uploaded_image:
            encoded_image = encode_image_to_base64(uploaded_image)
        
        payload = {
            "inputs": user_prompt
        }
        
        if encoded_image:
            payload["image"] = encoded_image

        if selected == 'Stable Diffusion xl':
            image_bytes = query(payload, REFINER_xl_API_URL)

        if selected == 'Stable Diffusion v1.5':
            image_bytes = query(payload, REFINER_1_5_API_URL)
        
        if selected == 'Stable Diffusion v2.1':
            image_bytes = query(payload, REFINER_2_1_API_URL)

        if selected == 'Flax Stable Diffusion v1.4':
            image_bytes = query(payload, REFINER_1_4_API_URL)


        if image_bytes:
            try:
                print("True")
                generated_image = Image.open(io.BytesIO(image_bytes))
                st.image(generated_image, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {e}")
                st.write("Raw image bytes (first 100 bytes):", image_bytes[:100] if isinstance(image_bytes, bytes) else "Not bytes")
