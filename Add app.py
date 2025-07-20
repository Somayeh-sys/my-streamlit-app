import streamlit as st
import gdown
import os
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="تشخیص دانه", layout="centered")
st.title("📸 مدل تشخیص دانه")

MODEL_PATH = "unet_resnet50_weights.h5"
MODEL_URL = "https://drive.google.com/uc?id=1ge1dbUZK-yT4KvkrDeo175APRBs3k4Jc"

# دانلود مدل اگر وجود نداشت
if not os.path.exists(MODEL_PATH):
    with st.spinner("در حال دانلود مدل..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# بارگذاری مدل
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

uploaded_file = st.file_uploader("یک تصویر آپلود کن", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)

    pred = model.predict(input_tensor)
    pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

    def mask_to_color(mask):
        COLORMAP = {
            0: (0, 0, 0),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (0, 0, 255),
        }
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in COLORMAP.items():
            color_mask[mask == cls_id] = color
        return color_mask

    colored_mask = mask_to_color(pred_mask)

    st.subheader("🔍 نتیجه مدل:")
    st.image(image, caption="تصویر ورودی", use_column_width=True)
    st.image(colored_mask, caption="ماسک پیش‌بینی‌شده", use_column_width=True)
