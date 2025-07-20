%%writefile app.py
import streamlit as st

st.set_page_config(page_title="تشخیص دانه", layout="centered")
st.title("📸 مدل تشخیص دانه")

uploaded_file = st.file_uploader("یک تصویر آپلود کن", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import io

    # بارگذاری مدل
    model = tf.keras.models.load_model("/content/drive/MyDrive/unet_resnet50_weights.h5", compile=False)

    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)

    # پیش‌بینی
    pred = model.predict(input_tensor)
    pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

    # تعریف رنگ‌ها
    def mask_to_color(mask):
        COLORMAP = {
            0: (0, 0, 0),       # مشکی
            1: (0, 255, 0),     # سبز
            2: (255, 0, 0),     # قرمز
            3: (0, 0, 255),     # آبی
        }
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in COLORMAP.items():
            color_mask[mask == cls_id] = color
        return color_mask

    colored_mask = mask_to_color(pred_mask)

    # نمایش تصویر اصلی و خروجی مدل
    st.subheader("🔍 نتیجه مدل:")
    st.image(image, caption="تصویر ورودی", use_column_width=True)
    st.image(colored_mask, caption="ماسک پیش‌بینی‌شده", use_column_width=True)
