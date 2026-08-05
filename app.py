import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(page_title="Automatic Photo Sorter", page_icon="📸")

st.title("📸 Automatic Photo Sorter")
st.write("Recognizes people and duplicate photos, gone in one click!")

# Menggunakan CascadeClassifier bawaan opencv secara aman
CASCADE_FILE = "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_detector():
    if os.path.exists(CASCADE_FILE):
        detector = cv2.CascadeClassifier(CASCADE_FILE)
        if not detector.empty():
            return detector
    return None

detector = load_detector()

if detector is None:
    st.warning("⚠️ Model deteksi wajah (.xml) sedang dimuat/tidak ditemukan di folder utama. Pastikan file 'haarcascade_frontalface_default.xml' sudah tersimpan di GitHub.")

uploaded_files = st.file_uploader("Upload your photos!", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Berhasil mengunggah {len(uploaded_files)} foto!")
    
    for uploaded_file in uploaded_files:
        # Konversi file upload Streamlit ke format yang dibaca OpenCV
        image_pil = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        faces = []
        if detector is not None and not detector.empty():
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        st.subheader(f"File: {uploaded_file.name}")
        if len(faces) > 0:
            st.write(f"✅ Ditemukan **{len(faces)} wajah**! (Kategori: Foto Orang)")
        else:
            st.write("📁 Tidak ada wajah terdeteksi. (Kategori: Gambar Umum/Sisa)")
            
        st.image(image_pil, width=300)
        st.divider()
