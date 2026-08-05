import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os

st.set_page_config(page_title="Photo Sorter", page_icon="📸")

st.title("📸 Automatic Photo Sorter")
st.write("Recognizes people and duplicate photos, gone in one click!")
st.write("Upload your photos!")

# Memuat file cascade yang ada di folder repository
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):
    st.error("File haarcascade_frontalface_default.xml tidak ditemukan di repository GitHub!")
else:
    face_cascade = cv2.CascadeClassifier(cascade_path)

    uploaded_files = st.file_uploader("Pilih foto-foto kamu", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"Berhasil mengunggah {len(uploaded_files)} foto!")
        
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            st.subheader(f"File: {uploaded_file.name}")
            if len(faces) > 0:
                st.write(f"✅ Ditemukan **{len(faces)} wajah**! (Kategori: Foto Orang)")
            else:
                st.write("📁 Tidak ada wajah. (Kategori: Gambar Umum/Sisa)")
                
            st.image(uploaded_file, width=300)
            st.divider()
