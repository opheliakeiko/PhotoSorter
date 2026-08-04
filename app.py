import streamlit as st
from PIL import Image
import cv2
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="Photo Sorter", page_icon="📸")

st.title("📸 Web Rapikan & Sortir Foto Otomatis")
st.write("Upload foto-fotomu di bawah ini untuk dideteksi wajahnya!")

# Fungsi untuk mengunduh model deteksi wajah jika belum ada di server
@st.cache_resource
def load_face_cascade():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
    return cv2.CascadeClassifier(cascade_path)

uploaded_files = st.file_uploader("Pilih foto-foto kamu", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Berhasil mengunggah {len(uploaded_files)} foto!")
    
    # Memuat detektor wajah
    face_cascade = load_face_cascade()
    
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        st.subheader(f"File: {uploaded_file.name}")
        if len(faces) > 0:
            st.write(f"✅ Ditemukan **{len(faces)} wajah**! (Kategori: Foto Orang)")
        else:
            st.write("📁 Tidak ada wajah. (Kategori: Gambar Umum/Sisa)")
            
        st.image(uploaded_file, width=300)
        st.divider()