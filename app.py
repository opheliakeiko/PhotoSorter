import streamlit as st
import cv2
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="Automatic Photo Sorter", page_icon="📸")

st.title("📸 Automatic Photo Sorter")
st.write("Recognizes people and duplicate photos, gone in one click!")

# Memastikan file cascade tersedia
CASCADE_PATH = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

@st.cache_resource
def get_face_cascade():
    if not os.path.exists(CASCADE_PATH):
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
    return cv2.CascadeClassifier(CASCADE_PATH)

try:
    face_cascade = get_face_cascade()
except Exception as e:
    st.error(f"Gagal memuat model deteksi wajah: {e}")
    face_cascade = None

uploaded_files = st.file_uploader("Upload your photos!", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Berhasil mengunggah {len(uploaded_files)} foto!")
    
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        if image is None:
            st.warning(f"File {uploaded_file.name} tidak dapat dibaca sebagai gambar.")
            continue
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        else:
            faces = []
        
        st.subheader(f"File: {uploaded_file.name}")
        if len(faces) > 0:
            st.write(f"✅ Ditemukan **{len(faces)} wajah**! (Kategori: Foto Orang)")
        else:
            st.write("📁 Tidak ada wajah. (Kategori: Gambar Umum/Sisa)")
            
        st.image(uploaded_file, width=300)
        st.divider()
