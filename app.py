import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Photo Sorter", page_icon="📸")

st.title("📸 Web Rapikan & Sortir Foto Otomatis")
st.write("Upload foto-fotomu di bawah ini untuk dideteksi wajahnya!")

uploaded_files = st.file_uploader("Pilih foto-foto kamu", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Berhasil mengunggah {len(uploaded_files)} foto!")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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