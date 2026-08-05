import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import hashlib
import io
import zipfile

st.set_page_config(page_title="Automatic Photo Sorter", page_icon="📸")

st.title("📸 Automatic Photo Sorter & Cleaner")
st.write("Dapat menghapus foto duplikat otomatis dan mengelompokkan foto berwajah ke dalam folder!")

# File model deteksi wajah dari repository GitHub
CASCADE_FILE = "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_detector():
    if os.path.exists(CASCADE_FILE):
        try:
            detector = cv2.CascadeClassifier(CASCADE_FILE)
            if not detector.empty():
                return detector
        except Exception:
            return None
    return None

detector = load_detector()

uploaded_files = st.file_uploader("Upload foto-foto kamu di sini", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Total file diunggah: {len(uploaded_files)} foto.")
    
    # Variabel pelacak duplikat dan penyimpan hasil
    seen_hashes = set()
    unique_files = []
    duplicate_count = 0
    
    # 1. PROSES DETEKSI DUPLIKAT
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        # Hitung hash MD5 dari isi file
        file_hash = hashlib.md5(bytes_data).hexdigest()
        
        if file_hash in seen_hashes:
            duplicate_count += 1
        else:
            seen_hashes.add(file_hash)
            # Simpan file unik kembali untuk diproses
            uploaded_file.seek(0)
            unique_files.append(uploaded_file)
            
    if duplicate_count > 0:
        st.warning(f"🧹 Berhasil mendeteksi dan mengabaikan **{duplicate_count} foto duplikat**!")
    else:
        st.success("✅ Tidak ditemukan foto duplikat.")
        
    st.write(f"Sisa foto bersih yang diproses: **{len(unique_files)} foto**.")
    st.divider()

    # 2. PROSES SORTIR WAJAH & PENYIAPAN ZIP
    zip_buffer = io.BytesIO()
    
    people_count = 0
    other_count = 0
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in unique_files:
            image_pil = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(image_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            faces = []
            if detector is not None:
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            # Tentukan folder tujuan di dalam file ZIP
            if len(faces) > 0:
                folder_target = "Foto_Orang"
                people_count += 1
                status_text = f"✅ Ada {len(faces)} wajah ➔ Dimasukkan ke `Foto_Orang/`"
            else:
                folder_target = "Foto_Lainnya"
                other_count += 1
                status_text = "📁 Tidak ada wajah ➔ Dimasukkan ke `Foto_Lainnya/`"
                
            # Simpan file ke dalam ZIP
            img_byte_arr = io.BytesIO()
            image_pil.save(img_byte_arr, format=image_pil.format if image_pil.format else 'JPEG')
            zip_file.writestr(f"{folder_target}/{uploaded_file.name}", img_byte_arr.getvalue())
            
            # Tampilkan Ringkasan di Layar
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_pil, width=150)
            with col2:
                st.subheader(uploaded_file.name)
                st.write(status_text)
            st.divider()

    # 3. TOMBOL DOWNLOAD HASIL SORTIRAN (ZIP)
    st.subheader("📦 Unduh Hasil Sortir & Bebas Duplikat")
    st.write(f"Ringkasan: **{people_count} Foto Orang** | **{other_count} Foto Lainnya** | **{duplicate_count} Duplikat Dibuang**")
    
    st.download_button(
        label="⬇️ Download File ZIP Foto Rapi",
        data=zip_buffer.getvalue(),
        file_name="Foto_Sudah_Dirapikan.zip",
        mime="application/zip"
    )
