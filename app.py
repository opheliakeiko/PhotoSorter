import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import io
import zipfile

st.set_page_config(page_title="Automatic Photo Sorter", page_icon="📸")

st.title("📸 Automatic Photo Sorter & Duplicate Cleaner")
st.write("Mendeteksi foto duplikat (secara visual) dan mengelompokkan foto berwajah secara otomatis!")

# File model deteksi wajah
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

# Fungsi untuk menghitung Difference Hash (dHash) visual foto
def get_dhash(image, hash_size=8):
    # Resize ke (hash_size + 1, hash_size) dan ubah ke grayscale
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    # Bandingkan pixel bersebelahan
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff

# Fungsi menghitung jarak perbedaan antara 2 hash (Hamming Distance)
def is_duplicate(hash1, hash2, threshold=5):
    # Jika perbedaan nilai pixel <= threshold, dianggap foto duplikat
    return np.count_nonzero(hash1 != hash2) <= threshold

uploaded_files = st.file_uploader("Upload foto-foto kamu di sini", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Total file diunggah: {len(uploaded_files)} foto.")
    
    unique_files = []
    unique_hashes = []
    duplicate_count = 0
    
    # Sensitivitas kemiripan (makin kecil makin ketat, default 5 cocok untuk duplikat visual)
    THRESHOLD = 5
    
    # 1. PROSES DETEKSI DUPLIKAT VISUAL
    for uploaded_file in uploaded_files:
        try:
            image_pil = Image.open(uploaded_file).convert('RGB')
            file_hash = get_dhash(image_pil)
            
            # Cek apakah hash visual foto ini mirip dengan foto yang sudah ada
            duplicate_found = False
            for existing_hash in unique_hashes:
                if is_duplicate(file_hash, existing_hash, threshold=THRESHOLD):
                    duplicate_found = True
                    break
            
            if duplicate_found:
                duplicate_count += 1
            else:
                unique_hashes.append(file_hash)
                uploaded_file.seek(0)
                unique_files.append(uploaded_file)
        except Exception:
            # Jika file rusak/gagal dibaca
            continue
            
    if duplicate_count > 0:
        st.warning(f"🧹 Berhasil mendeteksi dan membuang **{duplicate_count} foto duplikat/mirip** secara visual!")
    else:
        st.success("✅ Tidak ditemukan foto duplikat.")
        
    st.write(f"Sisa foto unik yang diproses: **{len(unique_files)} foto**.")
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
            
            if len(faces) > 0:
                folder_target = "Foto_Orang"
                people_count += 1
                status_text = f"✅ Ada {len(faces)} wajah ➔ Dimasukkan ke `Foto_Orang/`"
            else:
                folder_target = "Foto_Lainnya"
                other_count += 1
                status_text = "📁 Tidak ada wajah ➔ Dimasukkan ke `Foto_Lainnya/`"
                
            img_byte_arr = io.BytesIO()
            image_pil.save(img_byte_arr, format='JPEG')
            zip_file.writestr(f"{folder_target}/{uploaded_file.name}", img_byte_arr.getvalue())
            
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
