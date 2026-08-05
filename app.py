import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import io
import zipfile
import hashlib

st.set_page_config(page_title="Automatic Photo Sorter Pro", page_icon="📸", layout="wide")

st.title("📸 Automatic Photo Sorter Pro")
st.write("Dapat mendeteksi duplikat visual, **mengelompokkan tiap wajah orang yang berbeda ke folder masing-masing**, dan mendukung upload folder/ZIP!")

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

# 1. Fungsi dHash untuk Hapus Duplikat Visual
def get_dhash(image, hash_size=8):
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff

def is_duplicate(hash1, hash2, threshold=5):
    return np.count_nonzero(hash1 != hash2) <= threshold

# 2. Fungsi Ekstraksi Fitur Wajah (Histogram) untuk Membedakan Tiap Orang
def extract_face_feature(gray_img, face_box):
    x, y, w, h = face_box
    face_roi = gray_img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (100, 100))
    # Hitung histogram gradien visual wajah
    hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def match_person(new_hist, person_histograms, threshold=0.55):
    # Bandingkan kemiripan struktur wajah dengan data orang yang sudah ada
    for idx, known_hist in enumerate(person_histograms):
        similarity = cv2.compareHist(new_hist, known_hist, cv2.HISTCMP_CORREL)
        if similarity >= threshold:
            return idx # Mengembalikan index orang yang cocok
    return -1 # Orang baru

# --- UI FITUR UPLOAD ---
tab1, tab2 = st.tabs(["📁 Upload File / Folder", "☁️ Informasi Integrasi Google Drive"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload foto-foto atau isi folder kamu di sini:", 
        type=['jpg', 'png', 'jpeg', 'zip'], 
        accept_multiple_files=True
    )

with tab2:
    st.info("""
    ### 💡 Cara Menggunakan File dari Google Drive:
    1. **Cara Termudah**: Di Google Drive, klik kanan folder foto kamu ➔ **Download** (Drive akan otomatis mengompresnya jadi file **ZIP**).
    2. Upload file **.zip** tersebut ke tab **"Upload File / Folder"** di atas! Aplikasi ini akan otomatis mengekstrak dan memproses semua foto di dalamnya.
    """)

if uploaded_files:
    # Mengumpulkan semua file foto (termasuk membongkar ZIP jika ada)
    all_images = []
    
    for uf in uploaded_files:
        if uf.name.endswith('.zip'):
            try:
                with zipfile.ZipFile(uf) as z:
                    for filename in z.namelist():
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('__MACOSX'):
                            img_data = z.read(filename)
                            all_images.append((os.path.basename(filename), img_data))
            except Exception as e:
                st.error(f"Gagal membaca file ZIP {uf.name}: {e}")
        else:
            all_images.append((uf.name, uf.read()))

    st.info(f"Total gambar ditemukan untuk diproses: **{len(all_images)} foto**.")
    
    # --- PROSES DETEKSI DUPLIKAT VISUAL ---
    unique_images = []
    unique_hashes = []
    duplicate_count = 0
    
    for filename, img_bytes in all_images:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            file_hash = get_dhash(pil_img)
            
            is_dup = False
            for existing_hash in unique_hashes:
                if is_duplicate(file_hash, existing_hash, threshold=5):
                    is_dup = True
                    break
            
            if is_dup:
                duplicate_count += 1
            else:
                unique_hashes.append(file_hash)
                unique_images.append((filename, pil_img))
        except Exception:
            continue
            
    if duplicate_count > 0:
        st.warning(f"🧹 Berhasil membuang **{duplicate_count} foto duplikat/mirip**!")
    else:
        st.success("✅ Tidak ditemukan foto duplikat.")

    st.write(f"Sisa foto unik diproses: **{len(unique_images)} foto**.")
    st.divider()

    # --- PROSES PENGELOMPOKAN TENTANG WAJAH MINGGUAN (PER ORANG) ---
    person_histograms = []
    zip_buffer = io.BytesIO()
    
    folder_summary = {} # Menyimpan statistik per folder

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, pil_img in unique_images:
            img_np = np.array(pil_img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            faces = []
            if detector is not None:
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            if len(faces) > 0:
                # Ambil wajah terbesar jika ada beberapa wajah dalam 1 foto
                largest_face = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
                face_hist = extract_face_feature(gray, largest_face)
                
                person_id = match_person(face_hist, person_histograms)
                if person_id == -1:
                    # Ditemukan orang baru!
                    person_histograms.append(face_hist)
                    person_id = len(person_histograms)
                else:
                    person_id += 1 # Index + 1 untuk penamaan folder
                
                folder_target = f"Foto_Orang_{person_id}"
                status_text = f"👤 Dideteksi: **Orang {person_id}** ➔ Dimasukkan ke `{folder_target}/`"
            else:
                folder_target = "Foto_Tanpa_Wajah"
                status_text = "📁 Tidak ada wajah ➔ Dimasukkan ke `Foto_Tanpa_Wajah/`"
                
            # Update ringkasan
            folder_summary[folder_target] = folder_summary.get(folder_target, 0) + 1
            
            # Simpan ke ZIP
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            zip_file.writestr(f"{folder_target}/{filename}", img_byte_arr.getvalue())
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(pil_img, width=120)
            with col2:
                st.write(f"**{filename}**")
                st.write(status_text)
            st.divider()

    # --- RINGKASAN HASIL DAN DOWNLOAD ---
    st.subheader("📦 Unduh Hasil Terkelompok Per Orang & Bebas Duplikat")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write("📊 **Rincian Folder Dibuat:**")
        for folder_name, count in folder_summary.items():
            st.write(f"- `{folder_name}/`: **{count} foto**")
    with col_stat2:
        st.write(f" Total Orang Berbeda Terdeteksi: **{len(person_histograms)} Orang**")
        st.write(f" Total Duplikat Dibuang: **{duplicate_count} Foto**")
        
    st.download_button(
        label="⬇️ Download File ZIP Hasil Sortir (Per Orang)",
        data=zip_buffer.getvalue(),
        file_name="Hasil_Sortir_Foto_Per_Orang.zip",
        mime="application/zip"
    )
