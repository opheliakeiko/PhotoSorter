import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import io
import zipfile

st.set_page_config(page_title="Automatic Photo Sorter Pro", page_icon="📸", layout="wide")

st.title("📸 Automatic Photo Sorter Pro")
st.write("Mendeteksi duplikat visual, memisahkan foto sendiri per orang, serta memisahkan foto bareng-bareng ke folder **Foto Rombongan**!")

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

# 1. dHash untuk Hapus Duplikat Visual
def get_dhash(image, hash_size=8):
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff

def is_duplicate(hash1, hash2, threshold=5):
    return np.count_nonzero(hash1 != hash2) <= threshold

# 2. Ekstraksi Fitur Wajah (Histogram)
def extract_face_feature(gray_img, face_box):
    x, y, w, h = face_box
    face_roi = gray_img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (100, 100))
    hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def match_person(new_hist, person_histograms, threshold=0.55):
    for idx, known_hist in enumerate(person_histograms):
        similarity = cv2.compareHist(new_hist, known_hist, cv2.HISTCMP_CORREL)
        if similarity >= threshold:
            return idx
    return -1

# --- TAB UPLOAD & DRIVE ---
tab1, tab2 = st.tabs(["📁 Upload File / Folder", "☁️ Info Google Drive"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload foto-foto atau file .ZIP kamu di sini:", 
        type=['jpg', 'png', 'jpeg', 'zip'], 
        accept_multiple_files=True
    )

with tab2:
    st.info("""
    ### 💡 Cara Mengunggah Folder dari Google Drive:
    1. Di Google Drive, klik kanan folder foto kamu ➔ Pilih **Download**.
    2. Google Drive akan otomatis mengompresnya menjadi file **.zip**.
    3. Upload file **.zip** tersebut langsung ke tab **"Upload File / Folder"** di atas!
    """)

if uploaded_files:
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

    st.info(f"Total foto ditemukan: **{len(all_images)} foto**.")
    
    # --- DETEKSI DUPLIKAT ---
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
        st.warning(f"🧹 Berhasil mendeteksi & membuang **{duplicate_count} foto duplikat/mirip**!")
    else:
        st.success("✅ Tidak ada foto duplikat.")

    st.write(f"Sisa foto unik diproses: **{len(unique_images)} foto**.")
    st.divider()

    # --- SORTIR FOTO (SENDIRI vs ROMBONGAN) ---
    person_histograms = []
    zip_buffer = io.BytesIO()
    folder_summary = {}

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, pil_img in unique_images:
            img_np = np.array(pil_img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            faces = []
            if detector is not None:
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            num_faces = len(faces)
            
            if num_faces > 1:
                # BANYAK ORANG -> FOTO ROMBONGAN
                folder_target = "Foto_Rombongan"
                status_text = f"👥 Dideteksi **{num_faces} Orang** ➔ Dimasukkan ke `{folder_target}/`"
            elif num_faces == 1:
                # CUMA 1 ORANG -> MASUK FOTO INDIVIDU
                face_hist = extract_face_feature(gray, faces[0])
                person_id = match_person(face_hist, person_histograms)
                
                if person_id == -1:
                    person_histograms.append(face_hist)
                    person_id = len(person_histograms)
                else:
                    person_id += 1
                
                folder_target = f"Foto_Orang_{person_id}"
                status_text = f"👤 Dideteksi: **Orang {person_id}** ➔ Dimasukkan ke `{folder_target}/`"
            else:
                # TANPA WAJAH
                folder_target = "Foto_Tanpa_Wajah"
                status_text = "📁 Tidak ada wajah ➔ Dimasukkan ke `Foto_Tanpa_Wajah/`"
                
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

    # --- RINGKASAN & DOWNLOAD ---
    st.subheader("📦 Unduh Hasil Sortir")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write("📊 **Rincian Folder:**")
        for folder_name, count in folder_summary.items():
            st.write(f"- `{folder_name}/`: **{count} foto**")
    with col_stat2:
        st.write(f" Total Orang Berbeda (Foto Sendiri): **{len(person_histograms)} Orang**")
        st.write(f" Total Duplikat Dibuang: **{duplicate_count} Foto**")
        
    st.download_button(
        label="⬇️ Download File ZIP Hasil Sortir",
        data=zip_buffer.getvalue(),
        file_name="Foto_Sudah_Dirapikan.zip",
        mime="application/zip"
    )
