# ============================================================
# APLIKASI KEMATANGAN BUAH
# Deteksi Tingkat Kematangan Pisang & Tomat Berdasarkan Warna
# ============================================================
# Author : Deny Fajar Nugraha
# Institusi: Universitas Tidar
# ============================================================

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Aplikasi Kematangan Buah - Untidar",
    page_icon="logo_untidar.png",
    layout="wide"
)

# ============================================================
# HEADER DENGAN LOGO UNTIDAR
# ============================================================
col1, col2 = st.columns([1, 8])
with col1:
    logo_path = "logo_untidar.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.markdown("🖼️")
with col2:
    st.title("Aplikasi Analisis Kematangan Buah")
    st.markdown("**Universitas Tidar - Teknik Elektro**")
    st.markdown("Deteksi tingkat kematangan pisang dan tomat berdasarkan warna kulit.")

st.markdown("---")

# ============================================================
# FUNGSI UTAMA ANALISIS KEMATANGAN
# ============================================================
def analisis_kematangan(citra_bgr, jenis_buah="Pisang"):
    hsv = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2HSV)

    if jenis_buah == "Pisang":
        rentang = {
            "Hijau (Mentah)":   ([36, 25, 25],   [70, 255, 255]),
            "Kuning (Matang)":  ([20, 100, 100], [30, 255, 255]),
            "Coklat (Terlalu Matang)": ([10, 50, 20], [20, 255, 200])
        }
        label_map = {
            "Hijau (Mentah)": "Mentah",
            "Kuning (Matang)": "Matang",
            "Coklat (Terlalu Matang)": "Terlalu Matang"
        }
    else:  # Tomat
        rentang = {
            "Hijau (Mentah)":       ([40, 40, 40],   [80, 255, 255]),
            "Oranye (Setengah Matang)": ([10, 100, 100], [20, 255, 255]),
            "Merah (Matang)":       ([0, 100, 100],  [10, 255, 255])
        }
        label_map = {
            "Hijau (Mentah)": "Mentah",
            "Oranye (Setengah Matang)": "Setengah Matang",
            "Merah (Matang)": "Matang"
        }

    total_piksel = citra_bgr.shape[0] * citra_bgr.shape[1]
    hasil_persen = {}
    masker_dict = {}

    for nama, (bawah, atas) in rentang.items():
        lower = np.array(bawah, dtype=np.uint8)
        upper = np.array(atas, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        masker_dict[nama] = mask
        jumlah = np.sum(mask == 255)
        persen = (jumlah / total_piksel) * 100
        hasil_persen[nama] = persen

    dominan = max(hasil_persen, key=hasil_persen.get)
    label = label_map[dominan]

    return masker_dict, hasil_persen, label


def plot_histogram_garis(citra_bgr, title="Histogram Intensitas"):
    """Histogram garis untuk citra grayscale."""
    gray = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, color='blue')
    ax.set_xlim([0, 256])
    ax.set_xlabel('Intensitas')
    ax.set_ylabel('Frekuensi')
    ax.set_title(title)
    return fig


def plot_batang_persen(hasil_persen):
    """Diagram batang persentase area warna."""
    fig, ax = plt.subplots(figsize=(5, 4))
    nama = list(hasil_persen.keys())
    nilai = list(hasil_persen.values())
    warna_bar = []
    for n in nama:
        if 'Hijau' in n:
            warna_bar.append('green')
        elif 'Kuning' in n or 'Oranye' in n:
            warna_bar.append('gold')
        elif 'Coklat' in n:
            warna_bar.append('saddlebrown')
        elif 'Merah' in n:
            warna_bar.append('red')
        else:
            warna_bar.append('gray')

    bars = ax.bar(nama, nilai, color=warna_bar, alpha=0.7)
    ax.set_ylabel('Persentase Area (%)')
    ax.set_title('Distribusi Warna Kulit Buah')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, nilai):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📤 Unggah Foto Buah")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file gambar (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)
st.sidebar.markdown("---")
st.sidebar.header("Pilih Jenis Buah")
jenis_buah = st.sidebar.selectbox("Jenis Buah:", ["Pisang", "Tomat"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Petunjuk
- Gunakan foto dengan pencahayaan cukup.
- Latar belakang sebaiknya polos.
- **Pisang:** Hijau = Mentah, Kuning = Matang, Coklat = Terlalu Matang.
- **Tomat:** Hijau = Mentah, Oranye = Setengah Matang, Merah = Matang.
""")

# ============================================================
# AREA UTAMA
# ============================================================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masker, persen, label = analisis_kematangan(img_bgr, jenis_buah)

    # 1. CITRA ASLI + HISTOGRAM GARIS
    st.subheader("Citra Asli dan Histogram Intensitas")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_rgb, caption="Citra Asli", use_container_width=True)
    with col2:
        fig_hist_asli = plot_histogram_garis(img_bgr, "Histogram Intensitas (Grayscale)")
        st.pyplot(fig_hist_asli)

    st.markdown("---")

    # 2. HASIL SEGMENTASI (MASKER)
    st.subheader("Hasil Segmentasi Warna (Masker)")
    cols = st.columns(len(masker))
    for i, (nama, mask) in enumerate(masker.items()):
        with cols[i]:
            st.image(mask, caption=f"{nama}\n({persen[nama]:.1f}%)", use_container_width=True, clamp=True)

    st.markdown("---")

    # 3 & 4. HISTOGRAM HASIL? (di sini saya tampilkan diagram batang + kesimpulan)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Histogram Distribusi Warna")
        fig_batang = plot_batang_persen(persen)
        st.pyplot(fig_batang)

    with col_right:
        st.subheader("Kesimpulan Kematangan")
        # Warna latar sesuai label
        if label == "Mentah":
            bg_color = "#2e7d32"  # hijau tua
        elif label == "Matang":
            bg_color = "#f9a825"  # kuning emas
        elif label == "Setengah Matang":
            bg_color = "#ff9800"  # oranye
        else:
            bg_color = "#795548"  # coklat

        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; text-align: center;">
            <h2 style="color: white; margin: 0;">{label}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###  Detail Persentase:")
        for nama, val in persen.items():
            st.write(f"- **{nama}**: {val:.2f}%")

else:
    st.info("Silakan unggah foto buah terlebih dahulu melalui sidebar.")
    st.markdown("### Contoh Hasil Analisis:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Mentah")
        st.markdown("Warna hijau mendominasi (>60%)")
    with col2:
        st.markdown("#### Matang / Setengah Matang")
        st.markdown("Warna kuning/oranye mendominasi")
    with col3:
        st.markdown("#### Terlalu Matang")
        st.markdown("Warna coklat/merah tua mendominasi")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Aplikasi Kematangan Buah | Dibangun dengan Streamlit & OpenCV<br>"
    "© 2026 Deny Fajar Nugraha - Teknik Elektro Universitas Tidar"
    "</div>",
    unsafe_allow_html=True
)