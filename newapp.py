import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pickle

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistem Peringatan Dini TB Lampung",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan lebih profesional
st.markdown("""
    <style>
    .main-header {
        font-size: 26px; font-weight: bold; color: #1E3A8A;
        text-align: center; margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6; padding: 15px; border-radius: 10px;
        border-left: 5px solid #1E3A8A; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & DATA (CACHED)
# ==========================================
@st.cache_resource
def load_trained_model():
    # Ganti 'model_terbaik.pkl' sesuai nama file model terbaik milikmu
    with open('model_terbaik.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_spatial_data():
    # Load GeoJSON / Shapefile peta Lampung
    gdf = gpd.read_file('lampung_kabupaten.geojson')
    return gdf

# Memanggil fungsi load
try:
    model = load_trained_model()
    gdf_lampung = load_spatial_data()
except Exception as e:
    st.warning("⚠️ File model atau dataset geospatial belum ditemukan. Pastikan file `.pkl` dan `.geojson` berada di folder yang sama.")

# ==========================================
# 3. SIDEBAR & NAVIGASI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bb/Lambang_Kabupaten_Lampung_Selatan.png", width=100) # Bisa diganti logo Lampung / Kampus
st.sidebar.title("Navigasi Utama")
menu = st.sidebar.radio("Pilih Tampilan:", ["Peta Prediksi Spasial", "Detail per Kabupaten/Kota", "Rekomendasi Logistik"])

st.sidebar.markdown("---")
st.sidebar.info("**Sistem Pendukung Keputusan (DSS)**\nDiprediksi menggunakan Model Machine Learning Terpilih.")

# ==========================================
# 4. TAMPILAN 1: PETA PREDIKSI SPASIAL
# ==========================================
if menu == "Peta Prediksi Spasial":
    st.markdown('<div class="main-header">PETA PREDIKSI RISIKO INSIDEN TUBERKULOSIS</div>', unsafe_allow_html=True)
    st.write("Sistem menampilkan pemetaan estimasi jumlah kasus Tuberkulosis untuk periode bulan mendatang di seluruh Wilayah Provinsi Lampung.")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Contoh pembuatan peta choropleth menggunakan Folium
        m = folium.Map(location=[-5.0, 105.0], zoom_start=8, tiles="CartoDB positron")
        
        # Integrasi data GeoPandas ke Peta (asumsi kolom 'prediksi_kasus' sudah dihitung)
        # folium.Choropleth(
        #     geo_data=gdf_lampung,
        #     name="choropleth",
        #     data=gdf_lampung,
        #     columns=["kabupaten", "prediksi_kasus"],
        #     key_on="feature.properties.kabupaten",
        #     fill_color="YlOrRd",
        #     fill_opacity=0.7,
        #     line_opacity=0.2,
        #     legend_name="Estimasi Kasus TB"
        # ).add_to(m)

        st_folium(m, width="100%", height=500)

    with col2:
        st.subheader("Ringkasan Eksekutif")
        st.markdown("""
        <div class="metric-card">
            <h4>Total Prediksi Provinsi</h4>
            <h2 style="color:#1E3A8A;">1,240 Kasus</h2>
        </div>
        <div class="metric-card">
            <h4>Kabupaten Risiko Tinggi</h4>
            <h3 style="color:#DC2626;">Bandar Lampung</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("ℹ️ Daerah dengan warna merah pekat menandakan proyeksi lonjakan kasus tertinggi yang membutuhkan perhatian prioritas.")

# ==========================================
# 5. TAMPILAN 2: DETAIL PER KABUPATEN
# ==========================================
elif menu == "Detail per Kabupaten/Kota":
    st.markdown('<div class="main-header">DETAIL PROYEKSI KASUS KABUPATEN / KOTA</div>', unsafe_allow_html=True)
    
    kab_selected = st.selectbox("Pilih Kabupaten / Kota:", [
        "Kota Bandar Lampung", "Kota Metro", "Kabupaten Lampung Selatan", 
        "Kabupaten Lampung Tengah", "Kabupaten Lampung Utara"
    ])

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label=f"Estimasi Kasus TB di {kab_selected} (Bulan Depan)", value="145 Kasus", delta="+12 Kasus dari Bulan Lalu")
        st.write("**Tren Historis & Prediksi:**")
        
        # Contoh Plot Chart Sederhana
        chart_data = pd.DataFrame({
            'Bulan': ['Bulan-3', 'Bulan-2', 'Bulan-1', 'Prediksi Depan'],
            'Jumlah Kasus': [120, 135, 133, 145]
        })
        st.line_chart(chart_data.set_index('Bulan'))

    with col_b:
        st.subheader("Faktor Pendorong Utama")
        st.write("Variabel dominan yang mempengaruhi estimasi kasus di wilayah ini:")
        
        # Menampilkan indikator tanpa menyebutkan istilah teknik yang rumit
        st.progress(0.85, text="Riwayat Kasus Bulan Lalu (Sangat Tinggi)")
        st.progress(0.60, text="Riwayat Kasus TB-DM (Sedang)")
        st.progress(0.40, text="Kepadatan Penduduk (Moderat)")

# ==========================================
# 6. TAMPILAN 3: REKOMENDASI LOGISTIK
# ==========================================
elif menu == "Rekomendasi Logistik":
    st.markdown('<div class="main-header">REKOMENDASI OPERASIONAL LOGISTIK OBAT</div>', unsafe_allow_html=True)
    st.write("Fitur ini memberikan saran tindakan otomatis berdasarkan hasil estimasi kasus untuk mencegah kelangkaan obat TB.")

    # Contoh Tabel Rekomendasi
    data_rekomendasi = {
        "Kabupaten / Kota": ["Bandar Lampung", "Lampung Selatan", "Lampung Tengah", "Metro"],
        "Estimasi Kasus": [145, 110, 98, 42],
        "Status Cadangan": ["⚠️ Perlu Tambahan", "✅ Aman", "✅ Aman", "✅ Aman"],
        "Rekomendasi Alokasi OAT": ["+150 Paket OAT Kategori 1", "Sesuai Kuota Rutin", "Sesuai Kuota Rutin", "Sesuai Kuota Rutin"]
    }
    df_rek = pd.DataFrame(data_rekomendasi)
    
    st.dataframe(df_rek, use_container_width=True)

    st.success("💡 **Saran Tindakan:** Prioritaskan alokasi cadangan (*buffer stock*) Obat Anti Tuberkulosis (OAT) ke wilayah dengan Status 'Perlu Tambahan' sebelum awal bulan berjalan.")
