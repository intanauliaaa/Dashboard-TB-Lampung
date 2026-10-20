import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import warnings
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Dashboard Prediksi Tuberkulosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D9488 100%);
    padding: 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; color: white;
}
.metric-card {
    background: white; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 1.2rem;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 0.5rem;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #1E3A5F; }
.metric-label { font-size: 0.82rem; color: #64748B; margin-top: 0.3rem; }
.section-title {
    font-size: 1.2rem; font-weight: 700; color: #1E3A5F;
    border-left: 4px solid #0D9488; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
.info-box { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:1rem; margin:1rem 0; }
.success-box { background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px; padding:1rem; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA
# ============================================================
KOLOM_TARGET  = ['JML_KSO', 'JML_KRO', 'JML_TBHIV', 'JML_TBDM']
TARGET_LABELS = {
    'JML_KSO'  : 'TB Sensitif Obat (TB-SO)',
    'JML_KRO'  : 'TB Resisten Obat (TB-RO)',
    'JML_TBHIV': 'TB-HIV',
    'JML_TBDM' : 'TB-DM',
}
KOLOM_TANGGAL = 'TANGGAL'
TANGGAL_SPLIT = '2025-01-01'
LAG_PERIODS   = [1, 2, 3]

KOORDINAT = {
    'Bandar Lampung'      : (-5.3971, 105.2668), 'Lampung Barat'       : (-5.0232, 104.0732),
    'Lampung Selatan'     : (-5.5981, 105.5098), 'Lampung Tengah'      : (-4.8357, 105.3036),
    'Lampung Timur'       : (-5.1025, 105.7870), 'Lampung Utara'       : (-4.5833, 104.9167),
    'Mesuji'              : (-3.9833, 105.7500), 'Metro'               : (-5.1131, 105.3067),
    'Pesawaran'           : (-5.4167, 105.1833), 'Pesisir Barat'       : (-4.8667, 103.9667),
    'Pringsewu'           : (-5.3581, 104.9736), 'Tanggamus'           : (-5.3833, 104.6333),
    'Tulang Bawang'       : (-4.1000, 105.6833), 'Tulang Bawang Barat' : (-4.3667, 105.1833),
    'Way Kanan'           : (-4.3333, 104.5667),
}

REKOMENDASI = {
    'JML_SPS'    : "📋 Tingkatkan kapasitas skrining TB — jumlah suspek tinggi.",
    'JML_SPL'    : "🏥 Perkuat layanan TB di faskes tingkat lanjut.",
    'JML_PENDUDUK': "👥 Fokuskan program TB pada wilayah padat penduduk.",
    'DEFAULT'    : "📊 Lakukan evaluasi menyeluruh program TB di wilayah ini.",
}

# ============================================================
# FUNGSI DATA
# ============================================================
def load_semua_data(uploaded_files):
    data_kabupaten = {}
    for f in uploaded_files:
        nama_murni = os.path.splitext(f.name)[0]
        nama_kab   = nama_murni.replace("Data Used - ", "").replace("_", " ")
        df = pd.read_csv(f)
        if KOLOM_TANGGAL in df.columns:
            df[KOLOM_TANGGAL] = pd.to_datetime(df[KOLOM_TANGGAL])
        data_kabupaten[nama_kab] = df
    return data_kabupaten

def buat_lag(data_kabupaten):
    for nama_kab, df in data_kabupaten.items():
        df = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)
        for target in KOLOM_TARGET:
            if target not in df.columns: continue
            for lag in LAG_PERIODS:
                df[f"{target}_lag{lag}"] = df[target].shift(lag)
        lag_max   = max(LAG_PERIODS)
        cek_kolom = [f"{t}_lag{lag_max}" for t in KOLOM_TARGET if f"{t}_lag{lag_max}" in df.columns]
        df = df.dropna(subset=cek_kolom).reset_index(drop=True)
        data_kabupaten[nama_kab] = df
    return data_kabupaten

def split_data_test_only(data_kabupaten, tahun_pilihan):
    data_split = {}
    for nama_kab, df in data_kabupaten.items():
        df = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)
        test = df[df[KOLOM_TANGGAL].dt.year == tahun_pilihan].copy()
        if test.empty:
            test = df.tail(12).copy()
        kt = [c for c in KOLOM_TARGET if c in df.columns]
        data_split[nama_kab] = {'df_test_full': test, 'y_test': test[kt], 'kolom_tar': kt}
    return data_split

def predict_future_recursive(model, df_last_known, year_selected, feature_cols, base_year=2025):
    if model is None or not hasattr(model, 'predict'): return []
    if isinstance(feature_cols, dict):
        val = list(feature_cols.values())[0]
        feature_cols = val if isinstance(val, list) else list(feature_cols.keys())
    elif not isinstance(feature_cols, list):
        feature_cols = list(feature_cols)

    total_months = (year_selected - base_year) * 12
    if total_months <= 0: total_months = 12

    predictions = []
    current_row = df_last_known[feature_cols].iloc[-1:].copy()

    for i in range(total_months):
        raw_pred = model.predict(current_row)[0]
        y_pred_val = float(raw_pred[0]) if hasattr(raw_pred, '__len__') and not isinstance(raw_pred, (str, bytes)) else float(raw_pred)
        predictions.append(y_pred_val)

        lag_cols = sorted([c for c in feature_cols if 'lag' in c.lower()], reverse=True)
        if lag_cols:
            for j in range(len(lag_cols)):
                if j < len(lag_cols) - 1:
                    current_row[lag_cols[j]] = current_row[lag_cols[j+1]].values
                else:
                    current_row[lag_cols[j]] = y_pred_val
    return predictions[-12:]

def load_and_evaluate_model(data_split):
    hasil_model, hasil_eval, fitur_terpilih, importance_all = {}, [], {}, {}
    for nama_kab, split in data_split.items():
        hasil_model[nama_kab] = {}
        yte, kt = split['y_test'], split['kolom_tar']
        nama_kab_file = nama_kab.replace(" ", "_")
        
        path_rf  = f"model_final/RF_Tahap3_{nama_kab_file}.pkl"
        path_xgb = f"model_final/XGB_Tahap3_{nama_kab_file}.pkl"

        try:
            rf_m  = joblib.load(path_rf)
            xgb_m = joblib.load(path_xgb)
            fitur = list(rf_m.feature_names_in_)
            fitur_terpilih[nama_kab] = fitur
            
            Xte  = split['df_test_full'][fitur].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            t0 = time.time(); yp_rf  = rf_m.predict(Xte); rt_rf  = (time.time() - t0) * 1000
            t0 = time.time(); yp_xgb = xgb_m.predict(Xte); rt_xgb = (time.time() - t0) * 1000

            hasil_model[nama_kab]['rf']  = {'model': rf_m,  'y_pred': yp_rf,  'y_test': yte.values, 'fitur': fitur, 'rt': rt_rf}
            hasil_model[nama_kab]['xgb'] = {'model': xgb_m, 'y_pred': yp_xgb, 'y_test': yte.values, 'fitur': fitur, 'rt': rt_xgb}
            importance_all[nama_kab]     = pd.Series(rf_m.feature_importances_, index=fitur).sort_values(ascending=False)

            for mn, res in [('Random Forest', hasil_model[nama_kab]['rf']), ('XGBoost', hasil_model[nama_kab]['xgb'])]:
                for i, tgt in enumerate(kt):
                    yt, yp = res['y_test'][:, i], res['y_pred'][:, i]
                    hasil_eval.append({
                        'Kabupaten/Kota': nama_kab, 'Model': mn, 'Target': tgt, 'Target Label': TARGET_LABELS[tgt],
                        'MAE': round(mean_absolute_error(yt, yp), 3),
                        'RMSE': round(np.sqrt(mean_squared_error(yt, yp)), 3),
                        'R²': round(r2_score(yt, yp), 4),
                        'Running Time (ms)': round(res['rt'], 2),
                    })
        except FileNotFoundError:
            st.error(f"File model `.pkl` untuk {nama_kab} tidak ditemukan di folder model_final!")
            st.stop()
    return hasil_model, pd.DataFrame(hasil_eval), fitur_terpilih, importance_all

# ============================================================
# SIDEBAR (Sesuai Gambar 1)
# ============================================================
with st.sidebar:
    st.markdown("## 🫁 TB Lampung")
    st.markdown("**Dashboard Prediksi & Analisis**")
    st.divider()
    st.markdown("### 📂 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV per kabupaten/kota (15 file)",
        type=['csv'], accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file terupload")
    st.divider()
    
    st.markdown("### ⚙️ Parameter Prediksi")
    tahun_pilihan = st.selectbox(
        "Pilih Tahun Prediksi/Proyeksi:",
        options=[2023, 2024, 2025, 2026, 2027, 2028],
        index=2
    )
    st.divider()
    st.caption("© 2026 · Dashboard TB Lampung")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">🫁 Dashboard Prediksi Tuberkulosis</h1>
    <p style="margin:0.5rem 0 0 0;opacity:0.85;">Provinsi Lampung · Random Forest & XGBoost · 2021–2025</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_files:
    st.markdown("""<div class="info-box">
    <h3>👈 Mulai dengan Upload Data</h3>
    <p>Upload 15 file CSV kabupaten/kota di sidebar kiri.</p>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [(c1,'15','Kabupaten/Kota'),(c2,'4','Target Prediksi'),(c3,'2','Algoritma ML'),(c4,'2021–2025','Periode')]:
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.stop()

# Load & Proses Data
with st.spinner("Memuat data..."):
    data_kabupaten = load_semua_data(uploaded_files)
with st.spinner("Membuat lag features..."):
    data_kabupaten = buat_lag(data_kabupaten)

data_split = split_data_test_only(data_kabupaten, tahun_pilihan)
hasil_model, df_eval, fitur_terpilih, importance_all = load_and_evaluate_model(data_split)

# ============================================================
# TABS UTAMA
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Sebaran Kasus", "📊 Data & Tren", "🤖 Prediksi & Evaluasi", "💡 Rekomendasi"])

with tab1:
    st.markdown('<div class="section-title">Sebaran Kasus TB di Provinsi Lampung</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        t_peta = st.selectbox("Jenis TB", list(TARGET_LABELS.values()), key='t1')
        tk     = [k for k, v in TARGET_LABELS.items() if v == t_peta][0]
    with c2:
        m_peta = st.selectbox("Metrik", ['Total Kasus', 'Rata-rata per Bulan'])
    
    total = {}
    for nama, df in data_kabupaten.items():
        if tk in df.columns:
            total[nama] = df[tk].sum() if m_peta == 'Total Kasus' else round(df[tk].mean(), 2)
            
    df_p = pd.DataFrame([{'Kabupaten/Kota': k, 'Nilai': v, 'Lat': KOORDINAT.get(k, (-5, 105))[0], 'Lon': KOORDINAT.get(k, (-5, 105))[1]} for k, v in total.items()])
    if not df_p.empty:
        fig_map = px.scatter_map(
            df_p, lat='Lat', lon='Lon', size='Nilai', color='Nilai',
            hover_name='Kabupaten/Kota', color_continuous_scale='YlOrRd', size_max=50,
            zoom=7, center={'lat': -4.8, 'lon': 105.2}, map_style='open-street-map',
            title=f'{m_peta} {t_peta}'
        )
        st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Data Kasus TB & Tren</div>', unsafe_allow_html=True)
    t_tren = st.selectbox("Jenis TB", list(TARGET_LABELS.values()), key='t2')
    tk2    = [k for k, v in TARGET_LABELS.items() if v == t_tren][0]
    kab_sel = st.multiselect("Pilih Kabupaten/Kota", list(data_kabupaten.keys()), default=list(data_kabupaten.keys())[:3])
    if kab_sel:
        fig_tr = go.Figure()
        for kab in kab_sel:
            df = data_kabupaten[kab].sort_values(KOLOM_TANGGAL)
            if tk2 in df.columns:
                fig_tr.add_trace(go.Scatter(x=df[KOLOM_TANGGAL], y=df[tk2], mode='lines+markers', name=kab))
        st.plotly_chart(fig_tr, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Prediksi & Evaluasi Model</div>', unsafe_allow_html=True)
    kab_aktif = st.selectbox("Pilih Kabupaten/Kota untuk Proyeksi", list(hasil_model.keys()), key='kab_proj')
    
    if tahun_pilihan > 2025:
        model_rf_obj  = hasil_model[kab_aktif]['rf']['model']
        model_xgb_obj = hasil_model[kab_aktif]['xgb']['model']
        df_historis   = data_split[kab_aktif]['df_test_full']
        kolom_fitur   = fitur_terpilih[kab_aktif]

        pred_rf  = predict_future_recursive(model_rf_obj, df_historis, tahun_pilihan, kolom_fitur)
        pred_xgb = predict_future_recursive(model_xgb_obj, df_historis, tahun_pilihan, kolom_fitur)

        nama_bulan = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agt', 'Sep', 'Okt', 'Nov', 'Des']
        df_pred_tahun = pd.DataFrame({
            'Periode': [f"{b} {tahun_pilihan}" for b in nama_bulan],
            'Prediksi Random Forest': pred_rf,
            'Prediksi XGBoost': pred_xgb
        })
        st.markdown(f"### Proyeksi Kasus TB {kab_aktif} Tahun {tahun_pilihan}")
        st.line_chart(df_pred_tahun.set_index('Periode'))
        st.dataframe(df_pred_tahun, use_container_width=True)
    else:
        gc = df_eval.groupby('Model')[['MAE','RMSE','R²']].mean().round(3)
        st.dataframe(gc, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Rekomendasi Tindakan</div>', unsafe_allow_html=True)
    kab_rek = st.selectbox("Pilih Kabupaten", list(fitur_terpilih.keys()), key='kab_rek_s')
    if kab_rek:
        fitur_utama = fitur_terpilih[kab_rek][0]
        st.markdown(f"**Faktor Dominan di {kab_rek}:** `{fitur_utama}`")
        st.info(REKOMENDASI.get(fitur_utama, REKOMENDASI['DEFAULT']))
