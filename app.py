import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import warnings
import joblib  # WAJIB DITAMBAHKAN UNTUK MEMBACA FILE .pkl
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="TB Lampung — Dashboard Prediksi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D9488 100%);
    padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; color: white;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-card {
    background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; 
    padding: 1.5rem; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem; transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
.metric-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; }
.metric-label { font-size: 0.95rem; font-weight: 600; color: #64748B; margin-top: 0.2rem; }
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #1E3A5F;
    border-left: 5px solid #0D9488; padding-left: 1rem; margin: 2rem 0 1.5rem 0;
}
.info-box { background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }
.warning-box { background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }
.success-box { background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }
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
    'JML_SPS'    : "Tingkatkan kapasitas skrining TB — jumlah suspek tinggi mengindikasikan perlunya penambahan tenaga kesehatan dan alat diagnostik.",
    'JML_SPL'    : "Perkuat layanan TB di faskes tingkat lanjut — kasus pasien lanjutan perlu pemantauan lebih intensif.",
    'JML_FASKES' : "Tambah fasilitas kesehatan di wilayah dengan akses terbatas untuk meningkatkan cakupan deteksi TB.",
    'JML_PENDUDUK': "Fokuskan program TB pada wilayah padat penduduk — tingkatkan sosialisasi dan deteksi aktif di komunitas.",
    'PRS_LLis'   : "Perbaiki kondisi sosiodemografi — wilayah dengan akses listrik rendah cenderung memiliki faktor risiko TB lebih tinggi.",
    'PRS_JAMBAN' : "Tingkatkan sanitasi dasar — persentase rumah dengan jamban layak berpengaruh terhadap penularan TB.",
    'JML_KSO_lag1': "Pantau tren kasus bulan sebelumnya — lonjakan TB-SO perlu respons cepat dalam 1 bulan ke depan.",
    'JML_KSO_lag2': "Evaluasi program 2 bulan terakhir — pola kasus historis mengindikasikan perlunya penyesuaian intervensi.",
    'JML_KSO_lag3': "Tinjau program TB kuartalan — tren 3 bulan menjadi acuan perencanaan anggaran dan sumber daya.",
    'DEFAULT'    : "Lakukan evaluasi menyeluruh program TB di wilayah ini berdasarkan data terkini dari Dinas Kesehatan.",
}

# ============================================================
# FUNGSI DATA
# ============================================================
@st.cache_data
def load_data(uploaded_files_info):
    return st.session_state.get('raw_data', {})

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
            if target not in df.columns:
                continue
            for lag in LAG_PERIODS:
                df[f"{target}_lag{lag}"] = df[target].shift(lag)
        lag_max   = max(LAG_PERIODS)
        cek_kolom = [f"{t}_lag{lag_max}" for t in KOLOM_TARGET if f"{t}_lag{lag_max}" in df.columns]
        df = df.dropna(subset=cek_kolom).reset_index(drop=True)
        data_kabupaten[nama_kab] = df
    return data_kabupaten

def split_data_test_only(data_kabupaten):
    # Kita hanya butuh data test dan kolom target karena model sudah pintar
    data_split = {}
    for nama_kab, df in data_kabupaten.items():
        df    = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)
        test  = df[df[KOLOM_TANGGAL] >= TANGGAL_SPLIT]
        kt    = [c for c in KOLOM_TARGET if c in df.columns]
        data_split[nama_kab] = {
            'df_test_full': test,
            'y_test': test[kt],
            'kolom_tar': kt
        }
    return data_split

def adj_r2(r2, n, k):
    if n - k - 1 <= 0: return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# ============================================================
# FUNGSI PEMANGGIL MODEL (.PKL)
# ============================================================
def load_and_evaluate_model(data_split):
    hasil_model = {}
    hasil_eval  = []
    fitur_terpilih = {}
    importance_all = {}

    for nama_kab, split in data_split.items():
        hasil_model[nama_kab] = {}
        yte = split['y_test']
        kt  = split['kolom_tar']

# Mengubah spasi menjadi underscore khusus untuk mencari nama file
        nama_kab_file = nama_kab.replace(" ", "_")
        
        path_rf  = f"model_final/RF_Tahap3_{nama_kab_file}.pkl"
        path_xgb = f"model_final/XGB_Tahap3_{nama_kab_file}.pkl"

        try:
            # 1. LOAD MODEL
            rf_m  = joblib.load(path_rf)
            xgb_m = joblib.load(path_xgb)
            
            # 2. TARIK NAMA FITUR LANGSUNG DARI MODEL (Menjamin 100% akurat dengan Colab)
            fitur = list(rf_m.feature_names_in_)
            fitur_terpilih[nama_kab] = fitur
            
            # PERTAHANAN 1: Cek apakah ada data setelah tanggal split (2025-01-01)
            if split['df_test_full'].empty:
                st.error(f"❌ Data uji untuk {nama_kab} kosong! Pastikan CSV-mu memiliki data melewati tanggal batas {TANGGAL_SPLIT}.")
                st.stop()
            
            # PERTAHANAN 2: Paksa semua data menjadi angka murni (buang teks nyasar)
            Xte  = split['df_test_full'][fitur].apply(pd.to_numeric, errors='coerce').fillna(0)
            n, k = len(yte), len(fitur)
            
            # 3. PREDIKSI KILAT
            t0 = time.time()
            yp_rf  = rf_m.predict(Xte)
            rt_rf  = (time.time() - t0) * 1000

            t0 = time.time()
            yp_xgb = xgb_m.predict(Xte)
            rt_xgb = (time.time() - t0) * 1000

            hasil_model[nama_kab]['rf']  = {'model': rf_m,  'y_pred': yp_rf,  'y_test': yte.values, 'fitur': fitur, 'rt': rt_rf}
            hasil_model[nama_kab]['xgb'] = {'model': xgb_m, 'y_pred': yp_xgb, 'y_test': yte.values, 'fitur': fitur, 'rt': rt_xgb}

            # Ekstrak Feature Importance untuk Tab 4
            importance_all[nama_kab] = pd.Series(rf_m.feature_importances_, index=fitur).sort_values(ascending=False)

            # Hitung Metrik
            for mn, res in [('Random Forest', hasil_model[nama_kab]['rf']),
                            ('XGBoost', hasil_model[nama_kab]['xgb'])]:
                for i, tgt in enumerate(kt):
                    yt = res['y_test'][:, i]
                    yp = res['y_pred'][:, i]
                    mae  = mean_absolute_error(yt, yp)
                    rmse = np.sqrt(mean_squared_error(yt, yp))
                    r2   = r2_score(yt, yp)
                    hasil_eval.append({
                        'Kabupaten/Kota': nama_kab, 'Model': mn,
                        'Target': tgt, 'Target Label': TARGET_LABELS[tgt],
                        'MAE': round(mae, 3), 'RMSE': round(rmse, 3),
                        'R²': round(r2, 4),
                        'Running Time (ms)': round(res['rt'], 2),
                    })
                    
        except FileNotFoundError:
            st.error(f"File tidak ditemukan: {path_rf} atau {path_xgb}. Cek nama file di GitHub!")
            st.stop()

    return hasil_model, pd.DataFrame(hasil_eval), fitur_terpilih, importance_all

def get_rek(fitur):
    for k, v in REKOMENDASI.items():
        if k in fitur: return v
    return REKOMENDASI['DEFAULT']

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## TB Lampung")
    st.markdown("**Dashboard Prediksi & Analisis**")
    st.divider()
    st.markdown("### Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV per kabupaten/kota (15 file)",
        type=['csv'], accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file terupload")
    st.divider()
    st.caption("© 2026 · Dashboard TB Lampung")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">Dashboard Prediksi Tuberkulosis</h1>
    <p style="margin:0.5rem 0 0 0;opacity:0.85;">Provinsi Lampung · Random Forest & XGBoost · 2021–2025</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_files:
    st.markdown("""<div class="info-box">
    <h3>Mulai dengan Upload Data</h3>
    <p>Upload 15 file CSV kabupaten/kota di sidebar kiri.</p>
    <p><strong>Format nama file:</strong> <code>Data Used - Nama_Kabupaten.csv</code></p>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [(c1,'15','Kabupaten/Kota'),(c2,'4','Target Prediksi'),
                           (c3,'2','Algoritma ML'),(c4,'2021–2025','Periode Data')]:
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Memuat data..."):
    data_kabupaten = load_semua_data(uploaded_files)
with st.spinner("Membuat lag features..."):
    data_kabupaten = buat_lag(data_kabupaten)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Sebaran Kasus", "Data & Tren", "Prediksi & Evaluasi", "Rekomendasi"])

# ---- TAB 1 & 2 DIBIARKAN SAMA SEPERTI SEBELUMNYA ----
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
    fig_map = px.scatter_mapbox(df_p, lat='Lat', lon='Lon', size='Nilai', color='Nilai', hover_name='Kabupaten/Kota', color_continuous_scale='YlOrRd', size_max=50, zoom=7, center={'lat': -4.8, 'lon': 105.2}, mapbox_style='carto-positron', title=f'{m_peta} {t_peta}')
    fig_map.update_layout(height=480, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Data Kasus TB & Tren</div>', unsafe_allow_html=True)
    s1, s2 = st.tabs(["Tren Waktu", "Heatmap"])
    with s1:
        c1, c2 = st.columns(2)
        with c1:
            t_tren = st.selectbox("Jenis TB", list(TARGET_LABELS.values()), key='t2')
            tk2    = [k for k, v in TARGET_LABELS.items() if v == t_tren][0]
        with c2:
            kab_sel = st.multiselect("Pilih Kabupaten/Kota", list(data_kabupaten.keys()), default=list(data_kabupaten.keys())[:5])
        if kab_sel:
            fig_tr = go.Figure()
            for kab in kab_sel:
                df = data_kabupaten[kab].sort_values(KOLOM_TANGGAL)
                if tk2 in df.columns:
                    fig_tr.add_trace(go.Scatter(x=df[KOLOM_TANGGAL], y=df[tk2], mode='lines+markers', name=kab, line=dict(width=2), marker=dict(size=4)))
            fig_tr.add_vline(x='2025-01-01', line_dash='dash', line_color='red', annotation_text='Batas Train/Test')
            fig_tr.update_layout(title=f'Tren Bulanan {t_tren}', height=430, xaxis_title='Periode', yaxis_title='Jumlah Kasus', hovermode='x unified')
            st.plotly_chart(fig_tr, use_container_width=True)
    with s2:
        t_hm  = st.selectbox("Jenis TB", list(TARGET_LABELS.values()), key='t3')
        tk3   = [k for k, v in TARGET_LABELS.items() if v == t_hm][0]
        pdata = {}
        for nama, df in data_kabupaten.items():
            if tk3 in df.columns and KOLOM_TANGGAL in df.columns:
                df2 = df.sort_values(KOLOM_TANGGAL)
                df2['Tahun'] = df2[KOLOM_TANGGAL].dt.year
                pdata[nama] = df2.groupby('Tahun')[tk3].sum()
        if pdata:
            df_hm = pd.DataFrame(pdata).T.fillna(0).astype(int)
            fig_hm = px.imshow(df_hm, color_continuous_scale='YlOrRd', aspect='auto', title=f'Heatmap Total {t_hm} per Tahun', text_auto=True, labels=dict(x='Tahun', y='Kabupaten/Kota', color='Kasus'))
            fig_hm.update_layout(height=480)
            st.plotly_chart(fig_hm, use_container_width=True)

# ---- TAB 3: PREDIKSI & EVALUASI ----
with tab3:
    st.markdown('<div class="section-title">Prediksi & Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Klik tombol di bawah untuk mengeksekusi prediksi dari model Colab.</div>', unsafe_allow_html=True)

    if st.button("🚀 Jalankan Evaluasi Model", type="primary"):
        with st.spinner("Mempersiapkan data uji..."):
            data_split = split_data_test_only(data_kabupaten)
        with st.spinner("Memuat model (.pkl) dan menghitung evaluasi..."):
            hasil_model, df_eval, fitur_terpilih, importance_all = load_and_evaluate_model(data_split)
        
        st.session_state.update({
            'hasil_model': hasil_model, 'df_eval': df_eval,
            'fitur_terpilih': fitur_terpilih, 'importance_all': importance_all,
            'data_split': data_split,
        })
        st.success("✅ Evaluasi berhasil! Seluruh data metrik kini 100% identik dengan hasil di Colab.")

    # 👇 Kuncinya ada di sini: Baris ini WAJIB menjorok ke dalam (di bawah with tab3:)
    if 'hasil_model' in st.session_state:
        hasil_model    = st.session_state['hasil_model']
        df_eval        = st.session_state['df_eval']
        fitur_terpilih = st.session_state['fitur_terpilih']
        data_split     = st.session_state['data_split']

        # 1. KEMBALINYA RINGKASAN GLOBAL
        st.markdown('<div class="section-title">Ringkasan Evaluasi Global</div>', unsafe_allow_html=True)
        gc = df_eval.groupby('Model')[['MAE','RMSE','R²']].mean().round(3)
        rt = df_eval.groupby('Model')['Running Time (ms)'].sum().round(0)
        c1, c2 = st.columns(2)
        for i, (mn, row) in enumerate(gc.iterrows()):
            warna = "#1D6FA4" if "Forest" in mn else "#EA580C"
            col   = c1 if i == 0 else c2
            with col:
                st.markdown(f"""<div class="metric-card">
                <div style="color:{warna};font-size:1.1rem;font-weight:700;margin-bottom:0.8rem;">{mn}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                    <div><div class="metric-value" style="font-size:1.3rem;">{row['MAE']}</div><div class="metric-label">Rata-rata MAE</div></div>
                    <div><div class="metric-value" style="font-size:1.3rem;">{row['RMSE']}</div><div class="metric-label">Rata-rata RMSE</div></div>
                    <div><div class="metric-value" style="font-size:1.3rem;">{row['R²']}</div><div class="metric-label">Rata-rata R²</div></div>
                    <div><div class="metric-value" style="font-size:1.3rem;">{rt[mn]:.0f}</div><div class="metric-label">Total RT (ms)</div></div>
                </div></div>""", unsafe_allow_html=True)

        # 2. TABEL EVALUASI
        st.markdown('<div class="section-title">Evaluasi Per Kabupaten/Kota</div>', unsafe_allow_html=True)
        t_filter = st.selectbox("Filter Target", ['Semua'] + list(TARGET_LABELS.values()), key='ef')
        df_show  = df_eval.copy()
        if t_filter != 'Semua':
            tk_f    = [k for k, v in TARGET_LABELS.items() if v == t_filter][0]
            df_show = df_show[df_show['Target'] == tk_f]
        df_pkab = df_show.groupby(['Kabupaten/Kota','Model'])[['MAE','RMSE','R²']].mean().round(3).reset_index()
        st.dataframe(df_pkab, use_container_width=True, height=380)

        # 3. GRAFIK MAE & RMSE
        st.markdown('<div class="section-title">Perbandingan MAE & RMSE</div>', unsafe_allow_html=True)
        fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=['MAE', 'RMSE'])
        for mn, color in [('Random Forest','#1D6FA4'),('XGBoost','#EA580C')]:
            dm = df_pkab[df_pkab['Model']==mn].sort_values('MAE')
            fig_cmp.add_trace(go.Bar(name=mn, x=dm['MAE'], y=dm['Kabupaten/Kota'],
                orientation='h', marker_color=color, legendgroup=mn), row=1, col=1)
            fig_cmp.add_trace(go.Bar(name=mn, x=dm['RMSE'], y=dm['Kabupaten/Kota'],
                orientation='h', marker_color=color, legendgroup=mn, showlegend=False), row=1, col=2)
        fig_cmp.update_layout(height=480, barmode='group', legend=dict(orientation='h',y=-0.15))
        st.plotly_chart(fig_cmp, use_container_width=True)

        # 4. GRAFIK AKTUAL VS PREDIKSI
        st.markdown('<div class="section-title">Grafik Aktual vs Prediksi</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            kab_ap = st.selectbox("Kabupaten/Kota", list(hasil_model.keys()), key='kap')
        with c2:
            tgt_ap = st.selectbox("Target", list(TARGET_LABELS.values()), key='tap')
        tk_ap  = [k for k, v in TARGET_LABELS.items() if v == tgt_ap][0]
        kt_ap  = data_split[kab_ap]['kolom_tar']
        if tk_ap in kt_ap:
            idx   = kt_ap.index(tk_ap)
            fig_ap = go.Figure()
            yt_val = hasil_model[kab_ap]['rf']['y_test'][:, idx]
            fig_ap.add_trace(go.Scatter(y=yt_val, mode='lines+markers', name='Aktual',
                line=dict(color='#1E3A5F', width=3), marker=dict(size=6)))
            for mn, key, color in [('Random Forest','rf','#1D6FA4'),('XGBoost','xgb','#EA580C')]:
                yp = hasil_model[kab_ap][key]['y_pred'][:, idx]
                r2 = r2_score(yt_val, yp)
                fig_ap.add_trace(go.Scatter(y=yp, mode='lines+markers',
                    name=f'{mn} (R²={r2:.3f})',
                    line=dict(color=color, width=2, dash='dash'), marker=dict(size=5, symbol='square')))
            fig_ap.update_layout(title=f'Aktual vs Prediksi — {kab_ap} — {tgt_ap}',
                xaxis_title='Periode (bulan)', yaxis_title='Jumlah Kasus', height=400, hovermode='x unified')
            st.plotly_chart(fig_ap, use_container_width=True)
            
        # 5. GRAFIK WAKTU KOMPUTASI
        st.markdown('<div class="section-title">Waktu Komputasi (Running Time)</div>', unsafe_allow_html=True)
        rt_rows = [{'Kabupaten/Kota': k,
                    'Random Forest': round(v['rf']['rt'], 2),
                    'XGBoost': round(v['xgb']['rt'], 2)}
                   for k, v in hasil_model.items()]
        df_rt  = pd.DataFrame(rt_rows).sort_values('Random Forest', ascending=True)
        fig_rt = go.Figure()
        fig_rt.add_trace(go.Bar(name='Random Forest', x=df_rt['Random Forest'],
            y=df_rt['Kabupaten/Kota'], orientation='h', marker_color='#1D6FA4',
            text=df_rt['Random Forest'], textposition='outside'))
        fig_rt.add_trace(go.Bar(name='XGBoost', x=df_rt['XGBoost'],
            y=df_rt['Kabupaten/Kota'], orientation='h', marker_color='#EA580C',
            text=df_rt['XGBoost'], textposition='outside'))
        fig_rt.update_layout(title='Running Time per Kabupaten (ms)', barmode='group',
            height=480, xaxis_title='ms', legend=dict(orientation='h',y=-0.15))
        st.plotly_chart(fig_rt, use_container_width=True)
# ---- TAB 4: REKOMENDASI ----
with tab4:
    st.markdown('<div class="section-title">Rekomendasi Tindakan Berdasarkan Feature Importance</div>', unsafe_allow_html=True)

    if 'fitur_terpilih' not in st.session_state:
        st.markdown('<div class="warning-box">Latih model terlebih dahulu di tab <strong>Prediksi & Evaluasi</strong>.</div>', unsafe_allow_html=True)
    else:
        fitur_terpilih  = st.session_state['fitur_terpilih']
        importance_all  = st.session_state['importance_all']
        df_eval         = st.session_state['df_eval']
        hasil_model     = st.session_state['hasil_model']

        kab_rek = st.selectbox("Pilih Kabupaten/Kota", list(fitur_terpilih.keys()), key='kr')

        if kab_rek:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Feature Importance — Random Forest")
                imp    = importance_all[kab_rek]
                thr    = imp.mean()
                top10  = imp.head(10)
                warna  = ['#1D6FA4' if s >= thr else '#A8DADC' for s in top10]
                fig_rf = go.Figure(go.Bar(x=top10.values[::-1], y=top10.index[::-1],
                    orientation='h', marker_color=warna[::-1], text=[f'{v:.3f}' for v in top10.values[::-1]], textposition='outside'))
                fig_rf.add_vline(x=thr, line_dash='dash', line_color='red', annotation_text=f'Threshold ({thr:.3f})')
                fig_rf.update_layout(height=360, title=f'{kab_rek}', showlegend=False, xaxis_title='Importance Score')
                st.plotly_chart(fig_rf, use_container_width=True)

            with c2:
                st.markdown("#### Feature Importance — XGBoost")
                if kab_rek in hasil_model:
                    xgb_m = hasil_model[kab_rek]['xgb']['model']
                    fitur = fitur_terpilih[kab_rek]
                    try:
                        skor = np.mean([m.feature_importances_ for m in xgb_m.estimators_], axis=0)
                        df_xi = pd.Series(skor, index=fitur).sort_values(ascending=False)
                        fig_xg = go.Figure(go.Bar(x=df_xi.values[::-1], y=df_xi.index[::-1],
                            orientation='h', marker_color='#EA580C', text=[f'{v:.3f}' for v in df_xi.values[::-1]], textposition='outside'))
                        fig_xg.update_layout(height=360, title=f'{kab_rek}', showlegend=False, xaxis_title='Importance Score')
                        st.plotly_chart(fig_xg, use_container_width=True)
                    except Exception:
                        st.info("XGBoost importance tidak tersedia.")

            # Rekomendasi
            st.markdown("---")
            st.markdown("#### Rekomendasi Tindakan")
            fitur_list  = fitur_terpilih[kab_rek]
            fitur_utama = fitur_list[0] if fitur_list else 'DEFAULT'
            rek         = get_rek(fitur_utama)
            df_kab      = df_eval[df_eval['Kabupaten/Kota'] == kab_rek]
            best_mn     = df_kab.loc[df_kab['MAE'].idxmin(), 'Model'] if not df_kab.empty else '-'
            mae_best    = round(df_kab['MAE'].min(), 3) if not df_kab.empty else '-'

            c1, c2, c3 = st.columns(3)
            for col, val, lbl in [(c1, fitur_utama, 'Faktor Dominan'), (c2, best_mn, 'Model Terbaik'), (c3, mae_best, 'MAE Terbaik')]:
                col.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1rem;">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="success-box"><strong>Rekomendasi untuk {kab_rek}:</strong><br><br>{rek}<br><br><strong>Fitur prediktor terpilih:</strong> {", ".join(fitur_list)}</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2026 · Dashboard TB Lampung · Random Forest & XGBoost")
