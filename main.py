import streamlit as st
import json
import pandas as pd
import numpy as np
from data_processing import preprocess_data, handle_missing_data, train_som, fix_unhashable_columns, preprocess_data_interactive
from visualizations import (show_summary_table, show_visualizations, handle_meta_clustering, 
                           handle_neuron_details, handle_anomaly_detection, show_som_validation, 
                           show_meta_clustering_validation, show_advanced_analysis)
from session_state import initialize_session_state, reset_session_state
from text_content import get_main_description, get_som_description, get_user_gains
from meta_clustering_analysis import show_meta_clustering_analysis

# Meta kümeleme analizine ait session state başlangıç değerleri (her sekmede korunsun)
for key, default in {
    "meta_clustering_done": False,
    "direct_meta_labels": None,
    "direct_meta_metrics": None,
    "last_n_clusters": 5,
    "direct_meta_cluster_centers": None,
    "comparison_type": "Doğrudan K-means vs SOM"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

initialize_session_state()

st.set_page_config(
    page_title="Coraza WAF Log Analizi", 
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Gelişmiş CSS stilleri
st.markdown("""
<style>
    /* Ana tema renkleri - geliştirilmiş palet */
    :root {
        --primary-color: #1e40af;
        --primary-light: #3b82f6;
        --secondary-color: #64748b;
        --accent-color: #0f172a;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-300: #cbd5e1;
        --neutral-400: #94a3b8;
        --neutral-500: #64748b;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
        --neutral-900: #0f172a;
    }

    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global stil */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Typography iyileştirmeleri */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--neutral-900) !important;
        letter-spacing: -0.02em;
    }

    /* Ana başlık - geliştirilmiş */
    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid var(--neutral-200);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
    }
    
    .main-header h1 {
        color: var(--neutral-900) !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        letter-spacing: -0.025em;
    }
    
    .main-header p {
        color: var(--neutral-600);
        font-size: 1.125rem;
        margin: 0;
        font-weight: 400;
    }

    /* Özellik kartları - geliştirilmiş */
    .feature-card {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--primary-color);
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: var(--primary-light);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }

    .feature-card:hover::before {
        transform: scaleY(1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        display: block;
        opacity: 0.8;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--neutral-900) !important;
        margin-bottom: 0.75rem;
    }
    
    .feature-desc {
        color: var(--neutral-600);
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Upload bölümü - geliştirilmiş */
    .upload-section {
        background: linear-gradient(135deg, var(--neutral-50) 0%, white 100%);
        border: 2px dashed var(--neutral-300);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .upload-section:hover {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, white 0%, var(--neutral-50) 100%);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.1);
    }

    /* Upload section başlık ve metin stilleri - beyaz renkte */
    .upload-section h4 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        font-size: 1.125rem;
    }

    .upload-section p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0;
        font-size: 0.95rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }

    /* Upload section background güncelleme - daha koyu arka plan */
    .upload-section {
        background: linear-gradient(135deg, var(--neutral-600) 0%, var(--neutral-700) 100%);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .upload-section:hover {
        border-color: var(--primary-light);
        background: linear-gradient(135deg, var(--neutral-700) 0%, var(--neutral-600) 100%);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
    }

    /* Metrik kartları - geliştirilmiş */
    .metric-card {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }

    .metric-card:hover {
        border-color: var(--neutral-300);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* Sidebar geliştirilmiş */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--neutral-900) 0%, var(--neutral-800) 100%);
    }
    
    /* Analiz Menüsü Header - Geliştirilmiş */
    .sidebar-header {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.9) 0%, rgba(59, 130, 246, 0.8) 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #60a5fa, #34d399, #fbbf24);
        animation: pulse 2s ease-in-out infinite alternate;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    .sidebar-header h2 {
        color: white !important;
        text-align: center;
        margin: 0;
        font-weight: 700;
        font-size: 1.25rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        letter-spacing: 0.025em;
        position: relative;
        z-index: 1;
    }

    .sidebar-header::after {
        content: '📊';
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        font-size: 1.5rem;
        opacity: 0.7;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }

    /* Sidebar Menu Butonları - Geliştirilmiş */
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .css-1d391kg .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.9) 0%, rgba(30, 64, 175, 0.8) 100%) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.4) !important;
    }

    .css-1d391kg .stButton > button:hover::before {
        left: 100%;
    }

    /* Aktif menü göstergesi - geliştirilmiş */
    .css-1d391kg .markdown-text-container {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500 !important;
        background: rgba(34, 197, 94, 0.2) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin-top: 1rem !important;
        backdrop-filter: blur(5px) !important;
        text-align: center !important;
    }

    .css-1d391kg .markdown-text-container strong {
        color: #60f282 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }

    /* Butonlar - geliştirilmiş */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--neutral-300);
        font-weight: 500;
        transition: all 0.3s ease;
        background: white;
        color: var(--neutral-900);
        font-family: 'Inter', sans-serif;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.15);
        transform: translateY(-1px);
    }

    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        border-color: var(--primary-color);
        color: white;
    }

    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, var(--primary-color) 100%);
        border-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.25);
    }

    /* Form elemanları - geliştirilmiş */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid var(--neutral-300);
        transition: all 0.2s ease;
    }

    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
    }
    
    .stNumberInput > div > div {
        border-radius: 10px;
        border: 1px solid var(--neutral-300);
        transition: all 0.2s ease;
    }

    .stNumberInput > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
    }

    /* Expander - geliştirilmiş */
    .streamlit-expanderHeader {
        background: var(--neutral-50);
        border: 1px solid var(--neutral-200);
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background: var(--neutral-100);
        border-color: var(--neutral-300);
    }

    /* Tabs - geliştirilmiş */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--neutral-300);
        background: var(--neutral-50);
    }

    /* Alert'ler - geliştirilmiş */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Progress bar - geliştirilmiş */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
        border-radius: 10px;
    }

    /* Dataframe - geliştirilmiş */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--neutral-200);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Radio butonları - geliştirilmiş */
    .stRadio > div {
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        padding: 1rem;
        background: white;
        transition: all 0.2s ease;
    }

    .stRadio > div:hover {
        border-color: var(--neutral-300);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Radio label görünürlüğü */
    .stRadio label {
        color: var(--neutral-900) !important;
        font-weight: 500;
        font-size: 1rem;
    }

    /* Radio seçenekleri */
    .stRadio div[role="radiogroup"] > label {
        color: var(--neutral-700) !important;
        background: var(--neutral-50);
        border: 1px solid var(--neutral-200);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem;
        transition: all 0.2s ease;
    }

    .stRadio div[role="radiogroup"] > label:hover {
        background: white;
        border-color: var(--primary-color);
        color: var(--primary-color) !important;
    }

    /* Metrik değerleri */
    .metric-container .metric-value {
        color: var(--neutral-900) !important;
        font-weight: 600;
    }

    /* Dark mode uyumluluğu */
    @media (prefers-color-scheme: dark) {
        .main-header, .feature-card, .metric-card {
            background: var(--neutral-800);
            border-color: var(--neutral-700);
            color: white;
        }
        
        .feature-title, .main-header h1 {
            color: white !important;
        }
        
        .feature-desc, .main-header p {
            color: var(--neutral-300);
        }
        
        .upload-section {
            background: var(--neutral-800);
            border-color: var(--neutral-700);
        }
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .feature-card {
            margin: 0.5rem 0;
            padding: 1.5rem;
        }
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Scroll bar - geliştirilmiş */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--neutral-400), var(--neutral-500));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--neutral-500), var(--neutral-600));
    }

    /* Gelişmiş animasyonlar */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Focus states */
    button:focus, input:focus, select:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section - minimal
st.markdown("""
    <div class="main-header">
        <h1>Coraza WAF Log Analizi</h1>
        <p>Coraza Web Application Firewall loglarınızı analiz edin ve güvenlik tehditleri tespit edin</p>
        <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
            🎯 Coraza WAF log formatı için özel olarak optimize edilmiştir
        </p>
    </div>
""", unsafe_allow_html=True)

# Ana bilgi kartları - sade
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Analiz</div>
            <div class="feature-desc">
                WAF loglarındaki anormal davranışları tespit edin ve güvenlik tehditleri analiz edin.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">Teknoloji</div>
            <div class="feature-desc">
                Self-Organizing Map ve Meta Kümeleme algoritmalarıyla makine öğrenmesi.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Sonuç</div>
            <div class="feature-desc">
                Detaylı görselleştirmeler ve PDF raporlarıyla kapsamlı analiz sonuçları.
            </div>
        </div>
    """, unsafe_allow_html=True)

# SOM Açıklaması
with st.expander("Self-Organizing Map (SOM) Nedir?", expanded=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(get_som_description())
    with col2:
        st.markdown("""
        **Faydalar:**
        - Anomali tespiti
        - Veri kümeleme  
        - Görselleştirme
        - Hızlı analiz
        """)

# Kullanıcı faydaları
st.markdown("---")
st.markdown("### Bu Analiz Size Ne Sağlar?")

benefits_col1, benefits_col2 = st.columns(2)

with benefits_col1:
    st.markdown("""
    **Güvenlik Tehditleri Tespiti**  
    Anormal log davranışlarını otomatik tespit
    
    **Görsel Analiz**  
    İnteraktif grafikler ve haritalar
    
    **Kümeleme Analizi**  
    Benzer davranış gösteren logları gruplandırma
    """)

with benefits_col2:
    st.markdown("""
    **PDF Rapor Üretimi**  
    Profesyonel analiz raporları
    
    **Gerçek Zamanlı İşleme**  
    Hızlı ve etkili veri analizi
    
    **Kapsamlı Metrikler**  
    Silüet skoru, küme analizi ve daha fazlası
    """)

st.markdown("---")

# Veri yükleme bölümü
st.markdown("### Veri Yükleme")
st.markdown("Analizi başlatmak için **Coraza WAF** log dosyanızı yükleyin veya örnek veri kullanın.")

st.info("""
**📋 Coraza Log Format Gereksinimleri:**
- **Format:** JSON (satır başına bir JSON objesi)
- **Encoding:** UTF-8
- **Alan adları:** Coraza WAF standart alanları
- **Örnek:** `transaction.id`, `transaction.client_ip`, `rules.matched` vb.
""", icon="ℹ️")

# Yükleme seçenekleri
upload_col1, upload_col2 = st.columns([1, 1])

with upload_col1:
    st.markdown("""
        <div class="upload-section">
            <h4>📤 Kendi Dosyanızı Yükleyin</h4>
            <p>JSON formatında WAF log dosyası</p>
        </div>
    """, unsafe_allow_html=True)

with upload_col2:
    st.markdown("""
        <div class="upload-section">
            <h4>🧪 Örnek Veri Kullanın</h4>
            <p>Hemen test etmek için hazır veri</p>
        </div>
    """, unsafe_allow_html=True)

upload_method = st.radio(
    "Veri yükleme yöntemi seçin:",
    ["📤 Dosya Yükle", "🧪 Örnek Veri Kullan"],
    horizontal=True,
    label_visibility="visible",
    key="upload_method_radio"
)

if upload_method == "📤 Dosya Yükle":
    # Dosya formatı açıklaması
    with st.expander("📋 Desteklenen JSON Formatları", expanded=False):
        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
            <h4 style="margin: 0; color: #856404;">⚠️ Önemli Format Bilgileri</h4>
            <p style="margin: 0.5rem 0 0 0;">Dosyanız aşağıdaki formatlardan birinde olmalıdır:</p>
        </div>
        
        ### 1️⃣ Tek Transaction Formatı:
        ```json
        {
          "transaction": {
            "client_port": 12345,
            "request": {"uri": "/login", "method": "POST"},
            "timestamp": "2023-01-01T10:20:30Z",
            "is_interrupted": false
          }
        }
        ```
        
        ### 2️⃣ Transaction Array Formatı:
        ```json
        [
          {"transaction": {...}},
          {"transaction": {...}}
        ]
        ```
        
        ### 3️⃣ Düz JSON Array Formatı:
        ```json
        [
          {
            "client_port": 12345,
            "request.uri": "/login",
            "timestamp": "2023-01-01T10:20:30Z"
          }
        ]
        ```
        """)
    
    uploaded_file = st.file_uploader(
        "📁 Log dosyanızı seçin",
        type=["json"],
        help="JSON formatında WAF log dosyası yükleyin"
    )
    
elif upload_method == "🧪 Örnek Veri Kullan":
    st.info("🎯 Uygulamayı hemen test etmek için örnek WAF log verisi kullanabilirsiniz.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Örnek Veri Yükle", type="primary", use_container_width=True):
            # Örnek veri yükleme kodu aynı kalacak...
            import io
            import json
            
            example_data = [
                {
                    "transaction": {
                        "client_port": 12345,
                        "request": {
                            "uri": "/login",
                            "method": "POST"
                        },
                        "timestamp": "2023-01-01T10:20:30Z",
                        "is_interrupted": False
                    }
                },
                {
                    "transaction": {
                        "client_port": 67890,
                        "request": {
                            "uri": "/admin",
                            "method": "GET"
                        },
                        "timestamp": "2023-01-01T10:25:30Z",
                        "is_interrupted": True
                    }
                },
                {
                    "transaction": {
                        "client_port": 54321,
                        "request": {
                            "uri": "/dashboard",
                            "method": "GET"
                        },
                        "timestamp": "2023-01-01T10:30:30Z",
                        "is_interrupted": False
                    }
                },
                {
                    "transaction": {
                        "client_port": 11111,
                        "request": {
                            "uri": "/api/users",
                            "method": "GET"
                        },
                        "timestamp": "2023-01-01T11:20:30Z",
                        "is_interrupted": False
                    }
                },
                {
                    "transaction": {
                        "client_port": 22222,
                        "request": {
                            "uri": "/login",
                            "method": "POST"
                        },
                        "timestamp": "2023-01-01T12:20:30Z",
                        "is_interrupted": True
                    }
                }
            ]
            
            for i in range(20):
                import random
                import datetime
                
                uri_choices = ["/login", "/admin", "/dashboard", "/api/users", "/logout", "/profile", "/settings"]
                method_choices = ["GET", "POST", "PUT", "DELETE"]
                
                example_data.append({
                    "transaction": {
                        "client_port": random.randint(10000, 60000),
                        "request": {
                            "uri": random.choice(uri_choices),
                            "method": random.choice(method_choices)
                        },
                        "timestamp": (datetime.datetime(2023, 1, 1, 10, 0, 0) + 
                                      datetime.timedelta(minutes=random.randint(0, 1440))).isoformat(),
                        "is_interrupted": random.random() < 0.3
                    }
                })
            
            example_json = json.dumps(example_data)
            uploaded_file = io.BytesIO(example_json.encode())
            uploaded_file.name = "example_data.json"
            
            st.success("✅ Örnek veri başarıyla yüklendi! Analiz başlayabilir.")
        else:
            uploaded_file = None

# Progress göstergesi
if uploaded_file:
    st.markdown("---")
    st.markdown("### 🔄 Veri İşleme Durumu")
    
    st.markdown("""
        <div class="progress-container">
            <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">📊 İşleme Durumu</h4>
        </div>
    """, unsafe_allow_html=True)
    
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    with progress_col2:
        step_info = st.empty()

if uploaded_file and st.session_state.df is None:
    try:
        status_text.text("📁 Dosya okunuyor...")
        progress_bar.progress(20)
        step_info.info("Adım 1/5")
        
        raw_data = json.load(uploaded_file)
        
        status_text.text("🔄 JSON yapısı işleniyor...")
        progress_bar.progress(40)
        step_info.info("Adım 2/5")
        
        # JSON yapısını işle
        transactions = []
        if isinstance(raw_data, dict):
            if 'transaction' in raw_data:
                transactions.append(raw_data['transaction'])
            else:
                transactions.append(raw_data)
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    if 'transaction' in item:
                        transactions.append(item['transaction'])
                    else:
                        transactions.append(item)
        
        status_text.text("📊 DataFrame oluşturuluyor...")
        progress_bar.progress(60)
        step_info.info("Adım 3/5")
        
        # Nokta ayracı ile normalize et
        try:
            df = pd.json_normalize(transactions, sep='.')
            df = fix_unhashable_columns(df)
            st.session_state.df = df
        except Exception as e:
            st.warning(f"JSON normalize hatası: {str(e)}. Basit formata dönüştürülüyor.")
            df = pd.DataFrame(transactions)
            df = fix_unhashable_columns(df)
            st.session_state.df = df
        
        status_text.text("✅ Veri başarıyla yüklendi!")
        progress_bar.progress(100)
        step_info.success("Tamamlandı!")
        
        # Veri önizleme
        st.markdown("### 📋 Veri Önizleme")
        preview_col1, preview_col2 = st.columns([2, 1])
        
        with preview_col1:
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        with preview_col2:
            st.metric("Toplam Kayıt", len(st.session_state.df))
            st.metric("Sütun Sayısı", len(st.session_state.df.columns))
            st.write("**Sütunlar:**")
            for col in st.session_state.df.columns[:5]:
                st.write(f"• {col}")
            if len(st.session_state.df.columns) > 5:
                st.write(f"• ... ve {len(st.session_state.df.columns) - 5} sütun daha")
        
    except Exception as e:
        st.error(f"❌ Dosya okuma hatası: {str(e)}")
        st.stop()

if st.session_state.df is not None:
    handle_missing_data()

if st.session_state.X is not None and st.session_state.som is None:
    st.markdown("---")
    st.markdown("### 🧠 SOM Eğitimi")
    
    # SOM parametreleri
    n_samples = len(st.session_state.df)
    grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
    st.session_state.grid_size = grid_size
    
    # SOM bilgileri
    som_col1, som_col2 = st.columns([2, 1])
    with som_col1:
        st.info(f"""
        **🎯 SOM Parametreleri:**
        - Grid Boyutu: {grid_size}x{grid_size}
        - Veri Boyutu: {n_samples} kayıt
        - Özellik Sayısı: {st.session_state.X.shape[1]}
        """)
    
    with som_col2:
        if st.button("🚀 SOM Eğitimini Başlat", type="primary"):
            st.session_state.start_som_training = True
    
    # SOM eğitimi
    if st.session_state.get('start_som_training', False) or st.session_state.som is None:
        sigma = grid_size / 2.0
        learning_rate = 0.5
        num_iterations = 1000
        
        # Progress bar için
        som_progress = st.progress(0)
        som_status = st.empty()
        
        som_status.text("🧠 SOM ağı eğitiliyor...")
        
        with st.spinner("SOM eğitimi devam ediyor..."):
            st.session_state.som = train_som(st.session_state.X, grid_size, sigma, learning_rate, num_iterations)
        
        som_progress.progress(50)
        som_status.text("📍 BMU koordinatları hesaplanıyor...")
        
        # BMU koordinatlarını ekle
        bmu_coords = np.array([st.session_state.som.winner(x) for x in st.session_state.X])
        st.session_state.df['bmu_x'] = bmu_coords[:, 0]
        st.session_state.df['bmu_y'] = bmu_coords[:, 1]
        st.session_state.df['quantization_error'] = [st.session_state.som.quantization_error(x.reshape(1, -1)) for x in st.session_state.X]
        
        som_progress.progress(80)
        som_status.text("🔧 Veri yapısı optimize ediliyor...")
        
        # BMU koordinatlarını tekrar hashlenebilir yap
        st.session_state.df = fix_unhashable_columns(st.session_state.df)
        
        # SOM etiketlerini oluştur
        st.session_state.som_labels = bmu_coords[:, 0] * grid_size + bmu_coords[:, 1]
        
        # SOM eğitimi tamamlandı
        st.session_state.som_done = True
        st.session_state.start_som_training = False
        
        som_progress.progress(100)
        som_status.success("✅ SOM eğitimi başarıyla tamamlandı!")
        
        st.balloons()
        st.rerun()

if st.session_state.som is not None:
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("""
        <div class="sidebar-header">
            <h2>Analiz Menüsü</h2>
        </div>
    """, unsafe_allow_html=True)
    
    menu_options = {
        "Temel Analizler": "Temel veri analizi ve görselleştirmeler",
        "Gelişmiş Analizler": "İleri seviye algoritmalar ve metrikler", 
        "Meta Kümeleme": "Meta kümeleme analizi ve karşılaştırmalar"
    }
    
    # Menu seçimi
    selected_menu = None
    for option, description in menu_options.items():
        if st.sidebar.button(
            option, 
            use_container_width=True,
            help=description
        ):
            selected_menu = option
            st.session_state.current_menu = option
    
    # Session state'ten mevcut menüyü al
    current_menu = st.session_state.get('current_menu', 'Temel Analizler')
    
    # Mevcut menüyü göster
    st.sidebar.markdown(f"**Aktif:** {current_menu}")
    
    # Analiz gösterimi
    if current_menu == "Temel Analizler":
        st.header("Temel Analizler")
        st.markdown("WAF loglarınızın temel analizi ve görselleştirmeleri")
        show_summary_table()
        show_visualizations()
        handle_neuron_details()
        handle_anomaly_detection()
        handle_meta_clustering()
        show_som_validation()
        show_meta_clustering_validation()
        
    elif current_menu == "Gelişmiş Analizler":
        st.header("Gelişmiş Analizler")
        st.markdown("İleri seviye algoritmalar ve detaylı metrik analizleri")
        show_advanced_analysis()
        
    elif current_menu == "Meta Kümeleme":
        show_meta_clustering_analysis()

# Footer
st.markdown("---")
reset_col1, reset_col2, reset_col3 = st.columns([1, 1, 1])

with reset_col2:
    if st.button("Analizi Sıfırla", type="secondary", use_container_width=True):
        reset_session_state()
        st.rerun()

# Sonuç bölümü
with st.expander("Analiz Hakkında", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Bu Analiz Ne Sağlar?
        
        **Güvenlik Analizi:**
        - Anormal log davranışlarını tespit eder
        - Potansiyel güvenlik tehditleri belirler
        - Saldırı desenlerini görselleştirir
        
        **Performans İzleme:**
        - WAF performansını değerlendirir
        - Engelleme oranlarını analiz eder
        - Sistem kaynak kullanımını takip eder
        
        **Raporlama:**
        - Detaylı PDF raporları oluşturur
        - Görsel grafikler ve tablolar sunar
        - Yönetici özetleri hazırlar
        """)
    
    with col2:
        st.markdown("""
        ### Coraza WAF Hakkında
        
        **Coraza WAF Özellikleri:**
        - 🛡️ Modern, açık kaynak WAF
        - ⚡ Yüksek performanslı
        - 🔧 ModSecurity uyumlu
        - 🚀 Go dilinde yazılmış
        
        **Log Format Özellikleri:**
        - 📊 Yapılandırılmış JSON formatı
        - 🔍 Detaylı kural bilgileri
        - 📈 Zengin metrik verisi
        - 🎯 Analiz için optimize
        
        **Neden Sadece Coraza?**
        - Her WAF'ın log formatı farklıdır
        - Alan adları WAF'a göre değişir
        - Coraza için özel preprocessing
        - Optimum analiz sonuçları
        
        > **⚠️ Önemli:** Bu uygulama sadece Coraza WAF log formatı için 
        > tasarlanmıştır. Diğer WAF'lar farklı log yapısına sahiptir.
        """)
