import streamlit as st
from fpdf import FPDF
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import tempfile
from datetime import datetime
import shutil
import traceback
from PIL import Image

class PDF(FPDF):
    """
    Temel PDF sınıfı - özel font kullanmadan çalışır
    """
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        # Varsayılan fontları kullan (DejaVu yerine)
        self.add_page()
        # Daha yüksek çözünürlük için dpi ayarı
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        st.info("PDF oluşturucu başlatıldı (varsayılan fontlar kullanılıyor)")

    def header(self):
        # Başlık
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'SOM ve Meta Kumeleme Analizi Raporu', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        # Sayfa numarası
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image(self, img_data, x=None, y=None, w=190, h=None):
        """Görseli PDF'e ekler, otomatik yükseklik hesaplaması yapabilir"""
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
        
        # Görsel boyutlarını kontrol et
        if os.path.exists(img_data):
            try:
                # PIL ile görsel bilgilerini al
                img = Image.open(img_data)
                img_w, img_h = img.size
                
                # Oranı koru
                if h is None and w is not None:
                    h = w * img_h / img_w
            except Exception:
                # PIL işlemi başarısız olursa varsayılan h değerini kullan
                if h is None:
                    h = 100
        else:
            # Dosya bulunamazsa varsayılan değerleri kullan
            if h is None:
                h = 100
        
        # Sayfaya sığmıyorsa yeni sayfa ekle
        if y + h > self.h - 20:  # Sayfa alt sınırına yaklaşıyorsa
            self.add_page()
            y = self.get_y()
        
        try:
            self.image(img_data, x=x, y=y, w=w, h=h)
            self.ln(h + 10)  # Görsel sonrası boşluk ekle
        except Exception as e:
            st.warning(f"Görsel eklenirken hata: {str(e)}")
            self.ln(10)

    def add_table(self, data, headers=None):
        # Tablodaki her sütunun genişliğini hesapla
        col_width = self.w / (len(data[0]) + 1)
        
        # Başlıklar varsa ekle
        if headers:
            self.set_font('Arial', 'B', 10)
            for header in headers:
                self.cell(col_width, 7, str(header), 1, 0, 'C')
            self.ln()
        
        # Verileri ekle
        self.set_font('Arial', '', 10)
        for row in data:
            for item in row:
                self.cell(col_width, 7, str(item), 1, 0, 'C')
            self.ln()
        self.ln(5)

# BytesIO'yu base64 string'e dönüştürmek için yardımcı fonksiyon
def _buffer_to_base64(buffer):
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return img_str

# Base64 string'i BytesIO'ya dönüştürmek için yardımcı fonksiyon
def _base64_to_buffer(base64_str):
    try:
        if isinstance(base64_str, str):
            buffer = io.BytesIO(base64.b64decode(base64_str))
            buffer.seek(0)
            return buffer
        elif isinstance(base64_str, io.BytesIO):
            # Zaten BytesIO ise direkt döndür
            base64_str.seek(0)
            return base64_str
        else:
            st.warning(f"Desteklenmeyen veri türü: {type(base64_str)}")
            return None
    except Exception as e:
        st.warning(f"Base64 çözümleme hatası: {str(e)}")
        return None

# Görsel verisini (BytesIO veya base64 string) geçici dosyaya dönüştürmek için
def _save_visual_to_temp_file(visual_data, temp_dir, filename):
    """
    Görsel verisini geçici dosyaya kaydeder ve yolu döndürür
    Hem BytesIO hem de base64 string formatlarını destekler
    """
    if visual_data is None:
        st.warning(f"'{filename}' için görsel verisi bulunamadı.")
        return None
    
    temp_file = os.path.join(temp_dir, filename)
    
    try:
        if isinstance(visual_data, io.BytesIO):
            # BytesIO nesnesini geçici dosyaya kaydet
            visual_data.seek(0)
            with open(temp_file, 'wb') as f:
                f.write(visual_data.getvalue())
        elif isinstance(visual_data, str):
            # Base64 string'i geçici dosyaya kaydet
            try:
                with open(temp_file, 'wb') as f:
                    f.write(base64.b64decode(visual_data))
            except Exception as e:
                st.warning(f"Base64 çözümleme hatası: {str(e)}")
                return None
        else:
            st.warning(f"Desteklenmeyen görsel veri türü: {type(visual_data)}")
            return None
        
        # Dosya doğru şekilde oluşturuldu mu kontrol et
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            return temp_file
        else:
            st.warning(f"Geçerli bir görsel dosyası oluşturulamadı: {filename}")
            return None
    except Exception as e:
        st.warning(f"Görsel dosyası kaydedilirken hata: {str(e)}")
        return None

# Daha iyi görsel oluşturma için stili ayarla
def setup_plot_style():
    """Matplotlib grafik stilini rapor için optimize eder"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'

def create_pdf_report(title="SOM ve Meta Kumeleme Analizi Raporu", include_basic=True, include_advanced=True):
    """
    Tüm analizleri içeren PDF raporu oluşturur
    """
    if st.session_state.som is None or st.session_state.X is None:
        st.warning("SOM modeli henüz eğitilmemiş, rapor oluşturulamıyor.")
        return None
    
    temp_dir = None
    try:
        # Geçici dosya dizini oluştur
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, "som_report.pdf")
        
        # Grafik stillerini ayarla
        setup_plot_style()
        
        try:
            # PDF oluştur
            st.info("PDF raporu oluşturuluyor...")
            pdf = PDF()
            
            # Rapor bilgileri
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.chapter_body(f"Olusturulma Tarihi: {timestamp}")
            
            # Veri boyutunu kontrol et
            if 'df' in st.session_state and st.session_state.df is not None:
                pdf.chapter_body(f"Veri Sayisi: {len(st.session_state.df)}")
            else:
                pdf.chapter_body("Veri Sayisi: Bilinmiyor")
            
            # Grid boyutunu kontrol et
            if 'grid_size' in st.session_state and st.session_state.grid_size is not None:
                pdf.chapter_body(f"SOM Izgara Boyutu: {st.session_state.grid_size}x{st.session_state.grid_size}")
            else:
                pdf.chapter_body("SOM Izgara Boyutu: Bilinmiyor")
            
            pdf.ln(10)
            
            # SOM özet tablosu
            if include_basic and 'summary_df' in st.session_state and st.session_state.summary_df is not None:
                try:
                    pdf.chapter_title("SOM Ozet Tablosu")
                    
                    # Tabloyu görselleştir
                    summary_df_head = st.session_state.summary_df.head(20)  # İlk 20 satır
                    table_data = []
                    table_columns = ['Noron', 'Engellenme Orani', 'En Sik URI', 'Ort. Hata', 'Log Sayisi']
                    for _, row in summary_df_head.iterrows():
                        table_data.append([
                            row['Nöron'], 
                            f"{row['Engellenme Oranı']:.4f}", 
                            str(row['En Sık URI'])[:30], 
                            f"{row['Ort. Hata']:.4f}", 
                            str(row['Log Sayısı'])
                        ])
                    
                    # Matplotlib tablo oluştur
                    table_fig = plt.figure(figsize=(10, min(8, len(table_data) * 0.4 + 1)))
                    ax = table_fig.add_subplot(111)
                    ax.axis('off')
                    
                    # Tabloyu oluştur
                    table = ax.table(
                        cellText=table_data,
                        colLabels=table_columns,
                        loc='center',
                        cellLoc='center'
                    )
                    
                    # Tabloyu güzelleştir
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    
                    # Başlık satırını renklendir
                    for (i, j), cell in table.get_celld().items():
                        if i == 0:  # Başlık satırı
                            cell.set_facecolor('#4e73df')
                            cell.set_text_props(color='white')
                        elif i % 2 == 0:  # Çift satırlar
                            cell.set_facecolor('#f8f9fc')
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'table.png')
                    plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    # PDF'e ekle
                    if os.path.exists(temp_file):
                        pdf.add_image(temp_file)
                    else:
                        st.warning("SOM özet tablosu dosyası oluşturulamadı.")
                except Exception as e:
                    st.warning(f"SOM özet tablosu eklenirken hata: {str(e)}")
            
            # SOM Dağılımı
            if include_basic:
                try:
                    pdf.chapter_title("SOM Dagilimi Analizi")
                    pdf.chapter_body("Bu bölümde SOM ızgarasının 3 farklı görselleştirmesi sunulmuştur:")
                    pdf.chapter_body("1. Scatter Plot: Web arayüzündekiyle aynı görünüm - her nokta bir log kaydını temsil eder")
                    pdf.chapter_body("2. Histogram: Log yoğunluğunu gösterir - koyu renkler daha fazla log içeren nöronları gösterir")
                    pdf.chapter_body("3. U-Matrix: Nöronlar arası mesafeleri gösterir - koyu renkler farklı davranış bölgelerini ayırır")
                    
                    # 1. Scatter Plot (Web'tekiyle aynı)
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        st.session_state.df['bmu_x'], 
                        st.session_state.df['bmu_y'], 
                        c=st.session_state.df['quantization_error'],
                        cmap='viridis',
                        alpha=0.6,
                        s=20
                    )
                    plt.colorbar(scatter, label='Niceleme Hatasi')
                    plt.title('SOM Izgara Dagilimi (Scatter Plot)', fontsize=16, pad=20)
                    plt.xlabel('BMU X', fontsize=12)
                    plt.ylabel('BMU Y', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'som_scatter.png')
                    plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    # PDF'e ekle
                    if os.path.exists(temp_file):
                        pdf.add_image(temp_file)
                    
                    # 2. Histogram (Log yoğunluğu)
                    plt.figure(figsize=(10, 8))
                    histogram = plt.hist2d(
                        st.session_state.df['bmu_x'], 
                        st.session_state.df['bmu_y'], 
                        bins=st.session_state.grid_size,
                        cmap='viridis'
                    )
                    plt.colorbar(histogram[3], label='Log Sayisi')
                    plt.title('SOM Izgara Yogunluk Haritasi (Histogram)', fontsize=16, pad=20)
                    plt.xlabel('BMU X', fontsize=12)
                    plt.ylabel('BMU Y', fontsize=12)
                    plt.tight_layout()
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'som_histogram.png')
                    plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    # PDF'e ekle
                    if os.path.exists(temp_file):
                        pdf.add_image(temp_file)
                    
                    # 3. SOM Distance Map (Nöron mesafe haritası)
                    try:
                        plt.figure(figsize=(10, 8))
                        distance_map = st.session_state.som.distance_map().T
                        im = plt.imshow(distance_map, cmap='viridis')
                        plt.colorbar(im, label='Noron Mesafesi')
                        plt.title('SOM Mesafe Haritasi (U-Matrix)', fontsize=16, pad=20)
                        plt.xlabel('Izgara X', fontsize=12)
                        plt.ylabel('Izgara Y', fontsize=12)
                        plt.tight_layout()
                        
                        # Geçici dosyaya kaydet
                        temp_file = os.path.join(temp_dir, 'som_distance_map.png')
                        plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                        plt.close()
                        
                        # PDF'e ekle
                        if os.path.exists(temp_file):
                            pdf.add_image(temp_file)
                    except Exception as e:
                        st.warning(f"SOM mesafe haritası eklenirken hata: {str(e)}")
                    
                    # 4. Niceleme Hatası Dağılımı (aynı bölümde)
                    try:
                        pdf.add_page()
                        pdf.chapter_title("Niceleme Hatasi Dagilimi")
                        pdf.chapter_body("Niceleme hatası, her log kaydının en yakın nörona ne kadar uzak olduğunu gösterir.")
                        pdf.chapter_body("Yüksek değerler potansiyel anomalileri işaret eder.")
                        
                        # Niceleme hatasını görselleştir
                        plt.figure(figsize=(10, 6))
                        
                        # Veri kontrolü 
                        if 'quantization_error' in st.session_state.df.columns:
                            plt.hist(st.session_state.df['quantization_error'], bins=50, color='#1f77b4', alpha=0.8)
                            plt.axvline(
                                st.session_state.df['quantization_error'].mean(), 
                                color='red', 
                                linestyle='dashed', 
                                linewidth=2, 
                                label=f'Ortalama: {st.session_state.df["quantization_error"].mean():.4f}'
                            )
                            plt.legend()
                            plt.title('Niceleme Hatasi Dagilimi', fontsize=16, pad=20)
                            plt.xlabel('Niceleme Hatasi', fontsize=12)
                            plt.ylabel('Frekans', fontsize=12)
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            
                            # Geçici dosyaya kaydet
                            temp_file = os.path.join(temp_dir, 'quant_error.png')
                            plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                            plt.close()
                            
                            # PDF'e ekle
                            if os.path.exists(temp_file):
                                pdf.add_image(temp_file)
                            else:
                                st.warning("Niceleme hatası dağılımı dosyası oluşturulamadı.")
                        else:
                            st.warning("Niceleme hatası verisi bulunamadı.")
                    except Exception as e:
                        st.warning(f"Niceleme hatası dağılımı eklenirken hata: {str(e)}")
                    
                except Exception as e:
                    st.warning(f"SOM dağılımı eklenirken hata: {str(e)}")
            
            # Meta Kümeleme Sonuçları
            if include_basic and 'meta_clusters' in st.session_state and st.session_state.meta_clusters is not None:
                try:
                    pdf.add_page()
                    pdf.chapter_title("Meta Kumeleme Sonuclari")
                    
                    # Meta küme sayısını al
                    if 'optimal_k' in st.session_state and st.session_state.optimal_k is not None:
                        pdf.chapter_body(f"Optimal Kume Sayisi: {st.session_state.optimal_k} (Otomatik Belirlendi)")
                    elif 'df_meta' in st.session_state and st.session_state.df_meta is not None:
                        n_clusters = st.session_state.df_meta['meta_cluster'].nunique()
                        pdf.chapter_body(f"Kume Sayisi: {n_clusters} (Manuel Secildi)")
                    
                    # Meta küme dağılımını görselleştir
                    if 'df_meta' in st.session_state and st.session_state.df_meta is not None:
                        plt.figure(figsize=(10, 8))
                        
                        # Renklendirme için özel bir renk paleti kullan
                        n_clusters = st.session_state.df_meta['meta_cluster'].nunique()
                        cmap = plt.cm.get_cmap('viridis', n_clusters)
                        
                        scatter = plt.scatter(
                            st.session_state.df_meta['bmu_x'], 
                            st.session_state.df_meta['bmu_y'],
                            c=st.session_state.df_meta['meta_cluster'], 
                            cmap=cmap,
                            alpha=0.8,
                            edgecolors='w',
                            linewidth=0.5,
                            s=70
                        )
                        
                        cbar = plt.colorbar(scatter, label='Meta Kume')
                        cbar.set_ticks(range(n_clusters))
                        cbar.set_ticklabels(range(n_clusters))
                        
                        plt.title('Meta Kume Dagilimi', fontsize=16, pad=20)
                        plt.xlabel('BMU X', fontsize=12)
                        plt.ylabel('BMU Y', fontsize=12)
                        plt.tight_layout()
                        
                        # Geçici dosyaya kaydet
                        temp_file = os.path.join(temp_dir, 'meta_cluster.png')
                        plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                        plt.close()
                        
                        # PDF'e ekle
                        if os.path.exists(temp_file):
                            pdf.add_image(temp_file)
                        else:
                            st.warning("Meta küme dağılımı dosyası oluşturulamadı.")
                        
                        # Meta küme bazında log sayısı
                        cluster_counts = st.session_state.df_meta['meta_cluster'].value_counts().sort_index()
                        
                        plt.figure(figsize=(10, 6))
                        bars = plt.bar(
                            cluster_counts.index, 
                            cluster_counts.values,
                            color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))),
                            alpha=0.8,
                            edgecolor='black',
                            linewidth=0.5
                        )
                        
                        # Çubukların üzerine değerleri yazdır
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(
                                bar.get_x() + bar.get_width()/2., 
                                height + 5,
                                f'{int(height)}',
                                ha='center', 
                                va='bottom',
                                fontsize=10
                            )
                        
                        plt.title('Meta Kume Bazinda Log Sayisi', fontsize=16, pad=20)
                        plt.xlabel('Meta Kume', fontsize=12)
                        plt.ylabel('Log Sayisi', fontsize=12)
                        plt.xticks(cluster_counts.index)
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        
                        # Geçici dosyaya kaydet
                        temp_file = os.path.join(temp_dir, 'cluster_counts.png')
                        plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                        plt.close()
                        
                        # PDF'e ekle
                        if os.path.exists(temp_file):
                            pdf.add_image(temp_file)
                        else:
                            st.warning("Küme dağılımı grafiği dosyası oluşturulamadı.")
                except Exception as e:
                    st.warning(f"Meta kümeleme sonuçları eklenirken hata: {str(e)}")
                    st.warning(traceback.format_exc())
            
            # Gelişmiş analiz sonuçları
            if include_advanced and 'advanced_analysis_results' in st.session_state:
                try:
                    pdf.add_page()
                    pdf.chapter_title("Gelismis Analiz Sonuclari")
                    
                    advanced_results = st.session_state.advanced_analysis_results
                    
                    # Optimal Küme Sayısı
                    if 'optimal_k' in advanced_results:
                        try:
                            pdf.chapter_title("Optimal Kume Sayisi Analizi")
                            pdf.chapter_body(f"Bulunan Optimal Kume Sayisi: {advanced_results['optimal_k']}")
                            
                            if 'optimal_k_visualization' in advanced_results:
                                try:
                                    # Görselleştirmeyi geçici dosyaya kaydet
                                    img_data = _base64_to_buffer(advanced_results['optimal_k_visualization'])
                                    temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'optimal_k.png')
                                    if temp_file and os.path.exists(temp_file):
                                        pdf.add_image(temp_file, w=180)
                                except Exception as e:
                                    st.warning(f"Optimal küme görselleştirmesi eklenirken hata: {str(e)}")
                        except Exception as e:
                            st.warning(f"Optimal küme sayısı analizi eklenirken hata: {str(e)}")
                    
                    # Alternatif Kümeleme
                    if 'clustering_comparison' in advanced_results:
                        try:
                            pdf.chapter_title("Alternatif Kumeleme Algoritmalari")
                            
                            if 'metrics_df' in advanced_results['clustering_comparison']:
                                try:
                                    metrics_df = advanced_results['clustering_comparison']['metrics_df']
                                    pdf.chapter_body("Kumeleme Algoritmalari Karsilastirma Metrikleri:")
                                    
                                    # Tabloyu görselleştir
                                    plt.figure(figsize=(10, 5))
                                    plt.axis('off')
                                    table_data = []
                                    
                                    for idx, row in metrics_df.iterrows():
                                        # Bazı metrikler NaN olabilir, kontrol ediyoruz
                                        siluet = row.get('Silüet Skoru', float('nan'))
                                        calinski = row.get('Calinski-Harabasz', float('nan'))
                                        davies = row.get('Davies-Bouldin', float('nan'))
                                        kume_sayisi = row.get('Küme Sayısı', float('nan'))
                                        
                                        # Daha güvenli formatlama
                                        siluet_str = f"{siluet:.4f}" if not pd.isna(siluet) else "N/A"
                                        calinski_str = f"{calinski:.1f}" if not pd.isna(calinski) else "N/A"
                                        davies_str = f"{davies:.4f}" if not pd.isna(davies) else "N/A"
                                        kume_str = str(int(kume_sayisi)) if not pd.isna(kume_sayisi) else "N/A"
                                        
                                        table_data.append([
                                            idx,  # Algoritma adı
                                            siluet_str,
                                            calinski_str,
                                            davies_str,
                                            kume_str
                                        ])
                                    
                                    # Matplotlib tablo
                                    table_fig = plt.figure(figsize=(10, min(6, len(table_data) * 0.5 + 1)))
                                    ax = table_fig.add_subplot(111)
                                    ax.axis('off')
                                    
                                    # Eğer hiç veri yoksa, mesajla bildir
                                    if not table_data:
                                        ax.text(0.5, 0.5, "Karşılaştırılabilir kümeleme sonucu bulunamadı", 
                                                ha='center', va='center', fontsize=12, color='gray')
                                        
                                        # Geçici dosyaya kaydet
                                        temp_file = os.path.join(temp_dir, 'clustering_table.png')
                                        plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                                        plt.close()
                                        
                                        if os.path.exists(temp_file):
                                            pdf.add_image(temp_file)
                                    else:
                                        table = ax.table(
                                            cellText=table_data,
                                            colLabels=['Algoritma', 'Silüet Skoru', 'Calinski-Harabasz', 'Davies-Bouldin', 'Küme Sayısı'],
                                            loc='center',
                                            cellLoc='center'
                                        )
                                        
                                        # Tabloyu güzelleştir
                                        table.auto_set_font_size(False)
                                        table.set_fontsize(10)
                                        table.scale(1, 1.5)
                                        
                                        # Başlık satırını renklendir
                                        for (i, j), cell in table.get_celld().items():
                                            if i == 0:  # Başlık satırı
                                                cell.set_facecolor('#4e73df')
                                                cell.set_text_props(color='white')
                                            elif i % 2 == 0:  # Çift satırlar
                                                cell.set_facecolor('#f8f9fc')
                                        
                                        # Geçici dosyaya kaydet
                                        temp_file = os.path.join(temp_dir, 'clustering_table.png')
                                        plt.savefig(temp_file, format='png', bbox_inches='tight', dpi=150)
                                        plt.close()
                                        
                                        if os.path.exists(temp_file):
                                            pdf.add_image(temp_file)
                                except Exception as e:
                                    st.warning(f"Kümeleme metrik tablosu eklenirken hata: {str(e)}")
                            
                            # Algoritma görselleştirmeleri
                            if 'visualizations' in advanced_results['clustering_comparison']:
                                try:
                                    visualizations = advanced_results['clustering_comparison']['visualizations']
                                    
                                    if visualizations:
                                        pdf.add_page()
                                        pdf.chapter_title("Kumeleme Algoritmalari Gorsel Karsilastirmasi")
                                        
                                        for algo_name, viz_data in visualizations.items():
                                            try:
                                                img_data = _base64_to_buffer(viz_data)
                                                temp_file = _save_visual_to_temp_file(img_data, temp_dir, f'clustering_{algo_name}.png')
                                                if temp_file and os.path.exists(temp_file):
                                                    # Algoritma adı ekleyerek güzelleştir
                                                    pdf.chapter_body(f"Algoritma: {algo_name}")
                                                    pdf.add_image(temp_file, w=180)
                                            except Exception as e:
                                                st.warning(f"{algo_name} görselleştirmesi eklenirken hata: {str(e)}")
                                except Exception as e:
                                    st.warning(f"Kümeleme görselleştirmeleri eklenirken hata: {str(e)}")
                        except Exception as e:
                            st.warning(f"Kümeleme karşılaştırması eklenirken hata: {str(e)}")
                    
                    # Kümeleme Stabilitesi
                    if 'stability_analysis' in advanced_results:
                        try:
                            pdf.add_page()
                            pdf.chapter_title("Kumeleme Stabilitesi Analizi")
                            
                            stability = advanced_results['stability_analysis']
                            
                            # Stabilite skoru kontrolü
                            if 'stability_score' in stability:
                                try:
                                    stability_score = float(stability['stability_score'])
                                    pdf.chapter_body(f"Ortalama Stabilite Skoru: {stability_score:.4f}")
                                    pdf.chapter_body("(Daha yüksek değer daha tutarlı kümeleme anlamına gelir)")
                                except (ValueError, TypeError):
                                    pdf.chapter_body(f"Ortalama Stabilite Skoru: {stability['stability_score']} (hesaplanamadı)")
                            
                            if 'visualization' in stability:
                                try:
                                    img_data = _base64_to_buffer(stability['visualization'])
                                    temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'stability.png')
                                    if temp_file and os.path.exists(temp_file):
                                        pdf.add_image(temp_file, w=180)
                                except Exception as e:
                                    st.warning(f"Stabilite görselleştirmesi eklenirken hata: {str(e)}")
                        except Exception as e:
                            st.warning(f"Stabilite analizi eklenirken hata: {str(e)}")
                    
                    # Boyut İndirgeme
                    if 'dimensionality_reduction' in advanced_results:
                        try:
                            pdf.add_page()
                            pdf.chapter_title("Boyut Indirgeme Analizi")
                            
                            dim_reduction = advanced_results['dimensionality_reduction']
                            
                            # PCA varyans açıklama oranı
                            if 'pca_explained_variance' in dim_reduction:
                                try:
                                    pca_variance = float(dim_reduction['pca_explained_variance'])
                                    pdf.chapter_body(f"PCA Kümülatif Varyans Açıklama Oranı: {pca_variance:.2%}")
                                except (ValueError, TypeError):
                                    pdf.chapter_body(f"PCA Kümülatif Varyans Açıklama Oranı: {dim_reduction['pca_explained_variance']} (hesaplanamadı)")
                            
                            # Boyut indirgeme görselleştirmeleri
                            if 'visualizations' in dim_reduction:
                                visualizations = dim_reduction['visualizations']
                                for method_name, viz_data in visualizations.items():
                                    try:
                                        pdf.chapter_body(f"{method_name} Analizi:")
                                        img_data = _base64_to_buffer(viz_data)
                                        temp_file = _save_visual_to_temp_file(img_data, temp_dir, f'dim_reduction_{method_name}.png')
                                        if temp_file and os.path.exists(temp_file):
                                            pdf.add_image(temp_file, w=180)
                                    except Exception as e:
                                        st.warning(f"{method_name} görselleştirmesi eklenirken hata: {str(e)}")
                        except Exception as e:
                            st.warning(f"Boyut indirgeme analizi eklenirken hata: {str(e)}")
                    
                    # Çapraz Doğrulama
                    if 'cross_validation' in advanced_results:
                        try:
                            pdf.add_page()
                            pdf.chapter_title("Capraz Dogrulama Sonuclari")
                            
                            cv_results = advanced_results['cross_validation']
                            
                            # CV sonuçlarını ekle
                            if 'mean_silhouette' in cv_results and 'std_silhouette' in cv_results:
                                try:
                                    mean_silhouette = float(cv_results['mean_silhouette'])
                                    std_silhouette = float(cv_results['std_silhouette'])
                                    pdf.chapter_body(f"Ortalama Silüet Skoru: {mean_silhouette:.4f}")
                                    pdf.chapter_body(f"Standart Sapma: {std_silhouette:.4f}")
                                except (ValueError, TypeError):
                                    pdf.chapter_body("Silüet skorları hesaplanamadı.")
                            
                            if 'visualization' in cv_results:
                                try:
                                    img_data = _base64_to_buffer(cv_results['visualization'])
                                    temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'cross_validation.png')
                                    if temp_file and os.path.exists(temp_file):
                                        pdf.add_image(temp_file, w=180)
                                except Exception as e:
                                    st.warning(f"Çapraz doğrulama görselleştirmesi eklenirken hata: {str(e)}")
                        except Exception as e:
                            st.warning(f"Çapraz doğrulama sonuçları eklenirken hata: {str(e)}")
                except Exception as e:
                    st.warning(f"Gelişmiş analiz sonuçları eklenirken hata: {str(e)}")
                    st.warning(traceback.format_exc())
            
            # PDF'i oluştur
            try:
                st.info("PDF dosyası kaydediliyor...")
                pdf.output(temp_pdf_path)
                
                # PDF dosyasını kontrol et
                if not os.path.exists(temp_pdf_path) or os.path.getsize(temp_pdf_path) < 100:
                    st.error("PDF dosyası oluşturulamadı veya dosya boyutu çok küçük.")
                    return None
                    
                # PDF dosyasını oku ve BytesIO'ya aktar
                with open(temp_pdf_path, 'rb') as pdf_file:
                    pdf_data = pdf_file.read()
                
                pdf_bytes = io.BytesIO(pdf_data)
                pdf_bytes.seek(0)
                
                st.success("PDF raporu başarıyla oluşturuldu!")
                return pdf_bytes
            except Exception as e:
                st.error(f"PDF oluşturma işlemi başarısız: {str(e)}")
                st.error(traceback.format_exc())
                return None
            
        except Exception as e:
            st.error(f"PDF oluşturma hatası: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    except Exception as e:
        st.error(f"PDF oluşturma işlemi başlatılamadı: {str(e)}")
        st.error(traceback.format_exc())
        return None
    finally:
        # Temizleme işlemi
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                st.warning(f"Geçici dosyalar temizlenirken hata: {str(e)}")

def get_pdf_download_link(pdf_output, filename="SOM_Analiz_Raporu.pdf"):
    """PDF'i indirilebilir link olarak döndürür"""
    if pdf_output is None:
        return None
    
    b64 = base64.b64encode(pdf_output.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Raporu İndir</a>'
    return href 