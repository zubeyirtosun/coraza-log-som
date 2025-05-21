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

class PDF(FPDF):
    """
    Temel PDF sınıfı - özel font kullanmadan çalışır
    """
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        # Varsayılan fontları kullan (DejaVu yerine)
        self.add_page()
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
    
    def add_image(self, img_data, x=None, y=None, w=190, h=100):
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
        
        # Sayfaya sığmıyorsa yeni sayfa ekle
        if y + h > self.h - 30:  # Sayfa alt sınırına yaklaşıyorsa
            self.add_page()
            y = self.get_y()
        
        try:
            self.image(img_data, x=x, y=y, w=w, h=h)
            self.ln(h + 10)
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
    buffer = io.BytesIO(base64.b64decode(base64_str))
    buffer.seek(0)
    return buffer

# Görsel verisini (BytesIO veya base64 string) geçici dosyaya dönüştürmek için
def _save_visual_to_temp_file(visual_data, temp_dir, filename):
    if visual_data is None:
        return None
    
    temp_file = os.path.join(temp_dir, filename)
    
    if isinstance(visual_data, io.BytesIO):
        # BytesIO nesnesini geçici dosyaya kaydet
        visual_data.seek(0)
        with open(temp_file, 'wb') as f:
            f.write(visual_data.getvalue())
    elif isinstance(visual_data, str):
        # Base64 string'i geçici dosyaya kaydet
        with open(temp_file, 'wb') as f:
            f.write(base64.b64decode(visual_data))
    
    return temp_file

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
                    plt.figure(figsize=(10, 6))
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
                    
                    table_fig = plt.figure(figsize=(10, 5))
                    ax = table_fig.add_subplot(111)
                    ax.axis('off')
                    ax.table(
                        cellText=table_data,
                        colLabels=table_columns,
                        loc='center',
                        cellLoc='center'
                    )
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'table.png')
                    plt.savefig(temp_file, format='png', bbox_inches='tight')
                    plt.close()
                    
                    # PDF'e ekle
                    pdf.add_image(temp_file)
                except Exception as e:
                    st.warning(f"SOM özet tablosu eklenirken hata: {str(e)}")
            
            # SOM Dağılımı
            if include_basic:
                try:
                    pdf.chapter_title("SOM Dagilimi Analizi")
                    
                    # SOM dağılımını görselleştir
                    plt.figure(figsize=(10, 6))
                    plt.hist2d(st.session_state.df['bmu_x'], st.session_state.df['bmu_y'], bins=st.session_state.grid_size)
                    plt.colorbar(label='Log Sayisi')
                    plt.title('SOM Izgara Dagilimi')
                    plt.xlabel('BMU X')
                    plt.ylabel('BMU Y')
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'som_dist.png')
                    plt.savefig(temp_file, format='png')
                    plt.close()
                    
                    # PDF'e ekle
                    pdf.add_image(temp_file)
                except Exception as e:
                    st.warning(f"SOM dağılımı eklenirken hata: {str(e)}")
            
            # Diğer grafikler için yeni sayfa
            pdf.add_page()
            
            # Niceleme Hatası Dağılımı
            if include_basic:
                try:
                    pdf.chapter_title("Niceleme Hatasi Dagilimi")
                    
                    # Niceleme hatasını görselleştir
                    plt.figure(figsize=(10, 6))
                    plt.hist(st.session_state.df['quantization_error'], bins=50)
                    plt.title('Niceleme Hatasi Dagilimi')
                    plt.xlabel('Niceleme Hatasi')
                    plt.ylabel('Frekans')
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'quant_error.png')
                    plt.savefig(temp_file, format='png')
                    plt.close()
                    
                    # PDF'e ekle
                    pdf.add_image(temp_file)
                except Exception as e:
                    st.warning(f"Niceleme hatası dağılımı eklenirken hata: {str(e)}")
            
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
                        plt.figure(figsize=(10, 6))
                        scatter = plt.scatter(
                            st.session_state.df_meta['bmu_x'], 
                            st.session_state.df_meta['bmu_y'],
                            c=st.session_state.df_meta['meta_cluster'], 
                            cmap='viridis',
                            alpha=0.7
                        )
                        plt.colorbar(scatter, label='Meta Kume')
                        plt.title('Meta Kume Dagilimi')
                        plt.xlabel('BMU X')
                        plt.ylabel('BMU Y')
                        
                        # Geçici dosyaya kaydet
                        temp_file = os.path.join(temp_dir, 'meta_cluster.png')
                        plt.savefig(temp_file, format='png')
                        plt.close()
                        
                        # PDF'e ekle
                        pdf.add_image(temp_file)
                        
                        # Meta küme bazında log sayısı
                        cluster_counts = st.session_state.df_meta['meta_cluster'].value_counts().sort_index()
                        
                        plt.figure(figsize=(10, 6))
                        plt.bar(cluster_counts.index, cluster_counts.values)
                        plt.title('Meta Kume Bazinda Log Sayisi')
                        plt.xlabel('Meta Kume')
                        plt.ylabel('Log Sayisi')
                        
                        # Geçici dosyaya kaydet
                        temp_file = os.path.join(temp_dir, 'cluster_counts.png')
                        plt.savefig(temp_file, format='png')
                        plt.close()
                        
                        # PDF'e ekle
                        pdf.add_image(temp_file)
                except Exception as e:
                    st.warning(f"Meta kümeleme sonuçları eklenirken hata: {str(e)}")
            
            # Gelişmiş analiz sonuçları
            if include_advanced and 'advanced_analysis_results' in st.session_state:
                pdf.add_page()
                pdf.chapter_title("Gelismis Analiz Sonuclari")
                
                advanced_results = st.session_state.advanced_analysis_results
                
                # Optimal Küme Sayısı
                if 'optimal_k' in advanced_results:
                    pdf.chapter_title("Optimal Kume Sayisi Analizi")
                    pdf.chapter_body(f"Bulunan Optimal Kume Sayisi: {advanced_results['optimal_k']}")
                    
                    if 'optimal_k_visualization' in advanced_results:
                        try:
                            # Görselleştirmeyi geçici dosyaya kaydet
                            img_data = _base64_to_buffer(advanced_results['optimal_k_visualization'])
                            temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'optimal_k.png')
                            if temp_file and os.path.exists(temp_file):
                                pdf.add_image(temp_file)
                        except Exception as e:
                            st.warning(f"Optimal küme görselleştirmesi eklenirken hata: {str(e)}")
                
                # Alternatif Kümeleme
                if 'clustering_comparison' in advanced_results:
                    try:
                        pdf.chapter_title("Alternatif Kumeleme Algoritmalari")
                        
                        if 'metrics_df' in advanced_results['clustering_comparison']:
                            metrics_df = advanced_results['clustering_comparison']['metrics_df']
                            pdf.chapter_body("Kumeleme Algoritmalari Karsilastirma Metrikleri:")
                            
                            # Tabloyu görselleştir
                            plt.figure(figsize=(10, 5))
                            plt.axis('off')
                            table_data = []
                            
                            for idx, row in metrics_df.iterrows():
                                table_data.append([
                                    idx,  # Algoritma adı
                                    f"{row['Silüet Skoru']:.4f}" if not pd.isna(row['Silüet Skoru']) else "N/A",
                                    f"{row['Calinski-Harabasz']:.1f}" if not pd.isna(row['Calinski-Harabasz']) else "N/A",
                                    f"{row['Davies-Bouldin']:.4f}" if not pd.isna(row['Davies-Bouldin']) else "N/A",
                                    str(row['Küme Sayısı']) if not pd.isna(row['Küme Sayısı']) else "N/A"
                                ])
                            
                            # Matplotlib tablo
                            table_fig = plt.figure(figsize=(10, 5))
                            ax = table_fig.add_subplot(111)
                            ax.axis('off')
                            table = ax.table(
                                cellText=table_data,
                                colLabels=['Algoritma', 'Silüet Skoru', 'Calinski-Harabasz', 'Davies-Bouldin', 'Küme Sayısı'],
                                loc='center',
                                cellLoc='center'
                            )
                            
                            table.auto_set_font_size(False)
                            table.set_fontsize(10)
                            table.scale(1, 1.5)
                            
                            # Tabloyu dosyaya kaydet
                            temp_file = os.path.join(temp_dir, 'clustering_table.png')
                            plt.savefig(temp_file, format='png', bbox_inches='tight')
                            plt.close()
                            
                            if os.path.exists(temp_file):
                                pdf.add_image(temp_file)
                        
                        # Algoritma görselleştirmeleri
                        if 'visualizations' in advanced_results['clustering_comparison']:
                            visualizations = advanced_results['clustering_comparison']['visualizations']
                            
                            for algo_name, viz_data in visualizations.items():
                                try:
                                    img_data = _base64_to_buffer(viz_data)
                                    temp_file = _save_visual_to_temp_file(img_data, temp_dir, f'clustering_{algo_name}.png')
                                    if temp_file and os.path.exists(temp_file):
                                        pdf.add_image(temp_file)
                                except Exception as e:
                                    st.warning(f"{algo_name} görselleştirmesi eklenirken hata: {str(e)}")
                    except Exception as e:
                        st.warning(f"Kümeleme karşılaştırması eklenirken hata: {str(e)}")
                
                # Kümeleme Stabilitesi
                if 'stability_analysis' in advanced_results:
                    try:
                        pdf.add_page()
                        pdf.chapter_title("Kumeleme Stabilitesi Analizi")
                        
                        stability = advanced_results['stability_analysis']
                        pdf.chapter_body(f"Ortalama Stabilite Skoru: {stability['stability_score']:.4f}")
                        pdf.chapter_body("(Daha yüksek değer daha tutarlı kümeleme anlamına gelir)")
                        
                        if 'visualization' in stability:
                            img_data = _base64_to_buffer(stability['visualization'])
                            temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'stability.png')
                            if temp_file and os.path.exists(temp_file):
                                pdf.add_image(temp_file)
                    except Exception as e:
                        st.warning(f"Stabilite analizi eklenirken hata: {str(e)}")
                
                # Boyut İndirgeme
                if 'dimensionality_reduction' in advanced_results:
                    try:
                        pdf.add_page()
                        pdf.chapter_title("Boyut Indirgeme Analizi")
                        
                        dim_reduction = advanced_results['dimensionality_reduction']
                        
                        for method, data in dim_reduction.items():
                            if method not in ['pca_explained_variance', 'visualizations']:
                                pdf.chapter_body(f"{method} Analizi:")
                                
                                if 'visualizations' in dim_reduction and method in dim_reduction['visualizations']:
                                    try:
                                        img_data = _base64_to_buffer(dim_reduction['visualizations'][method])
                                        temp_file = _save_visual_to_temp_file(img_data, temp_dir, f'dim_reduction_{method}.png')
                                        if temp_file and os.path.exists(temp_file):
                                            pdf.add_image(temp_file)
                                    except Exception as e:
                                        st.warning(f"{method} görselleştirmesi eklenirken hata: {str(e)}")
                        
                        # PCA varyans açıklama oranı
                        if 'pca_explained_variance' in dim_reduction:
                            pdf.chapter_body(f"PCA Kümülatif Varyans Açıklama Oranı: {dim_reduction['pca_explained_variance']:.2%}")
                    except Exception as e:
                        st.warning(f"Boyut indirgeme analizi eklenirken hata: {str(e)}")
                
                # Çapraz Doğrulama
                if 'cross_validation' in advanced_results:
                    try:
                        pdf.add_page()
                        pdf.chapter_title("Capraz Dogrulama Sonuclari")
                        
                        cv_results = advanced_results['cross_validation']
                        pdf.chapter_body(f"Ortalama Silüet Skoru: {cv_results['mean_silhouette']:.4f}")
                        pdf.chapter_body(f"Standart Sapma: {cv_results['std_silhouette']:.4f}")
                        
                        if 'visualization' in cv_results:
                            try:
                                img_data = _base64_to_buffer(cv_results['visualization'])
                                temp_file = _save_visual_to_temp_file(img_data, temp_dir, 'cross_validation.png')
                                if temp_file and os.path.exists(temp_file):
                                    pdf.add_image(temp_file)
                            except Exception as e:
                                st.warning(f"Çapraz doğrulama görselleştirmesi eklenirken hata: {str(e)}")
                    except Exception as e:
                        st.warning(f"Çapraz doğrulama sonuçları eklenirken hata: {str(e)}")
            
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