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
    Basitleştirilmiş PDF sınıfı - standart fontları kullanır
    """
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        # Türkçe karakter desteği için encoding ayarı
        self.add_page()

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
        # Turkce karakterleri ASCII'ye donustur
        title = self._sanitize_text(title)
        self.set_font('Arial', 'B', 14)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def chapter_body(self, body):
        # Turkce karakterleri ASCII'ye donustur
        body = self._sanitize_text(body)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def _sanitize_text(self, text):
        # Turkce karakterleri ASCII esdeğerlerine cevir
        replacements = {
            'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 
            'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 
            'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text
        
    def add_image(self, img_data, x=None, y=None, w=190, h=100):
        if x is None:
            x = self.get_x()
        if y is None:
            y = self.get_y()
        
        # Sayfaya sığmıyorsa yeni sayfa ekle
        if y + h > self.h - 30:  # Sayfa alt sınırına yaklaşıyorsa
            self.add_page()
            y = self.get_y()
        
        self.image(img_data, x=x, y=y, w=w, h=h)
        self.ln(h + 10)

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
    
    try:
        # Geçici dosya dizini oluştur
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, "som_report.pdf")
        
        try:
            # PDF oluştur - basit yapı kullanalım, font sorunu yaşamamak için
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
            if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
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
            
            # SOM Dağılımı
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
            
            # Diğer grafikler için yeni sayfa
            pdf.add_page()
            
            # Niceleme Hatası Dağılımı
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
            
            # Meta Kümeleme Sonuçları
            if 'meta_clusters' in st.session_state and st.session_state.meta_clusters is not None:
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
            
            # Gelişmiş analizleri ekle (kullanıcı tarafından yapılmışsa)
            if include_advanced:
                # 1. Optimal Küme Sayısı Analizi
                if 'optimal_k_results' in st.session_state and st.session_state.optimal_k_results is not None:
                    pdf.add_page()
                    pdf.chapter_title("Optimal Kume Sayisi Analizi")
                    
                    results = st.session_state.optimal_k_results
                    
                    if 'optimal_k' in results:
                        pdf.chapter_body(f"Onerilen K degeri: {results['optimal_k']}")
                    
                    # Özet tablo
                    metrics_table = {
                        "Metrik": ["Dirsek Yontemi", "Siluet Skoru", "Calinski-Harabasz", "Davies-Bouldin"],
                        "Optimal K": [
                            results.get('elbow_k', 'Belirsiz'),
                            results.get('silhouette_k', 'Belirsiz'),
                            results.get('calinski_k', 'Belirsiz'),
                            results.get('davies_k', 'Belirsiz')
                        ]
                    }
                    
                    # Tabloyu görselleştir
                    plt.figure(figsize=(8, 4))
                    ax = plt.subplot(111)
                    ax.axis('off')
                    table = ax.table(
                        cellText=[[str(v) for v in metrics_table["Optimal K"]]],
                        rowLabels=["Optimal K"],
                        colLabels=metrics_table["Metrik"],
                        loc='center',
                        cellLoc='center'
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.2, 1.5)
                    
                    # Geçici dosyaya kaydet
                    temp_file = os.path.join(temp_dir, 'optimal_k_table.png')
                    plt.savefig(temp_file, format='png', bbox_inches='tight')
                    plt.close()
                    
                    # PDF'e ekle
                    pdf.add_image(temp_file, w=180, h=60)
                    
                    # Görselleştirme varsa ekle
                    if 'visualization' in results or 'visualization_base64' in results:
                        vis_key = 'visualization_base64' if 'visualization_base64' in results else 'visualization'
                        vis_data = results[vis_key]
                        
                        # Görselleştirmeyi geçici dosyaya kaydet
                        temp_file = _save_visual_to_temp_file(vis_data, temp_dir, 'optimal_k_viz.png')
                        
                        if temp_file:
                            pdf.add_image(temp_file)
                
                # 2. Alternatif Kümeleme Algoritmaları
                if 'alternative_clustering_results' in st.session_state and st.session_state.alternative_clustering_results is not None:
                    pdf.add_page()
                    pdf.chapter_title("Alternatif Kumeleme Algoritmalari")
                    
                    comparison_results = st.session_state.alternative_clustering_results
                    
                    if 'metrics' in comparison_results:
                        metrics_df = comparison_results['metrics']
                        
                        # Görselleştirme varsa ekle
                        if 'visualizations' in comparison_results and 'Metrikler' in comparison_results['visualizations']:
                            # Görselleştirmeyi geçici dosyaya kaydet
                            temp_file = _save_visual_to_temp_file(
                                comparison_results['visualizations']['Metrikler'], 
                                temp_dir, 
                                'algo_metrics.png'
                            )
                            
                            if temp_file:
                                pdf.add_image(temp_file)
                        
                        # Algoritma görselleştirmelerini ekle
                        if 'visualizations' in comparison_results:
                            algo_names = [name for name in comparison_results['visualizations'].keys() if name != 'Metrikler']
                            
                            for algo_name in algo_names:
                                pdf.chapter_body(f"Algoritma: {algo_name}")
                                
                                # Görselleştirmeyi geçici dosyaya kaydet
                                temp_file = _save_visual_to_temp_file(
                                    comparison_results['visualizations'][algo_name], 
                                    temp_dir, 
                                    f'algo_{algo_name}.png'
                                )
                                
                                if temp_file:
                                    pdf.add_image(temp_file)
                
                # 3. Kümeleme Stabilitesi
                if 'stability_results' in st.session_state and st.session_state.stability_results is not None:
                    pdf.add_page()
                    pdf.chapter_title("Kumeleme Stabilitesi Analizi")
                    
                    stability = st.session_state.stability_results
                    
                    if 'stability_score' in stability:
                        pdf.chapter_body(f"Stabilite Skoru: {stability['stability_score']:.4f}")
                        pdf.chapter_body("Not: 1.0'a yakin degerler daha stabil kumelemeyi gosterir.")
                    
                    # Stabilite matrisini ekle
                    if 'stability_matrix' in stability:
                        # Görselleştirmeyi geçici dosyaya kaydet
                        temp_file = None
                        
                        if 'visualization' in stability:
                            temp_file = _save_visual_to_temp_file(
                                stability['visualization'], 
                                temp_dir, 
                                'stability_viz.png'
                            )
                        
                        if temp_file:
                            pdf.add_image(temp_file)
                
                # 4. Boyut İndirgeme
                if 'dimensionality_reduction_results' in st.session_state and st.session_state.dimensionality_reduction_results is not None:
                    pdf.add_page()
                    pdf.chapter_title("Boyut Indirgeme Analizleri")
                    
                    dr_results = st.session_state.dimensionality_reduction_results
                    
                    for method_name, visual_data in dr_results.items():
                        pdf.chapter_body(f"Metot: {method_name}")
                        
                        # Görselleştirmeyi geçici dosyaya kaydet
                        temp_file = _save_visual_to_temp_file(
                            visual_data, 
                            temp_dir, 
                            f'dr_{method_name}.png'
                        )
                        
                        if temp_file:
                            pdf.add_image(temp_file)
                
                # 5. Çapraz Doğrulama
                if 'cross_validation_results' in st.session_state and st.session_state.cross_validation_results is not None:
                    pdf.add_page()
                    pdf.chapter_title("Capraz Dogrulama Analizi")
                    
                    cv_results = st.session_state.cross_validation_results
                    
                    if 'avg_silhouette' in cv_results:
                        pdf.chapter_body(f"Ortalama Siluet Skoru: {cv_results['avg_silhouette']:.4f}")
                    
                    # Görselleştirme varsa ekle
                    if 'visualization' in cv_results:
                        temp_file = _save_visual_to_temp_file(
                            cv_results['visualization'], 
                            temp_dir, 
                            'cv_viz.png'
                        )
                        
                        if temp_file:
                            pdf.add_image(temp_file)
            
            # PDF'i oluştur
            pdf.output(temp_pdf_path)
            
            # PDF dosyasını oku ve BytesIO'ya aktar
            with open(temp_pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            pdf_bytes = io.BytesIO(pdf_data)
            pdf_bytes.seek(0)
            
            # Geçici dizini temizle
            shutil.rmtree(temp_dir)
            
            return pdf_bytes
            
        except Exception as e:
            st.error(f"PDF oluşturma hatası: {str(e)}")
            st.error(traceback.format_exc())
            
            # Temizleme işlemi
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            return None
    
    except Exception as e:
        st.error(f"PDF oluşturma işlemi başlatılamadı: {str(e)}")
        st.error(traceback.format_exc())
        return None

def get_pdf_download_link(pdf_output, filename="SOM_Analiz_Raporu.pdf"):
    """PDF'i indirilebilir link olarak döndürür"""
    if pdf_output is None:
        return None
    
    b64 = base64.b64encode(pdf_output.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Raporu İndir</a>'
    return href 