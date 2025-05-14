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

class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        # UTF-8 desteği ekle
        self.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf', uni=True)
    
    def header(self):
        # Logo veya başlık eklenebilir
        self.set_font('DejaVu', 'B', 15)
        self.cell(0, 10, 'SOM ve Meta Kümeleme Analizi Raporu', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def chapter_body(self, body):
        self.set_font('DejaVu', '', 12)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_image(self, img_data, x=None, y=None, w=190, h=100):
        if x is None:
            x = self.get_x()
        if y is None:
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

def create_pdf_report(title="SOM ve Meta Kümeleme Analizi Raporu", include_basic=True, include_advanced=True):
    if st.session_state.som is None or st.session_state.X is None:
        st.warning("SOM modeli henüz eğitilmemiş, rapor oluşturulamıyor.")
        return None
    
    try:
        # DejaVu font'larının varlığını kontrol et
        import subprocess
        import sys
        
        try:
            # Sistemdeki fontları kontrol et
            result = subprocess.run(['fc-list'], stdout=subprocess.PIPE, text=True, check=True)
            font_list = result.stdout
            
            # DejaVu yoksa uyarı ver
            if 'DejaVu' not in font_list:
                st.warning("DejaVu yazı tipi bulunamadı. PDF Unicode karakterleri doğru göstermeyebilir.")
                
                # Alternatif çözüm - simple PDF oluşturma
                simple_pdf = True
            else:
                simple_pdf = False
        except:
            # Komut çalıştırılamazsa varsayılan olarak basit PDF kullan
            simple_pdf = True
            
        # Geçici dosya dizini oluştur
        temp_dir = tempfile.mkdtemp()
        
        # PDF oluştur
        if simple_pdf:
            # Alternatif çözüm - basit PDF ve ASCII karakterlere dönüşüm yap
            pdf = FPDF()
            
            # ASCII dışı karakterleri dönüştür
            def sanitize(text):
                # Türkçe karakterleri ASCII eşdeğerleriyle değiştir
                replacements = {
                    'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 
                    'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 
                    'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U'
                }
                for k, v in replacements.items():
                    text = text.replace(k, v)
                return text
                
            # Orijinal methodları sakla
            orig_chapter_title = pdf.chapter_title if hasattr(pdf, 'chapter_title') else None
            orig_chapter_body = pdf.chapter_body if hasattr(pdf, 'chapter_body') else None
            
            # Yeni metodlar tanımla
            def safe_chapter_title(self, title):
                if orig_chapter_title:
                    return orig_chapter_title(sanitize(title))
                else:
                    self.set_font('Arial', 'B', 14)
                    self.ln(5)
                    self.cell(0, 10, sanitize(title), 0, 1, 'L')
                    self.ln(5)
                    
            def safe_chapter_body(self, body):
                if orig_chapter_body:
                    return orig_chapter_body(sanitize(body))
                else:
                    self.set_font('Arial', '', 12)
                    self.multi_cell(0, 5, sanitize(body))
                    self.ln()
            
            # Metodları geçici olarak değiştir
            pdf.chapter_title = lambda title: safe_chapter_title(pdf, title)
            pdf.chapter_body = lambda body: safe_chapter_body(pdf, body)
        else:
            # UTF-8 destekli PDF kullan
            pdf = PDF()
            
        # Sayfa ekle
        pdf.add_page()
        
        # Başlık ayarla
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        pdf.ln(5)
        
        # Rapor bilgileri
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.chapter_body(f"Oluşturulma Tarihi: {timestamp}")
        
        # Veri boyutunu kontrol et
        if 'df' in st.session_state and st.session_state.df is not None:
            pdf.chapter_body(f"Veri Sayısı: {len(st.session_state.df)}")
        else:
            pdf.chapter_body("Veri Sayısı: Bilinmiyor")
        
        # Grid boyutunu kontrol et
        if 'grid_size' in st.session_state and st.session_state.grid_size is not None:
            pdf.chapter_body(f"SOM Izgara Boyutu: {st.session_state.grid_size}x{st.session_state.grid_size}")
        else:
            pdf.chapter_body("SOM Izgara Boyutu: Bilinmiyor")
        
        pdf.ln(10)
        
        # SOM özet tablosu
        if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
            pdf.chapter_title("SOM Özet Tablosu")
            
            # Tabloyu görselleştir
            plt.figure(figsize=(10, 6))
            summary_df_head = st.session_state.summary_df.head(20)  # İlk 20 satır
            table_data = []
            table_columns = ['Nöron', 'Engellenme Oranı', 'En Sık URI', 'Ort. Hata', 'Log Sayısı']
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
        pdf.chapter_title("SOM Dağılımı Analizi")
        
        # SOM dağılımını görselleştir
        plt.figure(figsize=(10, 6))
        plt.hist2d(st.session_state.df['bmu_x'], st.session_state.df['bmu_y'], bins=st.session_state.grid_size)
        plt.colorbar(label='Log Sayısı')
        plt.title('SOM Izgara Dağılımı')
        plt.xlabel('BMU X')
        plt.ylabel('BMU Y')
        
        # Geçici dosyaya kaydet
        temp_file = os.path.join(temp_dir, 'som_dist.png')
        plt.savefig(temp_file, format='png')
        plt.close()
        
        # PDF'e ekle
        pdf.add_image(temp_file)
        
        # Niceleme Hatası Dağılımı
        pdf.chapter_title("Niceleme Hatası Dağılımı")
        
        # Niceleme hatasını görselleştir
        plt.figure(figsize=(10, 6))
        plt.hist(st.session_state.df['quantization_error'], bins=50)
        plt.title('Niceleme Hatası Dağılımı')
        plt.xlabel('Niceleme Hatası')
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
            pdf.chapter_title("Meta Kümeleme Sonuçları")
            
            # Meta küme sayısını al
            if 'optimal_k' in st.session_state and st.session_state.optimal_k is not None:
                pdf.chapter_body(f"Optimal Küme Sayısı: {st.session_state.optimal_k} (Otomatik Belirlendi)")
            elif 'df_meta' in st.session_state and st.session_state.df_meta is not None:
                n_clusters = st.session_state.df_meta['meta_cluster'].nunique()
                pdf.chapter_body(f"Küme Sayısı: {n_clusters} (Manuel Seçildi)")
            
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
                plt.colorbar(scatter, label='Meta Küme')
                plt.title('Meta Küme Dağılımı')
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
                plt.title('Meta Küme Bazında Log Sayısı')
                plt.xlabel('Meta Küme')
                plt.ylabel('Log Sayısı')
                
                # Geçici dosyaya kaydet
                temp_file = os.path.join(temp_dir, 'cluster_counts.png')
                plt.savefig(temp_file, format='png')
                plt.close()
                
                # PDF'e ekle
                pdf.add_image(temp_file)
        
        # Optimal Küme Sayısı Analizi
        if 'k_range_metrics' in st.session_state and st.session_state.k_range_metrics is not None:
            pdf.add_page()
            pdf.chapter_title("Optimal Küme Sayısı Analizi")
            
            # Metrikleri görselleştir
            plt.figure(figsize=(10, 6))
            plt.plot(
                st.session_state.k_range_metrics['k_range'], 
                st.session_state.k_range_metrics['silhouette_scores'],
                'o-', label='Silüet Skoru'
            )
            plt.plot(
                st.session_state.k_range_metrics['k_range'], 
                np.array(st.session_state.k_range_metrics['calinski_scores']) / np.max(st.session_state.k_range_metrics['calinski_scores']),
                'o-', label='Normalize Calinski-Harabasz'
            )
            plt.plot(
                st.session_state.k_range_metrics['k_range'], 
                1 / (1 + np.array(st.session_state.k_range_metrics['davies_scores'])),
                'o-', label='Ters Davies-Bouldin'
            )
            plt.title('Farklı Küme Sayıları için Metrikler')
            plt.xlabel('Küme Sayısı')
            plt.ylabel('Metrik Değeri')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Geçici dosyaya kaydet
            temp_file = os.path.join(temp_dir, 'optimal_k.png')
            plt.savefig(temp_file, format='png')
            plt.close()
            
            # PDF'e ekle
            pdf.add_image(temp_file)
        
        # Alternatif Kümeleme Algoritmaları
        if 'alternative_clustering_results' in st.session_state and st.session_state.alternative_clustering_results is not None:
            pdf.add_page()
            pdf.chapter_title("Alternatif Kümeleme Algoritmaları Karşılaştırması")
            
            alt_results = st.session_state.alternative_clustering_results
            
            # Metrikler tablosunu oluştur
            if 'metrics' in alt_results:
                metrics_df = pd.DataFrame(alt_results['metrics'])
                
                plt.figure(figsize=(12, 5))
                ax = plt.subplot(111)
                ax.axis('off')
                table = ax.table(
                    cellText=metrics_df.values.round(4),
                    colLabels=metrics_df.columns,
                    rowLabels=metrics_df.index,
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Geçici dosyaya kaydet
                temp_file = os.path.join(temp_dir, 'metrics_table.png')
                plt.savefig(temp_file, format='png', bbox_inches='tight')
                plt.close()
                
                # PDF'e ekle
                pdf.add_image(temp_file)
            
            # Algoritmaların görselleştirmelerini ekle
            if 'visualizations' in alt_results:
                for algo_name, visual_data in alt_results['visualizations'].items():
                    if algo_name != 'Metrikler':  # Metrikler tablosunu zaten gösterdik
                        pdf.chapter_body(f"Algoritma: {algo_name}")
                        
                        # Görselleştirmeyi geçici dosyaya kaydet
                        temp_file = _save_visual_to_temp_file(
                            visual_data, 
                            temp_dir, 
                            f'algo_{algo_name}.png'
                        )
                        
                        if temp_file:
                            # PDF'e ekle
                            pdf.add_image(temp_file)
        
        # Kümeleme Stabilitesi Analizi
        if 'stability_results' in st.session_state and st.session_state.stability_results is not None:
            pdf.add_page()
            pdf.chapter_title("Kümeleme Stabilitesi Analizi")
            
            stability = st.session_state.stability_results
            if 'stability_score' in stability:
                pdf.chapter_body(f"Stabilite Skoru: {stability['stability_score']:.4f}")
                pdf.chapter_body(f"Not: 1.0'a yakın değerler daha stabil kümelemeyi gösterir.")
            
            if 'stability_matrix' in stability:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    np.array(stability['stability_matrix']), 
                    annot=True, 
                    cmap='viridis',
                    fmt='.2f'
                )
                plt.title('Kümeleme Stabilite Matrisi')
                plt.xlabel('Çalıştırma Numarası')
                plt.ylabel('Çalıştırma Numarası')
                
                # Geçici dosyaya kaydet
                temp_file = os.path.join(temp_dir, 'stability.png')
                plt.savefig(temp_file, format='png')
                plt.close()
                
                # PDF'e ekle
                pdf.add_image(temp_file)
            
            # Görselleştirme dosyasını ekle
            if 'visualization' in stability:
                temp_file = _save_visual_to_temp_file(
                    stability['visualization'], 
                    temp_dir, 
                    'stability_viz.png'
                )
                
                if temp_file:
                    # PDF'e ekle
                    pdf.add_image(temp_file)
        
        # Boyut İndirgeme Analizleri
        if 'dimensionality_reduction_results' in st.session_state and st.session_state.dimensionality_reduction_results is not None:
            pdf.add_page()
            pdf.chapter_title("Boyut İndirgeme Analizleri")
            
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
                    # PDF'e ekle
                    pdf.add_image(temp_file)
        
        # PDF'i kaydet ve indirilebilir link oluştur
        try:
            pdf_output = io.BytesIO()
            if simple_pdf:
                # Basit PDF için encode seçeneği ekle
                pdf.output(pdf_output, 'F')
            else:
                # UTF-8 PDF için normal çıktı
                pdf.output(pdf_output)
                
            pdf_output.seek(0)
            
            # Geçici dizini temizle
            shutil.rmtree(temp_dir)
            
            return pdf_output
        except Exception as pdf_output_error:
            st.error(f"PDF oluşturma çıktı hatası: {str(pdf_output_error)}")
            import traceback
            st.write(traceback.format_exc())
            
            # Geçici dizini temizlemeyi dene
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            return None
    
    except Exception as e:
        st.error(f"PDF oluşturma hatası: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        
        # Geçici dizini temizlemeyi dene
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return None

def get_pdf_download_link(pdf_output, filename="SOM_Analiz_Raporu.pdf"):
    """PDF'i indirilebilir link olarak döndürür"""
    if pdf_output is None:
        return None
    
    b64 = base64.b64encode(pdf_output.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Raporu İndir</a>'
    return href 