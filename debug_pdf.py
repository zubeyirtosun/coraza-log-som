import streamlit as st
import sys
import os

# Debug için session state'leri yazdır
def debug_session_state():
    st.write("## Session State Debug")
    
    key_groups = {
        "Ana Veriler": ['df', 'X', 'som', 'grid_size'],
        "SOM Sonuçları": ['som_done', 'som_weights_reshaped', 'summary_df'],
        "Meta Kümeleme": ['meta_clustering_done', 'meta_clusters', 'df_meta', 'optimal_k'],
        "Gelişmiş Analizler": [
            'optimal_k_results', 'alternative_clustering_results', 
            'stability_results', 'dimensionality_reduction_results', 
            'cross_validation_results', 'advanced_analysis_results'
        ],
        "PDF": ['pdf_report']
    }
    
    for group_name, keys in key_groups.items():
        st.write(f"### {group_name}")
        for key in keys:
            if key in st.session_state:
                value = st.session_state[key]
                if value is None:
                    st.write(f"- **{key}**: None")
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    try:
                        st.write(f"- **{key}**: {type(value).__name__} (boyut: {len(value)})")
                    except:
                        st.write(f"- **{key}**: {type(value).__name__}")
                else:
                    st.write(f"- **{key}**: {type(value).__name__}")
                    
                # Gelişmiş analiz sonuçları için detay
                if key == 'advanced_analysis_results' and value is not None:
                    st.write("  Detaylar:")
                    for sub_key in value.keys():
                        st.write(f"    - {sub_key}: {type(value[sub_key]).__name__}")
            else:
                st.write(f"- **{key}**: MEVCUT DEĞİL")

def test_pdf_components():
    st.write("## PDF Bileşenlerini Test Et")
    
    # Temel kontroller
    if st.session_state.som is None:
        st.error("❌ SOM modeli bulunamadı!")
        return False
    
    if st.session_state.X is None:
        st.error("❌ Veri matrisi bulunamadı!")
        return False
        
    if st.session_state.df is None:
        st.error("❌ DataFrame bulunamadı!")
        return False
    
    st.success("✅ Temel veriler mevcut")
    
    # Gelişmiş analiz kontrolü
    if 'advanced_analysis_results' not in st.session_state or st.session_state.advanced_analysis_results is None:
        st.warning("⚠️ Gelişmiş analiz sonuçları bulunamadı")
        return False
    
    advanced_results = st.session_state.advanced_analysis_results
    st.success("✅ Gelişmiş analiz sonuçları mevcut")
    
    # Her gelişmiş analiz bileşenini kontrol et
    components = {
        'optimal_k': 'Optimal K Analizi',
        'clustering_comparison': 'Kümeleme Algoritmaları Karşılaştırması',
        'stability_analysis': 'Stabilite Analizi',
        'dimensionality_reduction': 'Boyut İndirgeme',
        'cross_validation': 'Çapraz Doğrulama'
    }
    
    for comp_key, comp_name in components.items():
        if comp_key in advanced_results:
            st.success(f"✅ {comp_name} mevcut")
        else:
            st.warning(f"⚠️ {comp_name} bulunamadı")
    
    return True

def test_som_visualizations():
    st.write("## SOM Görselleştirme Testi")
    
    if st.session_state.som is None:
        st.error("❌ SOM modeli bulunamadı!")
        return
    
    if st.session_state.df is None:
        st.error("❌ DataFrame bulunamadı!")
        return
    
    # Temel kontroller
    required_cols = ['bmu_x', 'bmu_y', 'quantization_error']
    missing_cols = [col for col in required_cols if col not in st.session_state.df.columns]
    
    if missing_cols:
        st.error(f"❌ Eksik sütunlar: {missing_cols}")
        return
    
    st.success("✅ SOM görselleştirme için gerekli veriler mevcut")
    
    # SOM görselleştirme türlerini test et
    import matplotlib.pyplot as plt
    import io
    
    try:
        # 1. Scatter Plot Test
        st.write("### 1. Scatter Plot (Web'teki gibi)")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            st.session_state.df['bmu_x'], 
            st.session_state.df['bmu_y'], 
            c=st.session_state.df['quantization_error'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Niceleme Hatası')
        plt.title('SOM Scatter Plot')
        plt.xlabel('BMU X')
        plt.ylabel('BMU Y')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf)
        plt.close()
        st.success("✅ Scatter plot başarılı")
        
        # 2. Histogram Test
        st.write("### 2. Histogram (Yoğunluk Haritası)")
        plt.figure(figsize=(8, 6))
        histogram = plt.hist2d(
            st.session_state.df['bmu_x'], 
            st.session_state.df['bmu_y'], 
            bins=st.session_state.grid_size if st.session_state.grid_size else 10,
            cmap='viridis'
        )
        plt.colorbar(histogram[3], label='Log Sayısı')
        plt.title('SOM Histogram')
        plt.xlabel('BMU X')
        plt.ylabel('BMU Y')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf)
        plt.close()
        st.success("✅ Histogram başarılı")
        
        # 3. Distance Map Test
        st.write("### 3. Distance Map (U-Matrix)")
        plt.figure(figsize=(8, 6))
        distance_map = st.session_state.som.distance_map().T
        im = plt.imshow(distance_map, cmap='viridis')
        plt.colorbar(im, label='Nöron Mesafesi')
        plt.title('SOM Distance Map')
        plt.xlabel('Izgara X')
        plt.ylabel('Izgara Y')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf)
        plt.close()
        st.success("✅ Distance Map başarılı")
        
    except Exception as e:
        st.error(f"❌ SOM görselleştirme testi hatası: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

def main():
    st.title("PDF Debug Aracı")
    
    if st.button("Session State'i Kontrol Et"):
        debug_session_state()
    
    if st.button("PDF Bileşenlerini Test Et"):
        test_pdf_components()
    
    if st.button("SOM Görselleştirmelerini Test Et"):
        test_som_visualizations()
    
    if st.button("Test PDF Oluştur"):
        try:
            from pdf_report import create_pdf_report
            
            st.info("PDF oluşturuluyor...")
            pdf_output = create_pdf_report(
                title="Test PDF Raporu",
                include_basic=True,
                include_advanced=True
            )
            
            if pdf_output is not None:
                st.success("✅ PDF başarıyla oluşturuldu!")
                st.download_button(
                    "Test PDF'i İndir",
                    pdf_output,
                    "test_som_report.pdf",
                    "application/pdf"
                )
            else:
                st.error("❌ PDF oluşturulamadı!")
        except Exception as e:
            st.error(f"❌ PDF oluşturma hatası: {str(e)}")
            import traceback
            st.write(traceback.format_exc())

if __name__ == "__main__":
    main() 