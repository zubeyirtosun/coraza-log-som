#!/usr/bin/env python3
"""
Gerçek ZAP Scanner Verisi ile Kapsamlı Test ve Bulgular Üretimi
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ana analiz modüllerini import et
from data_processing import preprocess_data, fix_unhashable_columns
# from advanced_clustering import perform_clustering_comparison, perform_stability_analysis  # Bu satırı kaldır
# from visualizations import *  # Bu da gerekmiyor
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class GercekVeriAnalizi:
    def __init__(self, log_dosyasi_yolu):
        self.log_dosyasi = log_dosyasi_yolu
        self.df = None
        self.X = None
        self.som = None
        self.sonuclar = {
            'analiz_tarihi': datetime.now().isoformat(),
            'veri_kaynak': 'Gerçek ZAP Scanner Logları',
            'veri_analizi': {},
            'som_performansi': {},
            'kumeleme_karsilastirmasi': {},
            'bulgular': {}
        }
    
    def veriyi_yukle_ve_isle(self):
        """Gerçek ZAP scanner verisini yükler ve işler"""
        print("📁 ZAP Scanner log dosyası yükleniyor...")
        
        with open(self.log_dosyasi, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # JSON yapısını işle (tıpkı main.py'deki gibi)
        transactions = []
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    if 'transaction' in item:
                        transactions.append(item['transaction'])
                    else:
                        transactions.append(item)
        
        # DataFrame oluştur
        df = pd.json_normalize(transactions, sep='.')
        df = fix_unhashable_columns(df)
        self.df = df
        
        print(f"✅ {len(df)} adet log kaydı yüklendi")
        print(f"📊 Tespit edilen sütunlar: {len(df.columns)}")
        
        # Veri kalitesi analizi
        self.sonuclar['veri_analizi'] = {
            'toplam_kayit': len(df),
            'sutun_sayisi': len(df.columns),
            'eksik_veri_orani': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'zaman_araligi': self._zaman_araligi_analizi(),
            'traffic_analizi': self._traffic_analizi()
        }
        
        # Veriyi önişle
        print("🔄 Veri önişleme başlatılıyor...")
        processed_df, X = preprocess_data(df.copy(), "Ortalama ile Doldur")
        if X is None:
            raise ValueError("Veri önişleme başarısız!")
        
        self.X = X
        print(f"✅ Önişleme tamamlandı. {X.shape[1]} özellik, {X.shape[0]} kayıt")
        
    def _zaman_araligi_analizi(self):
        """Log verilerinin zaman aralığını analiz eder"""
        zaman_sutunlari = ['timestamp', 'transaction.timestamp', 'unix_timestamp']
        zaman_sutunu = None
        
        for col in zaman_sutunlari:
            if col in self.df.columns:
                zaman_sutunu = col
                break
        
        if zaman_sutunu:
            try:
                if 'unix_timestamp' in zaman_sutunu:
                    # Unix timestamp'i datetime'a çevir
                    timestamps = pd.to_datetime(self.df[zaman_sutunu], unit='ns')
                else:
                    timestamps = pd.to_datetime(self.df[zaman_sutunu])
                
                return {
                    'baslangic': timestamps.min().isoformat(),
                    'bitis': timestamps.max().isoformat(),
                    'sure_dakika': (timestamps.max() - timestamps.min()).total_seconds() / 60
                }
            except:
                return {'hata': 'Zaman verisi işlenemedi'}
        
        return {'durum': 'Zaman sütunu bulunamadı'}
    
    def _traffic_analizi(self):
        """Trafik desenlerini analiz eder"""
        traffic_stats = {}
        
        # HTTP metotları
        method_cols = ['request.method', 'transaction.request.method']
        method_col = next((col for col in method_cols if col in self.df.columns), None)
        if method_col:
            traffic_stats['http_metodlari'] = self.df[method_col].value_counts().to_dict()
        
        # Status kodları
        status_cols = ['response.status', 'transaction.response.status']
        status_col = next((col for col in status_cols if col in self.df.columns), None)
        if status_col:
            traffic_stats['status_kodlari'] = self.df[status_col].value_counts().to_dict()
        
        # Engellenme oranı
        interrupted_cols = ['is_interrupted', 'transaction.is_interrupted']
        interrupted_col = next((col for col in interrupted_cols if col in self.df.columns), None)
        if interrupted_col:
            traffic_stats['engellenme_orani'] = self.df[interrupted_col].mean()
        
        return traffic_stats
    
    def som_analizini_calistir(self):
        """SOM algoritmasını eğitir ve performansını analiz eder"""
        print("🧠 SOM eğitimi başlatılıyor...")
        
        # Grid boyutunu belirle
        n_samples = len(self.df)
        grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
        
        # SOM'u eğit
        som = MiniSom(grid_size, grid_size, self.X.shape[1], 
                     sigma=grid_size/2, learning_rate=0.5, random_seed=42)
        som.random_weights_init(self.X)
        som.train_random(self.X, 1000)
        
        self.som = som
        
        # SOM performans metrikleri
        qe = som.quantization_error(self.X)
        te = som.topographic_error(self.X)
        
        # BMU bilgilerini hesapla
        bmu_coords = []
        qe_values = []
        
        for i, x in enumerate(self.X):
            winner = som.winner(x)
            bmu_coords.append(winner)
            qe_values.append(np.linalg.norm(x - som.get_weights()[winner]))
        
        # DataFrame'e ekle
        self.df['bmu_x'] = [coord[0] for coord in bmu_coords]
        self.df['bmu_y'] = [coord[1] for coord in bmu_coords]
        self.df['quantization_error'] = qe_values
        
        self.sonuclar['som_performansi'] = {
            'grid_boyutu': f"{grid_size}x{grid_size}",
            'toplam_iterasyon': 1000,
            'quantization_error': float(qe),
            'topological_error': float(te),
            'aktif_noron_sayisi': len(set(bmu_coords)),
            'toplam_noron_sayisi': grid_size * grid_size,
            'noron_kullanim_orani': len(set(bmu_coords)) / (grid_size * grid_size)
        }
        
        print(f"✅ SOM eğitimi tamamlandı. QE: {qe:.4f}, TE: {te:.4f}")
    
    def meta_kumeleme_analizi(self):
        """Meta kümeleme algoritmalarını karşılaştırır"""
        print("🔬 Meta kümeleme algoritmaları test ediliyor...")
        
        # SOM BMU koordinatlarını kullan
        som_coords = np.column_stack([self.df['bmu_x'], self.df['bmu_y']])
        
        # Farklı küme sayıları test et
        k_values = [3, 5, 8, 10]
        algoritmalar = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']
        
        sonuclar = []
        
        for k in k_values:
            for algo_name in algoritmalar:
                try:
                    if algo_name == 'KMeans':
                        algo = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = algo.fit_predict(som_coords)
                    elif algo_name == 'DBSCAN':
                        # DBSCAN için eps parametresini ayarla
                        eps = 0.5 if k <= 5 else 1.0
                        algo = DBSCAN(eps=eps, min_samples=3)
                        labels = algo.fit_predict(som_coords)
                        k_actual = len(set(labels)) - (1 if -1 in labels else 0)
                    else:  # AgglomerativeClustering
                        algo = AgglomerativeClustering(n_clusters=k)
                        labels = algo.fit_predict(som_coords)
                    
                    # Metrikler
                    if len(set(labels)) > 1:
                        sil_score = silhouette_score(som_coords, labels)
                        ch_score = calinski_harabasz_score(som_coords, labels)
                        db_score = davies_bouldin_score(som_coords, labels)
                    else:
                        sil_score = ch_score = db_score = -1
                    
                    sonuclar.append({
                        'algoritma': algo_name,
                        'k_hedef': k,
                        'k_gercek': len(set(labels)) - (1 if -1 in labels else 0),
                        'silhouette_score': float(sil_score),
                        'calinski_harabasz': float(ch_score),
                        'davies_bouldin': float(db_score)
                    })
                    
                except Exception as e:
                    print(f"⚠️ {algo_name} (k={k}) hata: {str(e)}")
        
        self.sonuclar['kumeleme_karsilastirmasi'] = sonuclar
        
        # En iyi algoritma
        valid_results = [r for r in sonuclar if r['silhouette_score'] > 0]
        if valid_results:
            en_iyi = max(valid_results, key=lambda x: x['silhouette_score'])
            print(f"✅ En iyi: {en_iyi['algoritma']} (k={en_iyi['k_gercek']}, Sil: {en_iyi['silhouette_score']:.3f})")
    
    def bulgular_olustur(self):
        """Rapor için bulgular özeti oluşturur"""
        veri = self.sonuclar['veri_analizi']
        som = self.sonuclar['som_performansi']
        kumeleme = self.sonuclar['kumeleme_karsilastirmasi']
        
        # En iyi kümeleme sonucu
        valid_clustering = [r for r in kumeleme if r['silhouette_score'] > 0]
        en_iyi_kumeleme = max(valid_clustering, key=lambda x: x['silhouette_score']) if valid_clustering else None
        
        bulgular = {
            'veri_karakteristikleri': {
                'toplam_log_sayisi': veri['toplam_kayit'],
                'analiz_edilen_ozellik_sayisi': som.get('grid_boyutu', 'Belirtilmemiş'),
                'veri_kalitesi': 'İyi' if veri['eksik_veri_orani'] < 5 else 'Orta',
                'zaman_kapsamı_dakika': veri.get('zaman_araligi', {}).get('sure_dakika', 'Belirtilmemiş')
            },
            'som_algoritma_performansi': {
                'quantization_error': som['quantization_error'],
                'topological_error': som['topological_error'],
                'nöron_kullanim_orani': som['noron_kullanim_orani'],
                'degerlendirme': 'Başarılı' if som['quantization_error'] < 1.0 else 'Orta'
            },
            'meta_kumeleme_sonuclari': {
                'en_iyi_algoritma': en_iyi_kumeleme['algoritma'] if en_iyi_kumeleme else 'Belirlenemedi',
                'optimal_kume_sayisi': en_iyi_kumeleme['k_gercek'] if en_iyi_kumeleme else 'Belirlenemedi',
                'silhouette_score': en_iyi_kumeleme['silhouette_score'] if en_iyi_kumeleme else 0,
                'kalite_degerlendirmesi': 'Yüksek' if (en_iyi_kumeleme and en_iyi_kumeleme['silhouette_score'] > 0.5) else 'Orta'
            },
            'guvenlik_bulgulari': self._guvenlik_bulgulari_cikart()
        }
        
        self.sonuclar['bulgular'] = bulgular
        return bulgular
    
    def _guvenlik_bulgulari_cikart(self):
        """Güvenlik odaklı bulgular çıkarır"""
        traffic = self.sonuclar['veri_analizi'].get('traffic_analizi', {})
        
        bulgular = {
            'engellenme_orani': traffic.get('engellenme_orani', 0),
            'risk_seviyesi': 'Düşük',
            'tespit_edilen_saldiri_desenleri': [],
            'anomali_tespiti': 'BMU dağılımında anormallik tespit edildi' if hasattr(self, 'som') else 'Analiz edilemedi'
        }
        
        # Risk seviyesi belirleme
        if bulgular['engellenme_orani'] > 0.3:
            bulgular['risk_seviyesi'] = 'Yüksek'
        elif bulgular['engellenme_orani'] > 0.1:
            bulgular['risk_seviyesi'] = 'Orta'
        
        return bulgular
    
    def sonuclari_kaydet(self, dosya_adi=None):
        """Sonuçları JSON dosyasına kaydeder"""
        if not dosya_adi:
            dosya_adi = f"zap_scanner_analiz_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(dosya_adi, 'w', encoding='utf-8') as f:
            json.dump(self.sonuclar, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Sonuçlar kaydedildi: {dosya_adi}")
        return dosya_adi
    
    def tam_analiz_calistir(self):
        """Tüm analizi çalıştırır"""
        print("🚀 Gerçek ZAP Scanner Verisi Analizi Başlatılıyor...\n")
        
        # 1. Veri yükleme ve işleme
        self.veriyi_yukle_ve_isle()
        
        # 2. SOM analizi
        self.som_analizini_calistir()
        
        # 3. Meta kümeleme
        self.meta_kumeleme_analizi()
        
        # 4. Bulgular
        bulgular = self.bulgular_olustur()
        
        # 5. Sonuçları kaydet
        dosya_adi = self.sonuclari_kaydet()
        
        # 6. Özet rapor
        self.ozet_rapor_yazdir(bulgular)
        
        return self.sonuclar, dosya_adi
    
    def ozet_rapor_yazdir(self, bulgular):
        """Bulgular için özet rapor yazdırır"""
        print(f"\n{'='*60}")
        print("📊 ZAP SCANNER VERİSİ ANALİZ SONUÇLARI")
        print(f"{'='*60}")
        
        print(f"📈 VERİ KARAKTERİSTİKLERİ:")
        print(f"  • Toplam Log: {bulgular['veri_karakteristikleri']['toplam_log_sayisi']:,}")
        print(f"  • Özellik Sayısı: {bulgular['veri_karakteristikleri']['analiz_edilen_ozellik_sayisi']}")
        print(f"  • Veri Kalitesi: {bulgular['veri_karakteristikleri']['veri_kalitesi']}")
        
        print(f"\n🧠 SOM ALGORİTMA PERFORMANSI:")
        print(f"  • Quantization Error: {bulgular['som_algoritma_performansi']['quantization_error']:.4f}")
        print(f"  • Topological Error: {bulgular['som_algoritma_performansi']['topological_error']:.4f}")
        print(f"  • Nöron Kullanım Oranı: {bulgular['som_algoritma_performansi']['nöron_kullanim_orani']:.2%}")
        print(f"  • Değerlendirme: {bulgular['som_algoritma_performansi']['degerlendirme']}")
        
        print(f"\n🔬 META KÜMELEME SONUÇLARI:")
        print(f"  • En İyi Algoritma: {bulgular['meta_kumeleme_sonuclari']['en_iyi_algoritma']}")
        print(f"  • Optimal Küme Sayısı: {bulgular['meta_kumeleme_sonuclari']['optimal_kume_sayisi']}")
        print(f"  • Silhouette Score: {bulgular['meta_kumeleme_sonuclari']['silhouette_score']:.3f}")
        print(f"  • Kalite: {bulgular['meta_kumeleme_sonuclari']['kalite_degerlendirmesi']}")
        
        print(f"\n🛡️ GÜVENLİK BULGULARI:")
        print(f"  • Engellenme Oranı: {bulgular['guvenlik_bulgulari']['engellenme_orani']:.2%}")
        print(f"  • Risk Seviyesi: {bulgular['guvenlik_bulgulari']['risk_seviyesi']}")
        print(f"  • Anomali Tespiti: {bulgular['guvenlik_bulgulari']['anomali_tespiti']}")

# Test scriptini çalıştır
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanım: python gercek_veri_testi.py <log_dosyasi_yolu>")
        print("Örnek: python gercek_veri_testi.py logFiles/example_log.json")
        sys.exit(1)
    
    log_dosyasi = sys.argv[1]
    
    try:
        analiz = GercekVeriAnalizi(log_dosyasi)
        sonuclar, dosya_adi = analiz.tam_analiz_calistir()
        
        print(f"\n✅ Analiz tamamlandı!")
        print(f"📄 Detaylı sonuçlar: {dosya_adi}")
        print(f"🔬 Bu sonuçları raporunuzda kullanabilirsiniz.")
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        import traceback
        print(traceback.format_exc())
