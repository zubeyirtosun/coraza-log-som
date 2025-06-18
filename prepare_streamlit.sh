#!/bin/bash
echo "Streamlit Cloud için font hazırlığı yapılıyor..."

# static_fonts dizini oluştur (yoksa)
mkdir -p static_fonts

# dejavu fontlarını kopyala
if [ -d "fonts/dejavu-fonts-ttf-2.37/ttf" ]; then
    echo "DejaVu fontları bulundu, static_fonts dizinine kopyalanıyor..."
    cp -f fonts/dejavu-fonts-ttf-2.37/ttf/*.ttf static_fonts/
    cp -f fonts/dejavu-fonts-ttf-2.37/ttf/*.pkl static_fonts/ 2>/dev/null
else
    echo "DejaVu fontları bulunamadı!"
fi

# .pkl dosyaları oluştur (yoksa)
if [ ! -f "static_fonts/DejaVuSans.pkl" ]; then
    echo "PKL dosyaları eksik, boş dosya oluşturuluyor..."
    touch static_fonts/DejaVuSans.pkl
    touch static_fonts/DejaVuSans-Bold.pkl
    touch static_fonts/DejaVuSans-Oblique.pkl
fi

echo "Font hazırlığı tamamlandı." 