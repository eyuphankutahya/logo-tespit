## Karmaşık Sahnede Araç Logosu Tespiti – Streamlit Uygulaması  
*YOLOv8 + ByteTrack / BoT-SORT destekli fotoğraf & video logo algılama*

---

### İçindekiler
1. [Genel Bakış](#genel-bakış)  
2. [Öne Çıkan Yetenekler](#öne-çıkan-yetenekler)  
3. [Demo](#demo)  
4. [Hızlı Kurulum](#hızlı-kurulum)  
5. [Kullanım](#kullanım)  
6. [Proje Dizini](#proje-dizini)  
7. [Modeli Yeniden Eğitmek](#modeli-yeniden-eğitmek)  
8. [SSS](#sss)  
9. [Katkı Sağlama](#katkı-sağlama)  
10. [Lisans](#lisans)

---

### Genel Bakış
Bu repo, **karmaşık arka planlı görüntü / videolarda araç logosu tespiti** yapmak için hazırlanmış bir Streamlit uygulamasını içerir.  

- **YOLOv8** tabanlı <kbd>best.pt</kbd> dedektörü  
- Fotoğraflar için **Yüksek Doğruluk Modu** (4 tile + 2 aşamalı NMS)  
- Videolar için **ByteTrack** veya **BoT-SORT** takipçili gerçek-zamanlı izleme  
- FFmpeg ile otomatik H.264 MP4 çıktısı  
- Opsiyonel kare atlama, video sabitleme ve hızlı ön-izleme

---

### Öne Çıkan Yetenekler
| Özellik | Açıklama |
|---------|----------|
| **Tek-komut kurulum** | `pip install -r requirements.txt` |
| **GPU otomatiği** | CUDA algılanırsa model GPU’ya taşınır |
| **Yüksek doğruluk** | Küçük logoları yakalamak için görüntüyü 4 parçaya böler |
| **Gerçek-zamanlı takip** | ByteTrack / BoT-SORT ile kutular kareler arasında korunur |
| **Tarayıcı uyumlu çıktı** | Geçici AVI → H.264 MP4 (+ `--fast-preview` seçeneği) |
| **Kolay arayüz** | Streamlit ile yükle-işle-indir akışı |

---

### Demo
```bash
# Projeyi çalıştırmak için... 
streamlit run arac_logo_tespiti_fastaccurate.py --

