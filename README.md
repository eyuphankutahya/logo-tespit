## Karmaşık Sahnede Araç Logosu Tespiti – Streamlit Uygulaması  
*YOLOv8 + ByteTrack / BoT-SORT destekli fotoğraf & video logo algılama*

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
| **komut kurulum** | `python -m pip install --upgrade streamlit   ve  python -m pip install --upgrade imageio imageio-ffmpeg ` |
| **Projeyi Çalıştırmak İçin ** | `streamlit run arac_logo_tespiti_fastaccurate.py -- ` |
| **GPU otomatiği** | CUDA algılanırsa model GPU’ya taşınır |
| **Yüksek doğruluk** | Küçük logoları yakalamak için görüntüyü 4 parçaya böler |
| **Gerçek-zamanlı takip** | ByteTrack / BoT-SORT ile kutular kareler arasında korunur |
| **Tarayıcı uyumlu çıktı** | Geçici AVI → H.264 MP4 (+ `--fast-preview` seçeneği) |
| **Kolay arayüz** | Streamlit ile yükle-işle-indir akışı |

---

### Demo
```bash
# Projeyi çalıştırmak için öncelikle aşağıdaki kurulumları yapmak zorundayız.
 python -m pip install --upgrade streamlit   
  

Daha sonrasında projeyi çalıştırmak için:

