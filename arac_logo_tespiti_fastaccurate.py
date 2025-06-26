
# =============================================================================
# 		            ARAÇ LOGOSU TESPİT UYGULAMASI (BİRLEŞTİRİLMİŞ VERSİYON)
# =============================================================================
#
# BU KOD, TARAFINIZDAN SAĞLANAN İKİ FARKLI VERSİYONUN EN İYİ YÖNLERİNİ BİRLEŞTİRİR:
#
# 1. FOTOĞRAF İŞLEME:
#    - İlk kodunuzdaki `process_image` fonksiyonu BİREBİR KORUNMUŞTUR.
#    - Özellikle küçük logoları tespit etmek için görüntüyü parçalara ayırarak
#      analiz eden "Yüksek Doğruluk Modu" aynen çalışmaktadır.
#
# 2. VİDEO İŞLEME:
#    - İkinci kodunuzdaki hızlı ve verimli `process_video` mantığı alınmıştır.
#    - YOLOv8'in kendi optimize edilmiş takipçilerini ("ByteTrack", "BoT-SORT") kullanır.
#    - FFmpeg'i daha sağlam bir şekilde çağırarak uyumluluk sorunlarını azaltır.
#    - Kare atlama (frame-skip), hızlı önizleme gibi performans ayarları eklenmiştir.
#
# 3. ARAYÜZ ve YAPI:
#    - İlk kodun `LogoDetector` sınıf yapısı ve genel Streamlit arayüz akışı
#      (session_state yönetimi, UI fonksiyonları) temel alınmıştır.
#    - İkinci kodun video ayarları (takipçi seçimi, kare atlama vb.) bu yapıya
#      entegre edilmiştir.
#
# =============================================================================

# GEREKSİNİMLER:
# pip install streamlit ultralytics opencv-python torch torchvision Pillow vidstab imageio-ffmpeg

import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import time
import subprocess
from pathlib import Path
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from torchvision.ops import nms
from typing import Generator, Any, Dict

# imageio_ffmpeg, ffmpeg binary dosyasının yolunu güvenilir bir şekilde bulur.
# Bu, özellikle farklı sistemlerde (örn. Streamlit Cloud) dağıtım yaparken hataları önler.
import imageio_ffmpeg

# ------------------ STIL ve ÖN YÜKLEME ---------------------------------
def inject_css():
    """Uygulama için özel CSS stillerini enjekte eder."""
    st.markdown("""
        <style>
        .frame{position:relative;width:100%;padding-top:56.25%;border:2px solid #666;border-radius:10px;background:#181818;overflow:hidden;display:flex;align-items:center;justify-content:center;}
        .frame img,.frame video{position:absolute;inset:0;width:100%!important;height:100%!important;object-fit:contain;}
        div[data-testid="stFileUploadList"]{display:none!important;}
        .overlay-modal{position:fixed;inset:0;background:rgba(24,24,24,.57);backdrop-filter:blur(2px);display:flex;align-items:center;justify-content:center;z-index:1000;}
        .modal-content{padding:2.1rem 3.3rem;border-radius:18px;background:#21232c;color:#fff;font-weight:600;font-size:1.22rem;box-shadow:0 6px 20px #000a;text-align:center;}
        .spin{width:42px;height:42px;border:4px solid #8ec6ff;border-left-color:transparent;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 1rem;}
        @keyframes spin{to{transform:rotate(360deg);}}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner="🧠 Model belleğe yükleniyor...")
def load_yolo_model(path: str) -> YOLO:
    """YOLO modelini yükler ve önbelleğe alır. Varsa GPU'ya taşır."""
    model = YOLO(path)
    if torch.cuda.is_available():
        st.success("✅ CUDA (GPU) desteği bulundu, model GPU'ya yükleniyor.")
        model.to("cuda")
    else:
        st.warning("⚠️ CUDA (GPU) desteği bulunamadı. Model CPU üzerinde çalışacak.")
    return model

# ------------------ MANTIK KATMANI: LogoDetector Sınıfı --------------------------
class LogoDetector:
    def __init__(self, model: YOLO):
        self.model = model

    # !!! ÖNEMLİ: BU FONKSİYON İLK KODDAN HİÇ DEĞİŞTİRİLMEDEN ALINMIŞTIR !!!
    # Yüksek doğruluk modu, küçük nesneleri bulmak için gelişmiş bir mantık kullanır.
    def process_image(self, image_np: np.ndarray, conf: float, imgsz: int, high_accuracy: bool) -> np.ndarray:
        """Bir görüntüyü işler ve logoları tespit eder."""
        if not high_accuracy:
            # Standart, hızlı tespit
            return self.model.predict(image_np, conf=conf, imgsz=imgsz, verbose=False)[0].plot(img=image_np.copy())
        
        # Yüksek Doğruluk Modu: Görüntüyü 4 parçaya böl ve ayrıca tamamını analiz et.
        h, w = image_np.shape[:2]
        with torch.inference_mode():
            # 1. Tüm görüntü üzerinde tespit yap
            full_results = self.model.predict(image_np, conf=conf, imgsz=imgsz, iou=0.45, max_det=300, verbose=False)[0]
            
            # 2. Görüntüyü 4 çeyreğe böl
            tiles = [image_np[y:y + h // 2, x:x + w // 2] for y in (0, h // 2) for x in (0, w // 2)]
            grid = [(x, y) for y in (0, h // 2) for x in (0, w // 2)]
            
            # 3. Her bir parça üzerinde tespit yap
            tile_parts = self.model.predict(tiles, conf=conf, imgsz=imgsz, iou=0.45, max_det=300, verbose=False)

        all_boxes_list = []
        if hasattr(full_results.boxes, "data") and len(full_results.boxes.data) > 0:
            all_boxes_list.append(full_results.boxes.data)
        
        # 4. Parçalardan gelen kutu koordinatlarını orijinal görüntüye göre ayarla
        for res, (gx, gy) in zip(tile_parts, grid):
            if hasattr(res.boxes, "data") and len(res.boxes.data) > 0:
                offset = torch.tensor([gx, gy, gx, gy], dtype=res.boxes.data.dtype, device=res.boxes.data.device)
                cloned_boxes = res.boxes.data.clone()
                cloned_boxes[:, :4] += offset
                all_boxes_list.append(cloned_boxes)
        
        if not all_boxes_list:
            return full_results.plot(img=image_np.copy())
            
        # 5. Tüm kutuları birleştir ve Non-Maximum Suppression (NMS) ile çakışanları temizle
        combined_boxes = torch.cat(all_boxes_list, 0)
        final_results = full_results
        unique_classes = combined_boxes[:, 5].unique()
        
        class_specific_boxes = []
        for c in unique_classes:
            class_mask = combined_boxes[:, 5] == c
            class_boxes_data = combined_boxes[class_mask]
            keep = nms(class_boxes_data[:, :4], class_boxes_data[:, 4], iou_threshold=0.40)
            class_specific_boxes.append(class_boxes_data[keep])

        if not class_specific_boxes:
            final_results.boxes.data = torch.empty((0, 6), device=combined_boxes.device)
            return final_results.plot(img=image_np.copy())
            
        cleaned_boxes_stage1 = torch.cat(class_specific_boxes, 0)
        
        # 6. İkinci aşama NMS (farklı sınıflar arasında çakışma olabilir)
        keep_stage2 = nms(cleaned_boxes_stage1[:, :4], cleaned_boxes_stage1[:, 4], iou_threshold=0.40)
        cleaned_boxes_stage2 = cleaned_boxes_stage1[keep_stage2]

        # 7. İç içe geçmiş kutuları temizle (büyük olanın içinde küçük ve aynı tespit varsa)
        sorted_indices = torch.argsort(cleaned_boxes_stage2[:, 4], descending=True)
        sorted_boxes = cleaned_boxes_stage2[sorted_indices]
        
        to_keep_mask = torch.ones(sorted_boxes.shape[0], dtype=torch.bool, device=sorted_boxes.device)
        for i in range(sorted_boxes.shape[0]):
            if not to_keep_mask[i]: continue
            for j in range(i + 1, sorted_boxes.shape[0]):
                if not to_keep_mask[j]: continue
                
                inter_x1, inter_y1 = torch.max(sorted_boxes[i, 0], sorted_boxes[j, 0]), torch.max(sorted_boxes[i, 1], sorted_boxes[j, 1])
                inter_x2, inter_y2 = torch.min(sorted_boxes[i, 2], sorted_boxes[j, 2]), torch.min(sorted_boxes[i, 3], sorted_boxes[j, 3])
                intersection = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                area_j = (sorted_boxes[j, 2] - sorted_boxes[j, 0]) * (sorted_boxes[j, 3] - sorted_boxes[j, 1])
                
                if area_j > 0 and intersection / area_j > 0.9: # Eğer kutu j, kutu i'nin %90'ından fazlasını kaplıyorsa
                    to_keep_mask[j] = False

        final_results.boxes.data = sorted_boxes[to_keep_mask]
        return final_results.plot(img=image_np.copy())

    # --- YENİ VİDEO İŞLEME FONKSİYONU ---
    # Bu fonksiyon, ikinci kodun hızlı ve verimli mantığını kullanır.
    def process_video(self, video_path: str, conf: float, imgsz: int, skip_n: int, tracker: str, stabilize: bool, fast_preview: bool) -> Generator[Any, None, None]:
        """Bir videoyu işler, logoları tespit eder ve takip eder."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("🎥 Video dosyası açılamadı."); return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # 1. Adım: Ham işlenmiş videoyu geçici bir .avi dosyasına yaz.
        # MJPG codec, çoğu sistemde ek sürücü gerektirmeden çalıştığı için güvenilirdir.
        avi_tmp_path = tempfile.mktemp(suffix=".avi")
        writer = cv2.VideoWriter(avi_tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        if not writer.isOpened():
            st.error("❌ Geçici video yazıcı (MJPG codec) oluşturulamadı."); return

        # Opsiyonel video sabitleme (stabilizasyon)
        stabilizer = None
        if stabilize:
            try:
                from vidstab import VidStab
                stabilizer = VidStab()
            except ImportError:
                st.warning("▶️ `vidstab` kütüphanesi yüklü değil, stabilizasyon atlanıyor.")

        # Takipçi yapılandırmasını seç
        tracker_config = "bytetrack.yaml" if tracker == "ByteTrack" else "botsort.yaml"
        
        last_tracked_result = None
        start_time = time.time()

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            yield (frame_idx / total_frames, f"🎬 Kare {frame_idx}/{total_frames} işleniyor...")

            annotated_frame = frame
            # Her (skip_n + 1) karede bir tespit yap, aradaki karelerde son sonucu kullan
            if frame_idx % (skip_n + 1) == 0:
                last_tracked_result = self.model.track(
                    frame,
                    conf=conf,
                    imgsz=imgsz,
                    tracker=tracker_config,
                    persist=True,  # Kareler arasında takibi sürdür
                    verbose=False
                )[0]
            
            # Eğer bir tespit sonucu varsa, çiz; yoksa orijinal kareyi kullan
            if last_tracked_result:
                annotated_frame = last_tracked_result.plot()

            if stabilizer:
                stabilized_output = stabilizer.stabilize_frame(input_frame=annotated_frame, border_size=10)
                if stabilized_output is not None:
                    annotated_frame = stabilized_output
            
            writer.write(cv2.resize(annotated_frame, (W, H)))

        cap.release()
        writer.release()
        
        yield (1.0, f"✅ {total_frames} kare {time.time() - start_time:.1f} saniyede işlendi. Video yeniden kodlanıyor...")

        # 2. Adım: Oluşturulan .avi dosyasını tarayıcı uyumlu H.264 MP4 formatına dönüştür.
        # Bu yöntem, 'ffmpeg-python' kütüphanesinden daha kararlı çalışır.
        final_out_path = tempfile.mktemp(suffix=".mp4")
        scale_filter = "scale=iw*0.5:-2" if fast_preview else "scale=iw:-2"
        ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        
        cmd = [
            ffmpeg_executable, "-y", "-loglevel", "error", "-i", avi_tmp_path,
            "-vf", scale_filter,
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-b:v", "1.5M", # Kalite için bitrate biraz artırıldı
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", # Sesi kaldır
            final_out_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            with open(final_out_path, "rb") as f:
                yield f.read() # Final video baytlarını gönder
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg yeniden kodlama hatası! Komut çalıştırılamadı. Hata: {e}")
            yield None
        except Exception as e:
            st.error(f"Video dosyası okunurken hata oluştu: {e}")
            yield None


# ------------------ SUNUM KATMANI --------------------------
def setup_sidebar(processing_mode: str) -> Dict[str, Any]:
    """Kenar çubuğundaki ayar kontrollerini oluşturur ve ayarları döndürür."""
    st.sidebar.title("⚙️ Ayarlar")
    settings = {
        "conf": st.sidebar.slider("Güven Eşiği", 0.05, 1.0, 0.30, 0.05),
        "imgsz": st.sidebar.select_slider("Görüntü Boyutu (px)", options=[320, 480, 640, 720, 1080, 1280], value=640)
    }
    st.sidebar.markdown("---")

    if processing_mode == "Fotoğraf":
        settings["high_accuracy"] = st.sidebar.checkbox("Yüksek Doğruluk Modu", value=True, help="Küçük logoları daha iyi bulmak için görüntüyü parçalara ayırarak tarar. Daha yavaş çalışır.")
    else:
        st.sidebar.header("Video Parametreleri")
        settings["skip_n"] = st.sidebar.slider("Kare Atlama (Frame-Skip)", 0, 10, 1, help="Performansı artırmak için her N karede bir tespit yapar.")
        settings["tracker"] = st.sidebar.selectbox("Takipçi Algoritması", ["ByteTrack", "BoT-SORT"])
        settings["stabilize"] = st.sidebar.checkbox("Videoyu Sabitle (Deneysel)", value=False, help="VidStab kütüphanesi ile videodaki titreşimi azaltmaya çalışır.")
        settings["fast_preview"] = st.sidebar.checkbox("⚡ Hızlı Önizleme", value=False, help="Sonuç videosunu daha düşük çözünürlükte ve boyutta oluşturur.")
        
    return settings

def main_ui(detector: LogoDetector):
    """Ana kullanıcı arayüzünü oluşturur ve yönetir."""
    st.markdown("##  Araç Logosu Tespiti ")
    
    processing_mode = st.radio("İşlem Tipi:", ("Fotoğraf", "Video"), horizontal=True, key="processing_mode_radio")

    # Mod değiştiğinde eski sonuçları temizle
    if 'current_mode' not in st.session_state or st.session_state.current_mode != processing_mode:
        st.session_state.current_mode = processing_mode
        if 'processed_media' in st.session_state:
            del st.session_state.processed_media
        if 'uploaded_file_name' in st.session_state:
            del st.session_state.uploaded_file_name
    
    settings = setup_sidebar(processing_mode)

    uploader_label = "Bir Fotoğraf Yükleyin" if processing_mode == "Fotoğraf" else "Bir Video Yükleyin"
    uploader_types = ["png", "jpg", "jpeg", "webp", "bmp"] if processing_mode == "Fotoğraf" else ["mp4", "mov", "avi", "mkv"]
    
    uploaded_file = st.file_uploader(uploader_label, type=uploader_types, key="file_uploader")

    # Yeni dosya yüklendiğinde eski sonuçları temizle
    if uploaded_file is not None:
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            if 'processed_media' in st.session_state:
                del st.session_state.processed_media

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("#### Orijinal Dosya")
        if uploaded_file:
            if processing_mode == "Fotoğraf":
                st.image(uploaded_file, use_container_width=True)
            else:
                st.video(uploaded_file)
        else:
            st.markdown('<div class="frame"></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown("#### Tespit Sonucu")
        if 'processed_media' in st.session_state:
            if processing_mode == "Fotoğraf":
                st.image(st.session_state.processed_media, use_container_width=True)
            else:
                st.video(st.session_state.processed_media)
        else:
            st.markdown('<div class="frame"></div>', unsafe_allow_html=True)

    if uploaded_file and 'processed_media' not in st.session_state:
        modal = st.empty()
        modal.markdown('<div class="overlay-modal"><div class="modal-content"><div class="spin"></div>Model ile işleniyor…</div></div>', unsafe_allow_html=True)
        
        if processing_mode == "Fotoğraf":
            image_np = np.array(Image.open(uploaded_file).convert("RGB"))
            annotated_image = detector.process_image(image_np, **settings)
            st.session_state.processed_media = annotated_image
            st.toast("✅ Fotoğraf işlendi!", icon="🖼️")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                video_path = tmp.name
            
            progress_bar = st.progress(0.0, "Hazırlanıyor...")
            video_generator = detector.process_video(video_path=video_path, **settings)
            
            for result in video_generator:
                if isinstance(result, tuple):
                    progress, status_text = result
                    progress_bar.progress(progress, text=status_text)
                elif result is not None:
                    st.session_state.processed_media = result
            
            progress_bar.empty()
            if 'processed_media' in st.session_state:
                 st.toast("✅ Video takip ile tamamlandı!", icon="🎉")
            else:
                 st.error("❌ Video işlenirken bir hata oluştu ve sonuç üretilemedi.")


        modal.empty()
        st.rerun()

    if processing_mode == "Video" and 'processed_media' in st.session_state:
        st.download_button("📥 Sonuç Videosunu İndir", st.session_state.processed_media, "sonuc.mp4", "video/mp4")

if __name__ == "__main__":
    st.set_page_config(page_title="Araç Logosu Tespiti", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    model = load_yolo_model("best.pt")
    detector = LogoDetector(model)
    main_ui(detector)
