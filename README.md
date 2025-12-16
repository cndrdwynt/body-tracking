# Pengolahan Citra Video: Real-time Interactive 2D VTuber System

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tracking-0099CC?style=for-the-badge&logo=google&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-orange?style=for-the-badge)

## Ringkasan Proyek (Abstract)
Proyek ini adalah pengembangan sistem **VTuber (Virtual YouTuber)** 2D interaktif yang beroperasi secara *real-time*. Sistem ini dirancang untuk memungkinkan pengguna mengontrol avatar digital hanya menggunakan *webcam* standar dan mikrofon, tanpa memerlukan peralatan *motion capture* yang mahal.

Menggunakan algoritma *Computer Vision* dan *Deep Learning* (melalui MediaPipe Holistic), sistem ini mampu melacak orientasi wajah, ekspresi mulut, dan gestur tangan pengguna untuk diterjemahkan menjadi animasi avatar yang responsif dan alami.

### Demo Preview
Simak demonstrasi sistem yang sedang berjalan melalui tautan berikut:

[![Demo Video](https://img.shields.io/badge/▶_NONTON_VIDEO_DEMO-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1vX0njVSi1sGNz7uf29P-lsX9CleYxlFy/view?usp=sharing)

> **Link Alternatif:** [Klik di sini untuk melihat demo di Google Drive](https://drive.google.com/file/d/1vX0njVSi1sGNz7uf29P-lsX9CleYxlFy/view?usp=sharing)

---

## Fitur Utama

### 1. Advanced Tracking System
* **Holistic Tracking:** Menggunakan MediaPipe untuk mendeteksi *Face Mesh* (468 titik wajah), *Pose* (tubuh bagian atas), dan *Hand Tracking* secara simultan.
* **Face Direction Detection:** Algoritma khusus yang menghitung orientasi hidung relatif terhadap bahu untuk menentukan apakah avatar harus menoleh ke kiri, kanan, atau tengah.
* **Eye Idle Animation:** Sistem animasi prosedural untuk pergerakan pupil mata yang alami (lirik kiri/kanan) saat pengguna diam, mencegah avatar terlihat kaku ("dead stare").

### 2. Intelligent Lip Sync (Hybrid Mode)
Sistem sinkronisasi bibir menggunakan pendekatan **Hybrid** yang menggabungkan dua input:
* **Visual-based:** Analisis rasio aspek mulut (MAR) dari kamera untuk mendeteksi bentuk mulut (A, I, U, E, O).
* **Audio-based:** Analisis amplitudo suara (RMS) dari mikrofon untuk mendeteksi intensitas bicara.
* *Smooth Transition:* Algoritma interpolasi untuk transisi bentuk mulut yang halus antar *frame*.

### 3. Gesture Recognition & Reaction
Sistem mengenali gestur spesifik untuk memicu perubahan ekspresi dan pose avatar:
* **Wave:** Lambaian tangan (memicu animasi menyapa).
* **Blushing:** Menyilangkan tangan di depan dada/wajah (pose malu).
* **Laugh Detection:** Deteksi tawa berbasis kombinasi visual (mulut terbuka lebar) dan audio (level suara tinggi), memicu efek "guncangan" badan dan partikel tawa.
* **Thumbs Up, Peace, & OK:** Deteksi jari untuk pose tangan spesifik.

### 4. Physics & Rendering Engine
* **Procedural Hair Physics:** Simulasi fisika sederhana (spring/damper system) untuk pergerakan rambut yang merespons gerakan kepala (inersia dan gravitasi).
* **Video Effects Overlay:** Mendukung efek video *greenscreen* (chroma key) yang muncul saat emosi tertentu (misal: efek kaget/excited, efek tawa).
* **Dynamic Background:** Dukungan untuk background gambar statis, video loop, atau *green screen* untuk penggunaan di OBS.

---

## Struktur Proyek

Berikut adalah arsitektur modular dari kode sumber:

```text
Project Root
├──  main.py           # Entry point: Inisialisasi sistem, loop utama, dan manajemen resource.
├──  vtuber_core.py    # Core Logic: Pemrosesan MediaPipe, deteksi gestur, dan logika state avatar.
├──  renderer.py       # Graphics Engine: Fungsi menggambar (compositing) avatar, UI, dan background.
├──  utils.py          # Utilities: Helper untuk matematika vektor, chroma key, dan manipulasi gambar.
├──  config.py         # Configuration: Pusat pengaturan parameter (threshold, path file, konstanta fisika).
├──  images/           # Aset grafis avatar (PNG layers).
├──  effects/          # Aset video efek visual.
└──  backgrounds/      # Aset latar belakang.

```

## Instalasi & Penggunaan
Prasyarat
Pastikan Python 3.8+ terinstal. Install library yang dibutuhkan:

```
pip install opencv-python mediapipe pyaudio numpy
```

Untuk menjalankan Program
Jalankan file utama melalui terminal:

```
python main.py
```

## Kontrol Keyboard
* b: Mengganti Latar Belakang (Virtual Image -> Video -> Green Screen).
* q / Esc: Keluar dari program.

## Konfigurasi Teknis
Seluruh sensitivitas sistem dapat diatur melalui config.py, antara lain:

* LIP_SYNC_MODE: Ubah ke "VISUAL", "AUDIO", atau "HYBRID".
* HYBRID_VISUAL_WEIGHT: Bobot dominasi kamera vs mikrofon untuk mulut.
* POSE_SPRING_FACTOR: Mengatur kehalusan/kelambatan gerakan badan mengikuti kepala.
* LAUGH_RMS_THRESHOLD: Sensitivitas mikrofon untuk mendeteksi tawa.

## Author & Credits
Project ini dikembangkan sebagai implementasi Computer Vision untuk hiburan interaktif.

* Engine: Python & OpenCV
* AI Model: Google MediaPipe
* Assets: Karakter "Margo" (Custom Assets)
