Tugas Body Tracking with Character - Pengolahan Citra Video
Real-time VTuber system dengan full body tracking, gesture recognition, dan advanced animation menggunakan OpenCV dan MediaPipe.

Deskripsi Project

VTuber Margo adalah aplikasi motion capture real-time yang menangkap gerakan tubuh pengguna melalui webcam dan mentransformasikannya menjadi animasi karakter virtual 2D. System ini mengintegrasikan deteksi pose tubuh, gestur tangan, landmark wajah, dan input audio untuk menciptakan karakter virtual yang responsif.

Tujuan Pembelajaran

1. Implementasi body tracking menggunakan MediaPipe Holistic
2. Pengolahan citra real-time dengan OpenCV
3. Integrasi multiple sensors (video dan audio)
4. Implementasi finite state machine untuk gesture recognition
5. Physics simulation untuk animasi natural


Fitur Utama

1. Body & Hand Tracking

a) Full body pose detection dengan 33 landmark points
Hand gesture recognition untuk 10+ gestures:

	1. Peace Sign	: Jari bentuk 'V'.
	2. OK Sign	: Jari 'OK'.
	3. Thumbs Up	: Jempol ke atas.
	4. Wave		: Satu tangan di atas bahu.
	5. Excited	: Kedua tangan di atas bahu. 
	6. Blushing	: Kedua tangan menutupi wajah.

b) Audio-Based Features

	1. Real-time lip sync dengan 5 mouth states (closed, small, medium, wide, O)
	2. Laugh detection berdasarkan audio threshold dengan shake effect
	3. RMS monitoring untuk visual feedback audio level

c) Visual Effects

	1. 3 Background modes: Virtual Image, Virtual Video, Green Screen
	2. Video effects overlay dengan chroma key removal
	3. Smooth animations: spring physics untuk hair movement, breathing animation, bounce effect
	4. Idle animations: random blink dan look around sequence

Teknologi yang Digunakan

1. Python 3.11
2. OpenCV (Untuk tangkapan webcam, pemrosesan gambar, dan chroma key)
3. MediaPipe (Untuk model machine learning pelacakan holistik)
4. PyAudio (Untuk menangkap audio dari mikrofon)
5. NumPy (Untuk operasi matematika pada gambar/video)

Arsitektur Project

project/
├── main.py              
├── config.py            
├── vtuber_core.py       
├── renderer.py          
├── utils.py             
├── images/              
│   ├── margo_head_.png
│   ├── margo_body_.png
│   ├── hair_back.png
│   └── dll
├── backgrounds/         
│   ├── background_virtual.jpeg
│   └── background_video.mp4
├── effects/             
│   ├── excited_effect.mp4
│   ├── laugh_effect.mp4
│   └── thumbsup_effect.mp4
└── README.md

Modular Project

config.py
Berfungsi sebagai pusat konfigurasi utama. Seluruh parameter penting seperti sensitivitas mikrofon, kecepatan animasi, serta pengaturan input disimpan dalam file ini.
Perubahan nilai atau penyesuaian sistem dapat dilakukan di sini tanpa perlu mengubah logika utama program.

utils.py
Menampung kumpulan fungsi bantu (helper functions) yang digunakan di berbagai bagian proyek.
Fungsi-fungsi di modul ini membantu proses seperti pengolahan gambar, rendering teks, dan komputasi sederhana agar kode utama tetap bersih dan modular.

vtuber_core.py
Merupakan inti dari sistem. Modul ini menangani seluruh proses utama, mulai dari pemuatan aset karakter, deteksi pose dan gestur tangan, hingga perhitungan fisika sederhana untuk animasi rambut dan gerakan tubuh.
Selain itu, modul ini juga menganalisis input audio untuk menentukan kondisi karakter seperti berbicara atau tertawa.

renderer.py
Bertanggung jawab atas proses visualisasi. Modul ini menerima data hasil pemrosesan dari sistem utama dan menampilkannya ke layar dalam bentuk animasi karakter, latar belakang, serta elemen antarmuka pengguna.
Fokus modul ini adalah efisiensi rendering dan sinkronisasi visual yang halus.

main.py
Berperan sebagai titik masuk (entry point) program dan pengendali utama alur kerja sistem. Modul ini mengatur proses inisialisasi kamera, memanggil fungsi pendeteksian dan pemrosesan dari vtuber_core, serta mengoordinasikan proses rendering secara berurutan dalam loop utama.
Selain itu, modul ini juga menangani prosedur penghentian program dan pembersihan sumber daya ketika aplikasi ditutup.

Dependencies

opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pyaudio>=0.2.13

Cara Menggunakan

Kontrol Keyboard

B - Ganti background mode (Image, Video, Green Screen)
Q atau ESC - Keluar dari aplikasi

Gesture Controls

Peace
Telunjuk dan jari tengah tegak.
Efek: menampilkan pose tubuh “Peace”.

OK
Telunjuk menyentuh ibu jari.
Efek: menampilkan pose tubuh “OK”.

Thumbs Up
Ibu jari tegak, jari lainnya tertutup.
Efek: menampilkan pose tubuh dan efek video “Thumbs Up”.

Wave
Satu tangan di atas bahu.
Efek: menampilkan pose tubuh “Wave”.

Excited
Kedua tangan di atas bahu.
Efek: mengaktifkan ekspresi wajah “Excited” dan efek video pendukung.

Blushing
Kedua tangan di wajah.
Efek: mengaktifkan ekspresi wajah “Blushing”.

Audio Features

Bicara dengan volume normal mengaktifkan sinkronisasi bibir otomatis (lip sync) dengan empat bentuk mulut berbeda.
Suara dengan volume tinggi menghasilkan pergantian antara bentuk mulut lebar dan bentuk O.
Tertawa memicu ekspresi “Laugh Face”, disertai efek getar (shake effect) dan overlay video khusus.


Teknologi dan Algoritma

1. MediaPipe Holistic
2. System menggunakan MediaPipe Holistic untuk simultaneous tracking:

	a) 33 pose landmarks
	b) 468 face landmarks
	c) 21 hand landmarks per hand

1. Spring Physics untuk Rambut (Hair Simulation)
Menggunakan prinsip fisika pegas untuk membuat gerakan rambut terlihat alami dan lembut.
Algoritmanya bekerja seperti ini:

force = (target - current) * spring_factor
velocity = (velocity + force) * drag_factor
position += velocity


Rambut akan selalu “mengejar” posisi target (seperti mengikuti gerakan kepala), tetapi dengan efek elastis dan sedikit tertunda, menyerupai gerakan rambut nyata.

2. Gesture Recognition Pipeline
Alur pemrosesan untuk mengenali gestur tangan:

Raw landmarks 
→ Finger detection 
→ Gesture classification 
→ Buffer system (10 frames) 
→ Confirmation (4/10 threshold) 
→ Stable gesture output


Artinya, sistem membaca titik-titik (landmarks) tangan dari kamera, mengenali bentuk jari, lalu menentukan gestur.
Buffer digunakan untuk memastikan gestur stabil (harus muncul di minimal 4 dari 10 frame berturut-turut sebelum dianggap valid).

3. Audio-Based Lip Sync
Sinkronisasi gerakan mulut berdasarkan kekuatan suara (RMS dari audio).

Audio RMS 
→ Smoothing filter 
→ Threshold mapping 
→ Mouth state (closed/small/medium/wide/o) 
→ Hold frames untuk stability

Dengan cara ini, sistem dapat menyesuaikan bentuk mulut avatar secara real-time berdasarkan intensitas suara pengguna, namun tetap stabil dan tidak terlalu cepat berubah.

4. Idle State Machine
Menentukan perilaku avatar saat pengguna tidak aktif (idle).

User Idle? 
→ Timer countdown 
→ Sequential states 
   (NORMAL → LOOK_LEFT → NORMAL → LOOK_RIGHT → ...) 
→ Smooth sway interpolation

Saat tidak ada input dari pengguna, avatar akan bergerak perlahan ke kiri dan kanan agar tetap terlihat hidup, menggunakan interpolasi yang halus antarposisi.

Performance Optimization

1. Frame Skipping - Process MediaPipe setiap N frames untuk efisiensi
2. Result Caching - Reuse hasil deteksi terakhir yang valid
3. Lazy Loading - Assets dimuat sekali saat startup
4. Efficient Overlay - ROI-based image composition
5. Memory Management - Proper resource cleanup

Target Performance

1. FPS: 30-60 fps (tergantung hardware)
2. Latency: <50ms untuk gesture response
3. RAM Usage: sekitar 500MB dengan video effects loaded

Author
Kadek Candra Dwi Yanti
NRP: 5024231067
Mata Kuliah: Pengolahan Citra Video
Tahun: 2025

License
Project ini dibuat untuk keperluan edukasi sebagai bagian dari tugas Pengolahan Citra Video.
