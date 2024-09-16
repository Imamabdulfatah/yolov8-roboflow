import cv2
from ultralytics import YOLO

# Load model YOLOv8 yang sudah terlatih
model = YOLO("runs/detect/train/weights/best.pt")  # Ganti dengan path model best.pt Anda

# Inisialisasi kamera (0 adalah kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera")
    exit()

# Set threshold confidence untuk mengurangi sensitivitas
conf_threshold = 0.4  # Deteksi dengan confidence di bawah 0.5 tidak akan ditampilkan

# Loop untuk stream dari kamera
while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera")
        break

    # Deteksi objek pada frame
    results = model(frame)

    # Variabel untuk menyimpan deteksi dengan confidence tertinggi
    highest_confidence = 0
    best_detection = None

    # Periksa semua deteksi dan simpan yang dengan confidence tertinggi
    for result in results[0].boxes:
        confidence = result.conf[0]  # Confidence level dari deteksi

        # Jika confidence lebih tinggi dari highest_confidence, simpan deteksi ini
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_detection = result

    # Jika ada deteksi dengan confidence tertinggi, cetak informasinya
    if best_detection and highest_confidence > conf_threshold:
        # Ambil koordinat bounding box dari deteksi terbaik
        x1, y1, x2, y2 = map(int, best_detection.xyxy[0])

        # Hitung ukuran kotak (bounding box) dan koordinat center
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        ptx = x1  # ptx adalah koordinat X pojok kiri atas
        pty = y1  # pty adalah koordinat Y pojok kiri atas

        # Cetak informasi bounding box dengan confidence tertinggi di terminal
        print(f"Deteksi terbaik: {model.names[int(best_detection.cls)]}, Confidence: {highest_confidence:.2f}")
        print(f"Ukuran kotak - Lebar: {width}, Tinggi: {height}, Center: ({center_x}, {center_y})")
        print(f"Koordinat pojok kiri atas - ptx: {ptx}, pty: {pty}")

        # Tampilkan kotak deteksi terbaik di frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(best_detection.cls)]} {highest_confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame yang sudah diannotasi
    cv2.imshow('YOLOv8 Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()



