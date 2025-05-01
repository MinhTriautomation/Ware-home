import cv2
import serial
import time
import pytesseract
import sys
import numpy as np
from collections import Counter

from qr_processor_module import load_yolo_model, process_qr_with_yolo
from ocr_processor_module import extract_and_parse_ocr

YOLO_MODEL_PATH = r"D:\NCKH NHOM 2\yolov8s\Yolov8s\results\weights\last.pt"

TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

SERIAL_PORT = "COM3"
BAUD_RATE = 9600

CAMERA_ID = 1

YOLO_CONFIDENCE_THRESHOLD = 0.6

COLLECTION_FRAME_COUNT = 20

MAJORITY_THRESHOLD = 10

yolo_model = load_yolo_model(YOLO_MODEL_PATH)
if yolo_model is None:
    print("Không thể tải model YOLO. Thoát chương trình.")
    sys.exit()

ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Đã kết nối Serial tới {SERIAL_PORT} với Baud rate {BAUD_RATE}")
    time.sleep(2)
except serial.SerialException as e:
    print(f"Lỗi mở cổng Serial {SERIAL_PORT}: {e}")
    print("Chương trình sẽ chạy nhưng không thể giao tiếp với Arduino.")
    ser = None

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Không thể mở camera với ID {CAMERA_ID}.")
    if ser: ser.close()
    sys.exit()

print("Camera đã mở thành công. Nhấn 'q' để thoát.")

collection_state = 0
frame_counter = 0
collected_outcomes = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận khung (stream end?). Thoát ...")
        break

    qr_data = None
    qr_data_list = process_qr_with_yolo(frame, yolo_model, YOLO_CONFIDENCE_THRESHOLD)
    if qr_data_list:
        qr_data = qr_data_list[0]

    ocr_data = extract_and_parse_ocr(frame)

    current_frame_outcome = None

    if qr_data is not None and ocr_data:
        qr_id = qr_data.get("ID")
        qr_product = qr_data.get("Product")
        ocr_id = ocr_data.get("ID")

        is_match = False
        if qr_id is not None and ocr_id is not None and qr_id == ocr_id:
            is_match = True
            if qr_product is not None:
                 if qr_product == "Product A":
                     current_frame_outcome = "A"
                 elif qr_product == "Product B":
                     current_frame_outcome = "B"
                 elif qr_product == "Product C":
                     current_frame_outcome = "C"
                 else:
                     current_frame_outcome = "SAI"
            else:
                 current_frame_outcome = "SAI"
        else:
            current_frame_outcome = "SAI"

    if collection_state == 0:
        status_text = "Status: Waiting for QR and OCR"
        if qr_data is not None and ocr_data is not None and current_frame_outcome is not None:
            collection_state = 1
            frame_counter = 0
            collected_outcomes = []
            print("Chuyển sang Trạng thái Thu thập.")

    elif collection_state == 1:
        status_text = f"Status: Collecting... {frame_counter}/{COLLECTION_FRAME_COUNT}"

        if current_frame_outcome is not None:
            collected_outcomes.append(current_frame_outcome)
            frame_counter += 1

        if frame_counter >= COLLECTION_FRAME_COUNT:
            collection_state = 0

            final_sent_data = b"SAI"
            sent_outcome_text = "SAI"

            if collected_outcomes:
                outcome_counts = Counter(collected_outcomes)
                most_frequent_item = outcome_counts.most_common(1)

                if most_frequent_item:
                    most_frequent_outcome = most_frequent_item[0][0]
                    most_frequent_count = most_frequent_item[0][1]
                    print(f"Collection done. Outcomes: {outcome_counts}. Most frequent: '{most_frequent_outcome}' count: {most_frequent_count}")

                    if most_frequent_count >= MAJORITY_THRESHOLD:
                        if most_frequent_outcome == "A":
                            final_sent_data = b"A"
                        elif most_frequent_outcome == "B":
                            final_sent_data = b"B"
                        elif most_frequent_outcome == "C":
                            final_sent_data = b"C"
                        sent_outcome_text = most_frequent_outcome if most_frequent_outcome in ["A", "B", "C"] else "SAI (Threshold Met)"
                    else:
                        sent_outcome_text = f"SAI (Threshold not met for '{most_frequent_outcome}')"
                        print(f"Kết quả '{most_frequent_outcome}' không đạt ngưỡng {MAJORITY_THRESHOLD}. Gửi SAI.")

                else:
                     sent_outcome_text = "SAI (No valid frames collected)"
                     print("Kết thúc thu thập nhưng không có khung hình hợp lệ nào được thu thập. Gửi SAI.")

            else:
                 sent_outcome_text = "SAI (No frames collected)"
                 print("Kết thúc thu thập nhưng danh sách kết quả rỗng. Gửi SAI.")


            status_text = f"Status: Collection Done. Sent: {final_sent_data.decode()} ({sent_outcome_text})"

            if ser:
                try:
                    ser.write(final_sent_data + b'\n')
                    print(f"Đã gửi '{final_sent_data.decode()}' tới Arduino")
                except serial.SerialException as e:
                    print(f"Lỗi khi gửi dữ liệu qua Serial: {e}")


    if qr_data_list:
         results_draw = yolo_model(frame, verbose=False)
         for r_draw in results_draw:
              for box_draw in r_draw.boxes:
                   conf_draw = float(box_draw.conf[0])
                   cls_id_draw = int(box_draw.cls[0])
                   if conf_draw > YOLO_CONFIDENCE_THRESHOLD and cls_id_draw == 0:
                        x1, y1, x2, y2 = map(int, box_draw.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    status_color = (128, 128, 128)
    if "Collecting" in status_text:
        status_color = (255, 165, 0)
    elif "Collection Done" in status_text:
        if "Sent: A" in status_text:
            status_color = (0, 255, 0)
        elif "Sent: B" in status_text:
            status_color = (0, 255, 255)
        elif "Sent: C" in status_text:
            status_color = (255, 0, 0)
        else:
            status_color = (0, 0, 255)

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    qr_info_text = f"QR: ID={qr_data.get('ID')}, Prod={qr_data.get('Product')}" if qr_data else "QR: Not detected/decoded"
    ocr_info_text = f"OCR: ID={ocr_data.get('ID')}" if ocr_data and ocr_data.get('ID') else "OCR: ID not found or data missing"
    cv2.putText(frame, qr_info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, ocr_info_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('QR and OCR Comparison', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("Đã đóng kết nối Serial.")

print("Chương trình đã kết thúc.")
