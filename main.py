import cv2
import serial
import time
import pytesseract
import sys
import numpy as np
from collections import Counter

from qr_processor_module import load_yolo_model, process_qr_with_yolo
from ocr_processor_module import extract_and_parse_ocr

YOLO_MODEL_PATH = r"D:\NCKH NHOM 2\full code\last.pt"

# TESSERACT_CMD_PATH được cấu hình trong ocr_processor_module.py
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SERIAL_PORT = "COM6"
BAUD_RATE = 115200

CAMERA_ID = 2

YOLO_CONFIDENCE_THRESHOLD = 0.6

COLLECTION_FRAME_COUNT = 10
MAJORITY_THRESHOLD = 5

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
except Exception as e:
     print(f"Lỗi không xác định khi mở cổng Serial: {e}")
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
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        print("Không thể nhận khung (stream end?). Thoát ...")
        break

    qr_data = None
    try:
        qr_data_list = process_qr_with_yolo(frame, yolo_model, YOLO_CONFIDENCE_THRESHOLD)
        if qr_data_list:
            qr_data = qr_data_list[0]
    except Exception as e:
        print(f"Lỗi khi xử lý QR: {e}")
        qr_data = None

    ocr_data = {}
    try:
         ocr_data = extract_and_parse_ocr(frame)
    except Exception as e:
         print(f"Lỗi khi xử lý OCR: {e}")
         ocr_data = {}

    current_frame_outcome = None

    if qr_data is not None and 'ID' in qr_data and ocr_data is not None and 'ID' in ocr_data:
        qr_id = qr_data.get("ID")
        qr_product = qr_data.get("Product")
        ocr_id = ocr_data.get("ID")

        is_match = False
        if qr_id is not None and ocr_id is not None and str(qr_id).strip() == str(ocr_id).strip():
            is_match = True
            if qr_product is not None:
                 if str(qr_product).strip().upper() == "PRODUCT A":
                     current_frame_outcome = "A"
                 elif str(qr_product).strip().upper() == "PRODUCT B":
                     current_frame_outcome = "B"
                 elif str(qr_product).strip().upper() == "PRODUCT C":
                     current_frame_outcome = "C"
                 else:
                     current_frame_outcome = "SAI"
                     print(f"ID khớp ({qr_id}) nhưng Product từ QR không hợp lệ: '{qr_product}'")
            else:
                 current_frame_outcome = "SAI"
                 print(f"ID khớp ({qr_id}) nhưng không tìm thấy thông tin Product trong dữ liệu QR.")
        else:
            current_frame_outcome = "SAI"
            print(f"ID không khớp hoặc thiếu dữ liệu ID. QR ID: '{qr_id}', OCR ID: '{ocr_id}'")
    else:
         current_frame_outcome = "SAI"

    if collection_state == 0:
        status_text = "Status: Waiting for QR and OCR"
        if qr_data is not None and ocr_data and current_frame_outcome in ["A", "B", "C"]:
            collection_state = 1
            frame_counter = 0
            collected_outcomes = []
            print("Chuyển sang Trạng thái Thu thập.")
            if current_frame_outcome is not None:
                 collected_outcomes.append(current_frame_outcome)
                 frame_counter += 1

    elif collection_state == 1:
        status_text = f"Status: Collecting... {frame_counter}/{COLLECTION_FRAME_COUNT}"

        if current_frame_outcome is not None:
             collected_outcomes.append(current_frame_outcome)
        else:
             collected_outcomes.append("SAI")

        frame_counter += 1

        if frame_counter >= COLLECTION_FRAME_COUNT:
            collection_state = 0

            final_sent_data = b"SAI"
            sent_outcome_text = "SAI (Decision pending)"

            if collected_outcomes:
                outcome_counts = Counter(collected_outcomes)
                most_frequent_item = outcome_counts.most_common(1)

                if most_frequent_item:
                    most_frequent_outcome = most_frequent_item[0][0]
                    most_frequent_count = most_frequent_item[0][1]

                    print(f"Thu thập hoàn tất ({COLLECTION_FRAME_COUNT} khung). Kết quả: {outcome_counts}. Xuất hiện nhiều nhất: '{most_frequent_outcome}' số lần: {most_frequent_count}")

                    if most_frequent_count >= MAJORITY_THRESHOLD:
                        if most_frequent_outcome in ["A", "B", "C"]:
                             final_sent_data = most_frequent_outcome.encode('ascii')
                             sent_outcome_text = most_frequent_outcome
                             print(f"Kết quả phổ biến nhất '{most_frequent_outcome}' đạt ngưỡng {MAJORITY_THRESHOLD}/{COLLECTION_FRAME_COUNT}. Gửi '{most_frequent_outcome}'.")
                        else:
                             final_sent_data = b"SAI"
                             sent_outcome_text = "SAI (Majority SAI)"
                             print(f"Kết quả phổ biến nhất là SAI ({most_frequent_count}/{COLLECTION_FRAME_COUNT}) và đạt ngưỡng. Gửi SAI.")
                    else:
                        final_sent_data = b"SAI"
                        sent_outcome_text = f"SAI (Threshold not met for '{most_frequent_outcome}')"
                        print(f"Kết quả '{most_frequent_outcome}' ({most_frequent_count}/{COLLECTION_FRAME_COUNT}) không đạt ngưỡng {MAJORITY_THRESHOLD}. Gửi SAI.")

                else:
                    final_sent_data = b"SAI"
                    sent_outcome_text = "SAI (Error processing outcomes)"
                    print("Lỗi: Danh sách kết quả thu thập không rỗng nhưng không tìm thấy kết quả phổ biến nhất.")

            else:
                 final_sent_data = b"SAI"
                 sent_outcome_text = "SAI (No valid frames collected)"
                 print("Kết thúc thu thập nhưng không có khung hình hợp lệ nào được thu thập. Gửi SAI.")

            status_text = f"Status: Collection Done. Sent: {final_sent_data.decode()} ({sent_outcome_text})"

            if ser:
                try:
                    ser.write(final_sent_data + b'\n')
                    print(f"Đã gửi '{final_sent_data.decode()}' tới Arduino")
                except serial.SerialException as e:
                    print(f"Lỗi khi gửi dữ liệu qua Serial: {e}")
                except Exception as e:
                    print(f"Lỗi không xác định khi gửi dữ liệu: {e}")

    if yolo_model:
         try:
              results_draw = yolo_model(frame, verbose=False)
              for r_draw in results_draw:
                   if hasattr(r_draw, 'boxes') and r_draw.boxes is not None:
                        for box_draw in r_draw.boxes:
                             conf_draw = float(box_draw.conf[0])
                             cls_id_draw = int(box_draw.cls[0])
                             qr_class_id_in_yolo = 0
                             if conf_draw > YOLO_CONFIDENCE_THRESHOLD and cls_id_draw == qr_class_id_in_yolo:
                                  x1, y1, x2, y2 = map(int, box_draw.xyxy[0].tolist())
                                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

         except Exception as e:
              print(f"Lỗi khi vẽ bounding box YOLO: {e}")

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

    qr_info_text = f"QR: ID={qr_data.get('ID')}, Prod={qr_data.get('Product')}" if qr_data and 'ID' in qr_data else "QR: Not detected/decoded"
    ocr_info_parts = [f"{k}={v}" for k, v in ocr_data.items()]
    ocr_info_text = "OCR: " + ", ".join(ocr_info_parts) if ocr_info_parts else "OCR: Data not found"

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
