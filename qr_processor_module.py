
"""
QR Code Detection and Parsing using YOLOv8 and pyzbar

Module này dùng YOLOv8 để phát hiện vùng chứa QR code trong ảnh, sau đó sử dụng
pyzbar để giải mã nội dung QR code, và phân tích nội dung đó thành dạng key-value.

Yêu cầu:
- Model YOLOv8 định dạng .pt huấn luyện để phát hiện QR code.
- Thư viện ultralytics, OpenCV, và pyzbar.
"""

import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO


def load_yolo_model(model_path):
    """
    Tải model YOLOv8 từ tệp .pt.

    Args:
        model_path (str): Đường dẫn đến tệp model .pt.

    Returns:
        YOLO: Đối tượng model YOLO đã tải thành công, hoặc None nếu lỗi.
    """
    try:
        model = YOLO(model_path)
        print(f"QR Module: Đã tải model YOLO từ {model_path}")
        return model
    except Exception as e:
        print(f"QR Module ERROR: Không thể tải model YOLO từ {model_path}: {e}")
        return None


def process_qr_with_yolo(image_frame, yolo_model, confidence_threshold=0.5):
    """
    Phát hiện và giải mã QR Code từ ảnh đầu vào sử dụng YOLOv8 và pyzbar.

    Args:
        image_frame (np.ndarray): Ảnh đầu vào (OpenCV image).
        yolo_model (YOLO): Model YOLO đã được tải bằng hàm `load_yolo_model`.
        confidence_threshold (float): Ngưỡng độ tin cậy cho phát hiện QR.

    Returns:
        list[dict]: Danh sách các dictionary chứa thông tin đã giải mã từ các QR code.
                    Mỗi dictionary có dạng {Key1: Value1, Key2: Value2, ...}.
    """
    qr_data_list = []

    if image_frame is None or yolo_model is None:
        return qr_data_list

    try:
        results = yolo_model(image_frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                if conf > confidence_threshold and cls_id == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    h, w = image_frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    padding = 5
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(w, x2 + padding)
                    y2_pad = min(h, y2 + padding)

                    qr_roi = image_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    qr_codes_decoded = decode(qr_roi)

                    if qr_codes_decoded:
                        qr_data_bytes = qr_codes_decoded[0].data
                        try:
                            qr_data_str = qr_data_bytes.decode("utf-8")
                            qr_dict = {}

                            for item in qr_data_str.split(", "):
                                parts = item.split(":", 1)
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    value = parts[1].strip()
                                    if key:
                                        qr_dict[key] = value

                            if qr_dict:
                                qr_data_list.append(qr_dict)

                        except UnicodeDecodeError:
                            print("QR Module ERROR: Không thể giải mã QR (không phải UTF-8).")
                        except Exception as e:
                            print(f"QR Module ERROR: Lỗi phân tích dữ liệu QR: {e}")

    except Exception as e:
        print(f"QR Module ERROR: Lỗi xử lý YOLO hoặc pyzbar: {e}")

    return qr_data_list
