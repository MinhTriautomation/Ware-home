import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO

def load_yolo_model(model_path):
    """
    Tải model YOLOv8 từ tệp .pt.

    Args:
        model_path (str): Đường dẫn đến tệp model .pt.

    Returns:
        YOLO: Đối tượng model YOLO đã tải, hoặc None nếu lỗi.
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
    Phát hiện QR Code bằng model YOLO, giải mã vùng được phát hiện và phân tích dữ liệu.

    Args:
        image_frame (np.ndarray): Khung ảnh từ camera (đối tượng ảnh cv2).
        yolo_model (YOLO): Đối tượng model YOLO đã được tải.
        confidence_threshold (float): Ngưỡng tin cậy cho phát hiện YOLO.

    Returns:
        list: Danh sách các từ điển, mỗi từ điển chứa dữ liệu từ một QR Code đã giải mã
              thành công. Trả về danh sách rỗng nếu không phát hiện/giải mã được QR.
    """
    qr_data_list = []

    if image_frame is None:
        print("QR Module: Khung ảnh đầu vào là None.")
        return qr_data_list
    if yolo_model is None:
        print("QR Module ERROR: Model YOLO chưa được tải.")
        return qr_data_list

    try:
        # Thực hiện phát hiện với YOLO
        # verbose=False để giảm bớt output từ YOLO trong console
        results = yolo_model(image_frame, verbose=False)

        # Xử lý kết quả phát hiện
        for r in results:
            # Lặp qua các hộp giới hạn được phát hiện
            for box in r.boxes:
                # Lấy class ID, độ tin cậy và tọa độ hộp giới hạn
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist() # Định dạng [x1, y1, x2, y2]


                if conf > confidence_threshold and cls_id == 0: # <<< Cần kiểm tra class ID/name thực tế của model bạn
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Đảm bảo tọa độ nằm trong giới hạn của ảnh
                    h, w = image_frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Cắt vùng ảnh chứa QR Code
                    # Thêm một chút padding xung quanh hộp giới hạn có thể giúp giải mã tốt hơn
                    padding = 5
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(w, x2 + padding)
                    y2_pad = min(h, y2 + padding)

                    qr_roi = image_frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    # Giải mã QR Code trong vùng đã cắt
                    qr_codes_decoded = decode(qr_roi)

                    if qr_codes_decoded:
                        # Giả sử chỉ có 1 mã QR trong vùng ROI, lấy cái đầu tiên
                        qr_data_bytes = qr_codes_decoded[0].data
                        try:
                            # Giải mã bytes sang chuỗi UTF-8
                            qr_data_str = qr_data_bytes.decode("utf-8")
                            # print(f"QR Data String: {qr_data_str}") # Debug: xem chuỗi giải mã được

                            # Phân tích chuỗi QR thành từ điển
                            # Giả định định dạng là "Key: Value, Key2: Value2, ..."
                            qr_dict = {}
                            for item in qr_data_str.split(", "):
                                parts = item.split(":", 1) # Chỉ tách ở dấu ":" đầu tiên
                                if len(parts) >= 2:
                                    key = parts[0].strip()
                                    value = parts[1].strip()
                                    if key: # Đảm bảo key không rỗng
                                        qr_dict[key] = value
                                # else:
                                #     print(f"QR Module: Bỏ qua phần không đúng định dạng: {item}") # Debug

                            if qr_dict: # Chỉ thêm vào danh sách nếu phân tích được dữ liệu
                                qr_data_list.append(qr_dict)
                                print(f"QR Module: Đã giải mã QR: {qr_dict}") # Log dữ liệu giải mã thành công

                        except UnicodeDecodeError:
                            print(f"QR Module ERROR: Không thể giải mã dữ liệu QR (không phải UTF-8?). Dữ liệu thô: {qr_data_bytes}")
                        except Exception as e:
                            print(f"QR Module ERROR: Lỗi khi phân tích dữ liệu QR '{qr_data_str}': {e}")

    except Exception as e:
        print(f"QR Module ERROR: Lỗi trong quá trình xử lý YOLO hoặc pyzbar: {e}")

    return qr_data_list # Trả về danh sách các dữ liệu QR đã giải mã

