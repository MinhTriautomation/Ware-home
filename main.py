import cv2
import serial
import time
import pytesseract # Cần import để cấu hình đường dẫn Tesseract
import sys # Để thoát chương trình
import numpy as np

# Import các module tự tạo
from qr_processor_module import load_yolo_model, process_qr_with_yolo
from ocr_processor_module import extract_and_parse_ocr

# --- Cấu hình ---
# Đường dẫn đến tệp model YOLOv8 của bạn (.pt)
YOLO_MODEL_PATH = r"D:\NCKH NHOM 2\yolov8s\Yolov8s\results\weights\last.pt" # <<< Cần thay đổi đường dẫn này

# Đường dẫn đến tệp thực thi Tesseract OCR
# Hãy thay đổi đường dẫn này cho phù hợp với cài đặt của bạn
TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # <<< Cần thay đổi đường dẫn này
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

SERIAL_PORT = "COM3" # <<< Cần thay đổi cổng COM này
BAUD_RATE = 9600

# ID của camera (0 cho camera mặc định)
CAMERA_ID = 0

YOLO_CONFIDENCE_THRESHOLD = 0.6 # Có thể điều chỉnh

# --- Khởi tạo ---
# Tải model YOLO
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
if yolo_model is None:
    print("Không thể tải model YOLO. Thoát chương trình.")
    sys.exit()

# Mở kết nối Serial đến Arduino
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) # timeout=1s
    print(f"Đã kết nối Serial tới {SERIAL_PORT} với Baud rate {BAUD_RATE}")
    time.sleep(2) # Đợi Arduino reset sau khi kết nối Serial
except serial.SerialException as e:
    print(f"Lỗi mở cổng Serial {SERIAL_PORT}: {e}")
    print("Chương trình sẽ chạy nhưng không thể giao tiếp với Arduino.")
    ser = None # Đảm bảo ser là None nếu mở thất bại

# Mở camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Không thể mở camera với ID {CAMERA_ID}.")
    if ser: ser.close()
    sys.exit()

print("Camera đã mở thành công. Nhấn 'q' để thoát.")

# --- Vòng lặp chính ---
while True:
    # Đọc khung ảnh từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận khung (stream end?). Thoát ...")
        break

    # Chúng ta chỉ lấy mã QR đầu tiên được phát hiện và giải mã thành công
    qr_data = None
    qr_data_list = process_qr_with_yolo(frame, yolo_model, YOLO_CONFIDENCE_THRESHOLD)
    if qr_data_list:
        qr_data = qr_data_list[0] # Lấy dữ liệu của mã QR đầu tiên tìm thấy

    # --- Xử lý OCR (Sử dụng Module 1) ---
    # extract_and_parse_ocr trả về một từ điển dữ liệu OCR
    ocr_data = extract_and_parse_ocr(frame)

    # --- Logic So sánh và Gửi Serial ---
    # Ban đầu, không có dữ liệu nào để gửi
    sent_data = None
    status_text = "Status: Waiting for data" # Trạng thái ban đầu hiển thị trên ảnh
    is_match = False # Cờ báo hiệu ID có khớp hay không (dùng cho hiển thị và debug)


    # CHỈ so sánh và gửi dữ liệu khi CẢ QR DATA và OCR DATA đều có
    if qr_data is not None and ocr_data:
        # Cả hai dữ liệu đều có, tiến hành lấy ID và Product để so sánh
        qr_id = qr_data.get("ID")
        qr_product = qr_data.get("Product") # Lấy Product từ QR

        # Lấy giá trị 'ID' từ dữ liệu OCR (sử dụng .get để an toàn)
        ocr_id = ocr_data.get("ID")

        print(f"Dữ liệu lấy được: QR={{'ID': '{qr_id}', 'Product': '{qr_product}'}}, OCR={{'ID': '{ocr_id}'}}") # Debug chi tiết

        # So sánh giá trị ID chỉ khi cả hai đều tồn tại (khác None) VÀ trùng khớp
        if qr_id is not None and ocr_id is not None and qr_id == ocr_id:
            # IDs KHỚP!
            is_match = True # Cờ báo hiệu ID đã khớp

            # Bây giờ, kiểm tra giá trị của Product từ QR để xác định dữ liệu gửi
            if qr_product is not None:

                if qr_product == "Product A":
                    sent_data = b"A"
                    status_text = "Status: MATCH ID & Product A (A)"
                    print("Kết quả: ID KHỚP! Sản phẩm: Product A. Gửi A.")
                elif qr_product == "Product B":
                    sent_data = b"B"
                    status_text = "Status: MATCH ID & Product B (B)"
                    print("Kết quả: ID KHỚP! Sản phẩm: Product B. Gửi B.")
                elif qr_product == "Product C": # Thêm trường hợp cho Product C
                     sent_data = b"C"
                     status_text = "Status: MATCH ID & Product C (C)"
                     print("Kết quả: ID KHỚP! Sản phẩm: Product C. Gửi C.")
                # Thêm các elif khác cho các loại Product khác nếu cần
                else:
                    # Nếu key 'Product' tồn tại nhưng giá trị không khớp với A, B, C
                    sent_data = b"SAI" # Mặc định gửi SAI
                    status_text = "Status: MATCH ID - Unknown Product (SAI)"
                    print(f"Kết quả: ID KHỚP! Sản phẩm không xác định '{qr_product}'. Gửi SAI.")
            else:
                # Nếu key 'Product' bị thiếu trong dữ liệu QR, MẶC DÙ ID KHỚP
                sent_data = b"SAI" # Mặc định gửi SAI nếu thiếu Product
                status_text = "Status: MATCH ID - Product Missing (SAI)"
                print("Kết quả: ID KHỚP! Nhưng thiếu dữ liệu Product trong QR. Gửi SAI.")

        else:
            # IDs KHÔNG KHỚP (hoặc một trong hai ID bị thiếu trong dữ liệu đã lấy)
            is_match = False # Cờ báo hiệu ID không khớp
            sent_data = b"SAI" # Gửi SAI
            status_text = "Status: NO MATCH ID (SAI)"
            print("Kết quả: KHÔNG KHỚP ID hoặc thiếu dữ liệu ID để so sánh!")

        # --- Gửi dữ liệu đến Arduino qua Serial ---
        # Gửi dữ liệu CHỈ KHI ser đã kết nối VÀ sent_data đã được đặt (là b"A", b"B", b"C", hoặc b"SAI")
        if ser and sent_data is not None:
            try:
                ser.write(sent_data + b'\n') # Gửi dữ liệu, thêm ký tự xuống dòng '\n'
                print(f"Đã gửi '{sent_data.decode()}' tới Arduino")
            except serial.SerialException as e:
                print(f"Lỗi khi gửi dữ liệu qua Serial: {e}")
                # Có thể thêm logic thử kết nối lại ở đây nếu cần
    else:
        # Thiếu dữ liệu (QR hoặc OCR hoặc cả hai), KHÔNG GỬI GÌ CẢ
        # sent_data vẫn là None như đã khởi tạo
        status_text = "Status: Waiting for data (Missing QR or OCR)"

    if qr_data_list:
         results_draw = yolo_model(frame, verbose=False)
         for r_draw in results_draw:
              for box_draw in r_draw.boxes:
                  conf_draw = float(box_draw.conf[0])
                  cls_id_draw = int(box_draw.cls[0])
                  # Sử dụng cùng ngưỡng và class ID như khi xử lý
                  if conf_draw > YOLO_CONFIDENCE_THRESHOLD and cls_id_draw == 0: # <<< Cần kiểm tra class ID/name thực tế của model bạn
                       x1, y1, x2, y2 = map(int, box_draw.xyxy[0].tolist())
                       # Vẽ hình chữ nhật
                       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá

    # Hiển thị trạng thái trên ảnh
    # Màu sắc tùy thuộc vào trạng thái cuối cùng (MATCH ID + Product A/B/C hay NO MATCH ID hay SAI hay Waiting)
    status_color = (128, 128, 128) # Xám mặc định (Waiting)
    if "MATCH ID & Product A" in status_text:
        status_color = (0, 255, 0) # Xanh lá cây
    elif "MATCH ID & Product B" in status_text:
        status_color = (0, 255, 255) # Vàng
    elif "MATCH ID & Product C" in status_text:
        status_color = (255, 0, 0) # Xanh dương
    elif "NO MATCH ID" in status_text or "Unknown Product" in status_text or "Product Missing" in status_text:
        status_color = (0, 0, 255) # Đỏ

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Hiển thị dữ liệu QR và OCR đã lấy được (nếu có)
    qr_info_text = f"QR: ID={qr_data.get('ID')}, Prod={qr_data.get('Product')}" if qr_data else "QR: Not detected/decoded"
    # Chỉ hiển thị ID từ OCR theo logic so sánh hiện tại
    ocr_info_text = f"OCR: ID={ocr_data.get('ID')}" if ocr_data and ocr_data.get('ID') else "OCR: ID not found or data missing" # Hiển thị rõ hơn trạng thái OCR ID
    cv2.putText(frame, qr_info_text, (10, 60), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, ocr_info_text, (10, 90), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)


    # Hiển thị khung hình
    cv2.imshow('QR and OCR Comparison', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp ---
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("Đã đóng kết nối Serial.")

print("Chương trình đã kết thúc.")