import cv2
import pytesseract



def extract_and_parse_ocr(image_frame, lang="vie"):
    """
    Trích xuất văn bản từ khung ảnh cv2 bằng OCR và phân tích thành từ điển.

    Args:
        image_frame (np.ndarray): Khung ảnh từ camera (đối tượng ảnh cv2).
        lang (str): Ngôn ngữ cho Tesseract OCR (mặc định là 'vie' - Tiếng Việt).

    Returns:
        dict: Từ điển chứa dữ liệu từ OCR (key: value).
              Trả về từ điển rỗng nếu không trích xuất được văn bản hoặc lỗi.
    """
    ocr_dict = {}
    if image_frame is None:
        print("OCR Module: Khung ảnh đầu vào là None.")
        return ocr_dict

    try:
        # Chuyển ảnh sang ảnh xám để OCR tốt hơn
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)


        # Sử dụng Tesseract để nhận dạng văn bản
        text = pytesseract.image_to_string(gray, lang=lang)

        # Phân tích văn bản thành từ điển key-value
        # Giả định định dạng là "Key: Value" trên mỗi dòng
        for line in text.strip().split("\n"):
            if ": " in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key: # Đảm bảo key không rỗng
                         ocr_dict[key] = value
            elif ":" in line: # Xử lý trường hợp chỉ có ":"
                 parts = line.split(":", 1)
                 if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key:
                         ocr_dict[key] = value

    except pytesseract.TesseractNotFoundError:
        print("OCR Module ERROR: Không tìm thấy Tesseract. Hãy kiểm tra pytesseract.pytesseract.tesseract_cmd")
    except Exception as e:
        print(f"OCR Module ERROR: Lỗi trong quá trình OCR hoặc phân tích: {e}")

    return ocr_dict

