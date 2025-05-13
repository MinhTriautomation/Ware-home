import cv2
import pytesseract
import sys
from PIL import Image
import traceback

# Cấu hình đường dẫn Tesseract (hãy cập nhật đường dẫn cho phù hợp với máy của bạn)
TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    print(f"OCR Module: Đã cấu hình Tesseract tại: {TESSERACT_CMD_PATH}")
except pytesseract.TesseractNotFoundError:
    print(f"OCR Module ERROR: Không tìm thấy Tesseract tại {TESSERACT_CMD_PATH}. Vui lòng kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"OCR Module ERROR: Lỗi cấu hình Tesseract: {e}")


def extract_and_parse_ocr(image_frame, lang="vie", key_list=["STT", "Name", "Address", "ID", "Product"]):
    """
    Trích xuất văn bản từ khung hình và phân tích các cặp key-value từ kết quả OCR.

    Args:
        image_frame (np.ndarray): Khung hình đầu vào (định dạng OpenCV).
        lang (str): Mã ngôn ngữ cho Tesseract (mặc định: "vie").
        key_list (list): Danh sách các key cần tìm trong văn bản. Ví dụ: ["ID", "Name"].

    Returns:
        dict: Từ điển chứa các cặp key-value được trích xuất.
              Trả về dict rỗng nếu không tìm thấy dữ liệu hợp lệ.
    """
    ocr_dict = {}

    if image_frame is None:
        print("OCR Module: Nhận khung hình là None.")
        return ocr_dict

    try:
        # Tiền xử lý: chuyển đổi sang ảnh xám và chuyển thành ảnh nhị phân với Otsu
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_for_ocr = binary_img

        # Chuyển đổi ảnh cho Tesseract (RGB từ ảnh nhị phân đã chuyển đổi từ xám)
        img_pil = Image.fromarray(cv2.cvtColor(img_for_ocr, cv2.COLOR_GRAY2RGB))

        # Thực hiện OCR với mode psm=6 (1 block text)
        custom_config = r'--psm 6'
        data = pytesseract.image_to_data(img_pil, lang=lang, output_type=pytesseract.Output.DICT, config=custom_config)

        n_boxes = len(data['level'])
        if n_boxes == 0:
            print("OCR Module: Không tìm thấy bounding box văn bản.")
            return ocr_dict

        word_data_list = []
        for i in range(n_boxes):
            level = data['level'][i]
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            # Chỉ xử lý các word có conf hợp lệ và không rỗng tại cấp độ word
            if level == 5 and conf > 0 and text:
                word_data_list.append({
                    'text': text,
                    'box': (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
                    'conf': conf,
                    'block_num': data['block_num'][i],
                    'par_num': data['par_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                })

        # Sắp xếp các word theo block, paragraph, line và vị trí ngang
        word_data_list.sort(key=lambda item: (item['block_num'], item['par_num'], item['line_num'], item['box'][0]))

        current_key = None
        current_value_words = []
        last_processed_line_key = None

        for word_data in word_data_list:
            text = word_data['text']
            box = word_data['box']
            line_key = (word_data['block_num'], word_data['par_num'], word_data['line_num'])

            # Nếu chuyển dòng, kết thúc thu thập value cho key hiện tại
            if current_key is not None and last_processed_line_key is not None and line_key != last_processed_line_key:
                extracted_value = " ".join(current_value_words).strip()
                if extracted_value:
                    ocr_dict[current_key] = extracted_value
                current_key = None
                current_value_words = []
            last_processed_line_key = line_key

            # Nếu chưa có key cho dòng hiện tại, kiểm tra xem từ này có phải key không
            if current_key is None:
                clean_text = text.replace(':', '')
                if clean_text in key_list:
                    current_key = clean_text
                    current_value_words = []
            # Nếu đã có key, thu thập từ thành value
            elif current_key is not None:
                current_value_words.append(text)

        # Nếu còn key chưa được xử lý sau vòng lặp
        if current_key is not None:
            extracted_value = " ".join(current_value_words).strip()
            if extracted_value:
                ocr_dict[current_key] = extracted_value

    except pytesseract.TesseractNotFoundError:
        print("OCR Module ERROR: Không tìm thấy Tesseract. Kiểm tra lại pytesseract.pytesseract.tesseract_cmd.")
    except Exception as e:
        print(f"OCR Module ERROR: Lỗi xử lý OCR hoặc phân tích bounding box: {e}")
        traceback.print_exc()

    print(f"OCR Module: Kết quả OCR: {ocr_dict}")
    return ocr_dict


