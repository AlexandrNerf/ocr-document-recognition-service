import base64

import cv2
import fitz
import numpy as np


def encode_image(image_data: np.ndarray) -> str:
    img_bytes = image_data.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')


def decode_pdf(file_base64: str, dpi=200) -> np.ndarray:
    """Декодирование PDF в numpy массив"""
    pdf_bytes = base64.b64decode(file_base64)
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    page = doc.load_page(0)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    doc.close()
    return img_array


def decode_image(file_data: tuple[str, str]) -> np.ndarray:
    """Декодирование изображения или PDF"""
    file_base64, file_format = file_data
    if file_format == 'image':
        img_bytes = base64.b64decode(file_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        return image
    elif file_format == 'pdf':
        return decode_pdf(file_base64)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
