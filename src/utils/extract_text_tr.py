from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, img in enumerate(images):
        img_path = f'temp_page_{i}.png'
        img.save(img_path, 'PNG')
        image_paths.append(img_path)

    return image_paths

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"No se pudo cargar la imagen {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.medianBlur(thresh, 3)

    preprocessed_path = f'preprocessed_{os.path.basename(image_path)}'
    cv2.imwrite(preprocessed_path, blur)

    return preprocessed_path

def extract_text_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    full_text = ""

    for img_path in images:
        preprocessed_img = preprocess_image(img_path)
        text = pytesseract.image_to_string(Image.open(preprocessed_img), lang='spa')
        full_text += text + "\n"

        os.remove(img_path)
        os.remove(preprocessed_img)

    return full_text
