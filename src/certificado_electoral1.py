import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re
import os
from pdf2image import convert_from_path


# Configura el path de Tesseract si no está en tu PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def pdf_to_image(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    return np.array(pages[0])[:, :, ::-1]  # Convert to BGR


def crop_to_content(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return image[y:y + h, x:x + w]
    else:
        return image


def extract_text_from_region(image, region):
    x, y, w, h = region
    roi = image[y:y + h, x:x + w]
    config = '--oem 3 --psm 6'
    return pytesseract.image_to_string(roi, lang='spa', config=config).strip()


def extract_data(image):
    height, width = image.shape[:2]

    # Regiones aproximadas (ajustadas más abajo para texto manuscrito)
    """
    regions = {
        "numero": (int(width * 0.1), int(height * 0.60), int(width * 0.8), int(height * 0.08)),
        "nombres": (int(width * 0.1), int(height * 0.70), int(width * 0.8), int(height * 0.10)),
        "fecha": (int(width * 0.1), int(height * 0.82), int(width * 0.8), int(height * 0.08)),
    }
    """
    regions = {
        "numero": (220, 165, 370, 40),  # Cédula No.
        "nombres": (210, 215, 420, 45),  # Nombres y Apellidos
        "fecha": (220, 115, 390, 25),  # Fecha (impresa)
    }

    extracted = {}
    for campo, region in regions.items():
        texto = extract_text_from_region(image, region)
        extracted[campo] = texto

    extracted["numero"] = extract_document_number(extracted["numero"])
    extracted["fecha"] = extract_date(extracted["fecha"])

    return extracted


def extract_document_number(text):
    match = re.search(r'\d{5,}', text)
    return match.group(0) if match else ''


def extract_date(text):
    match = re.search(r'\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚ]{3,9}\s+DE\s+\d{4}', text.upper())
    return match.group(0) if match else ''


def validate_electoral_certificate(path_pdf, datos_input):
    image = pdf_to_image(path_pdf)
    image_cropped = crop_to_content(image)
    extracted = extract_data(image_cropped)

    numero_input = re.sub(r'\D', '', datos_input.get("numero", ""))
    nombres_input = datos_input.get("nombres", "").lower().strip()
    fecha_input = datos_input.get("fecha", "").lower().strip()

    numero_encontrado = numero_input in extracted.get("numero", "")
    nombres_encontrado = nombres_input in extracted.get("nombres", "").lower()
    fecha_encontrada = fecha_input in extracted.get("fecha", "").lower()

    return {
        "numero_encontrado": numero_encontrado,
        "nombres_encontrado": nombres_encontrado,
        "fecha_encontrada": fecha_encontrada,
        "es_valido": numero_encontrado and nombres_encontrado and fecha_encontrada,
        "texto_extraido": extracted
    }
