import easyocr
import cv2
from pdf2image import convert_from_path
import re
import os
import numpy as np


# Inicializa el lector con soporte para español
reader = easyocr.Reader(['es'], gpu=False)

def extract_text_from_file(file_path: str) -> str:
    """
    Convierte un PDF o imagen a texto usando EasyOCR.
    """
    text = ""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        # Convierte el PDF a imágenes
        pages = convert_from_path(file_path, dpi=300)
        for page in pages:
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            results = reader.readtext(img, detail=0)
            text += " ".join(results) + " "
    else:
        # Carga imagen directamente
        img = cv2.imread(file_path)
        results = reader.readtext(img, detail=0)
        text = " ".join(results)

    return text.lower()

def validate_identity_document(file_path: str, datos_input: dict) -> dict:
    """
    Compara los datos extraídos del documento con los proporcionados por el usuario.
    """
    texto_extraido = extract_text_from_file(file_path)

    numero = datos_input.get("numero", "").lower()
    nombre = datos_input.get("nombre", "").lower()
    apellido = datos_input.get("apellido", "").lower()

    resultado = {
        "numero_encontrado": numero in texto_extraido,
        "nombre_encontrado": nombre in texto_extraido,
        "apellido_encontrado": apellido in texto_extraido,
        "texto_extraido": texto_extraido  # opcional: para debugging
    }

    return resultado

input_datos = {
    "numero": "01101101 0110000",
    "nombre": "MARTÍN FRANCISCO",
    "apellido": "HOLA"
}

resultado = validate_identity_document("cedula.pdf", input_datos)
print(resultado)

