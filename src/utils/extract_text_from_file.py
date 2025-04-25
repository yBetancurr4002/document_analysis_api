import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr


reader = easyocr.Reader(['es'], gpu=False)
def extract_text_from_file(file_path: str) -> str:
    """
    Convierte un PDF o imagen a texto usando EasyOCR.
    """
    text = ""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        pages = convert_from_path(file_path, dpi=300)
        for page in pages:
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            results = reader.readtext(img, detail=0)
            text += " ".join(results) + " "
    else:
        img = cv2.imread(file_path)
        results = reader.readtext(img, detail=0)
        text = " ".join(results)

    return text.lower()