import os

from utils.extract_text_from_file import extract_text_from_file
from utils.normalize import normalize

def validate_icfes_document(file_path: str, datos_input: dict) -> dict:
    """
    Valida si el número SNP ICFES, nombre y apellido están presentes en el documento.
    """
    if not os.path.exists(file_path):
        return {"error": f"El archivo {file_path} no fue encontrado."}

    texto_extraido = extract_text_from_file(file_path)
    texto_normalizado = normalize(texto_extraido)

    snp = normalize(datos_input.get("Número de registro", ""))
    identificacion = normalize(datos_input.get("Identificación", ""))
    nombres = normalize(datos_input.get("nombre", ""))

    snp_encontrado = snp in texto_normalizado
    documento_encontrado = identificacion in texto_normalizado
    nombres_encontrado = nombres in texto_normalizado
    is_valid = snp_encontrado and documento_encontrado

    result = {
        "snp": snp_encontrado,
        "documento": documento_encontrado,
        "nombres": nombres_encontrado,
        "resultado": is_valid,
        "texto_extraido": texto_normalizado
    }

    if is_valid:
        return {
            "resultado": result["resultado"]
        }



    return result
