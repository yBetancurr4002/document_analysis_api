
from utils.extract_text_from_file import extract_text_from_file
from utils.normalize import normalize
from utils.format_document import format_document_number

def validate_school_certificate(file_path: str, datos_input: dict) -> dict:
    """
    Valida el certificado del colegio colombiano verificando:
    - nombres del admitido
    - número de documento
    - nombre de la institución
    - título obtenido
    """
    texto_extraido = extract_text_from_file(file_path)
    texto_normalizado = normalize(texto_extraido)

    nombres_input = normalize(datos_input.get("nombres", ""))
    documento_input = normalize(datos_input.get("documento", ""))
    institucion_input = normalize(datos_input.get("nombre_institucion", ""))
    titulo_input = normalize(datos_input.get("titulo_obtenido", ""))

    # Normaliza números encontrados en el texto
    documentos_extraidos = format_document_number(texto_normalizado)
    documento_encontrado = documento_input in documentos_extraidos

    resultado = {
        "nombre_encontrado": all(n in texto_normalizado for n in nombres_input.split()),
        "documento_encontrado": documento_encontrado,
        "institucion_encontrada": institucion_input in texto_normalizado,
        "titulo_encontrado": titulo_input in texto_normalizado
    }

    resultado["es_valido"] = all(resultado.values())
    resultado["texto_extraido"] = texto_extraido  # útil para depuración

    return resultado
