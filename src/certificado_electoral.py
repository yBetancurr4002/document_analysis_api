from utils.format_document import format_document_number
from utils.format_date import detect_birth_date
from utils.extract_text_from_file import extract_text_from_file
from utils.normalize import normalize

def validate_electoral_certificate(file_path: str, datos_input: dict) -> dict:
    """
    Compara los datos extraídos del documento con los proporcionados por el usuario.
    input_datos_electoral = {
    "nombres": "AREVALO MUÑOZ MONICA ALEJANDRA",
    "documento": "1021669298",
    "fecha": "13 DE MARZO 2022"
}

    """
    texto_extraido = extract_text_from_file(file_path)
    texto_normalizado = normalize(texto_extraido)

    numero_input = normalize(datos_input.get("numero", ""))
    nombres_input = normalize(datos_input.get("nombres", ""))
    fecha_input = normalize(detect_birth_date(datos_input.get("fecha_nacimiento", "")))

    numeros_normalizados = format_document_number(texto_normalizado)

    numero = numero_input in numeros_normalizados
    nombres = nombres_input in texto_normalizado
    fecha = fecha_input in texto_normalizado

    es_valido = numero and nombres and fecha

    resultado = {
        "numero_encontrado": numero,
        "nombres": nombres,
        "fecha": fecha,
        "es_valido": es_valido,
        "texto": texto_normalizado
    }

    if es_valido:
        return {
            "result": es_valido
        }


    return resultado
