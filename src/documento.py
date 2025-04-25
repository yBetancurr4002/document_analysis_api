from utils.format_document import format_document_number
from utils.format_date import detect_birth_date
from utils.extract_text_from_file import extract_text_from_file
from utils.normalize import normalize

def validate_identity_document(file_path: str, datos_input: dict) -> dict:
    """
    Compara los datos extraídos del documento con los proporcionados por el usuario.
    fecha_nacimiento: 12 MAR 2000 o 19-NOV-1991
    Sexo: (M, F),
    lugar_nacimiento: MEDELLIN

    """
    texto_extraido = extract_text_from_file(file_path)
    texto_normalizado = normalize(texto_extraido)

    numero_input = normalize(datos_input.get("numero", ""))
    sexo_input = normalize(datos_input.get("sexo", ""))
    fecha_input = normalize(detect_birth_date(datos_input.get("fecha_nacimiento", "")))
    lugar_input = normalize(datos_input.get("lugar_nacimiento", ""))

    numeros_normalizados = format_document_number(texto_normalizado)

    numero = numero_input in numeros_normalizados
    sexo = sexo_input in texto_normalizado
    fecha = fecha_input in texto_normalizado
    lugar = lugar_input in texto_normalizado

    es_valido = numero and sexo and fecha and lugar

    resultado = {
        "numero_encontrado": numero,
        "sexo_encontrado": sexo,
        "fecha_nacimiento_encontrada": fecha,
        "lugar_nacimiento_encontrado": lugar,
        "es_valido": es_valido,
        "texto": texto_normalizado
    }

    if es_valido:
        return {
            "result": es_valido
        }


    return resultado
