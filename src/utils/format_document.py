import re

def normalize_number(text: str) -> str:
    """Elimina puntos, comas y espacios para comparar números de documento."""
    return re.sub(r'[.,\s]', '', text)

def format_document_number(texto_normalizado):
    # Extraer posibles números de documento del texto (secuencias de dígitos)
    numeros_encontrados = re.findall(r'\d[\d\.,\s]*\d', texto_normalizado)

    # Normalizar cada número encontrado
    numeros_normalizados = [normalize_number(num) for num in numeros_encontrados]

    return numeros_normalizados