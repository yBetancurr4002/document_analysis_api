from unidecode import unidecode


# normaliza el texto
def normalize(text: str) -> str:
    """Normaliza el texto quitando tildes y convirtiendo a minúsculas."""
    return unidecode(text.strip().lower()) if text else ""

