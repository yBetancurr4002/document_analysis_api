import re as re
def detect_birth_date(text: str) -> str:
    """
    Detecta fechas con formato: 12 MAR 2000, 19-NOV-1991, etc.
    """
    match = re.search(r'\b\d{1,2}[ -/]?[A-Z]{3}[ -/]?\d{2,4}\b', text.upper())
    return match.group(0) if match else ''