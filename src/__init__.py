from src.documento import validate_identity_document
from icfes import validate_icfes_document
from acta_grado import validate_school_certificate
from certificado_electoral import validate_electoral_certificate

"""
DOCUMENTO
"""

input_datos = {
    "numero": "1021669298",
    "sexo": "F",
    "fecha_nacimiento": "02--NOV-2005",
    "lugar_nacimiento": "BOGOTA"
}

resultado = validate_identity_document("files/cedula2.pdf", input_datos)
print("Resultado validación CÉDULA:")
print(resultado)


"""
ICFES
"""


# Datos de entrada
input_datos_icfes = {
    "snp": "AC201910678672",
    "identificacion": "1061809206",
    "nombres": "GARCIA PINEDA VALENTINA"
}

# Ruta al documento a validar (PDF o imagen del resultado ICFES)
ruta_archivo = "files/resultado-icfes.pdf"

# Ejecutar la validación
resultado_icfes = validate_icfes_document(ruta_archivo, input_datos_icfes)

# Mostrar el resultado
print("Resultado validación ICFES:")
print(resultado_icfes)


"""
ACTA GRADO | CONSTANCIA
"""
ruta_acta = "files/4_cert_col.pdf"
input_datos_col = {
    "nombres": "AREVALO MUÑOZ MONICA ALEJANDRA",
    "documento": "1021669298",
    "nombre_institucion": "COLEGIO ATANASIO GIRARDOT IED",
    "titulo_obtenido": ""
}

# Mostrar el resultado
resultado_acta = validate_school_certificate(ruta_acta, input_datos_col)

print("Resultado validación acta:")
print(resultado_acta)


"""
CERTIFICADO ELECTORAL
"""
ruta_electoral = "files/Certificado_Electora.pdf"
input_datos_electoral = {
    "nombres": "AREVALO MUÑOZ MONICA ALEJANDRA",
    "documento": "1021669298",
    "fecha": "13 DE MARZO 2022"
}

resultado_electoral = validate_electoral_certificate(ruta_electoral, input_datos_electoral)

print("Resultado validación ELECTORAL:")
print(resultado_electoral)