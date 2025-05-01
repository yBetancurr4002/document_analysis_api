import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re
import os
from pdf2image import convert_from_path
import pandas as pd
from datetime import datetime
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar ruta de Tesseract (ajustar según instalación)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")



# Diccionario de meses en español para análisis de fechas
MESES_ES = {
    'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04', 'MAY': '05', 'JUN': '06',
    'JUL': '07', 'AGO': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
}


def pdf_a_imagen(ruta_pdf, dpi=300):
    """
    Convierte un PDF a imagen con alta resolución.
    Optimizado para extraer texto manuscrito.
    """
    try:
        logger.info(f"Convirtiendo PDF a imagen: {ruta_pdf}")
        # Usar mayor DPI para mejor calidad en la extracción de texto
        paginas = convert_from_path(ruta_pdf, dpi=dpi)
        if not paginas:
            raise Exception("No se pudo convertir el PDF a imagen")

        # Convertir a formato numpy array que OpenCV pueda procesar
        imagen = np.array(paginas[0])

        # Convertir de RGB a BGR (formato que usa OpenCV)
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            imagen = imagen[:, :, ::-1]

        return imagen
    except Exception as e:
        logger.error(f"Error al convertir PDF a imagen: {str(e)}")
        raise


def preprocesar_imagen(imagen):
    """
    Aplica técnicas de preprocesamiento para mejorar la extracción de texto,
    especialmente para texto manuscrito en certificados electorales.
    """
    try:
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Crear copias para diferentes técnicas de preprocesamiento
        resultados = []

        # 1. Umbralización adaptativa para manejar condiciones de iluminación variables
        thresh1 = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        resultados.append(thresh1)

        # 2. Umbralización Otsu para mejor separación de texto y fondo
        _, thresh2 = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resultados.append(thresh2)

        # 3. Mejora de contraste con ecualización de histograma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        mejorado = clahe.apply(gris)
        _, thresh3 = cv2.threshold(mejorado, 150, 255, cv2.THRESH_BINARY_INV)
        resultados.append(thresh3)

        # Crear imagen combinada (O lógico de todas las técnicas)
        combinada = resultados[0].copy()
        for img in resultados[1:]:
            combinada = cv2.bitwise_or(combinada, img)

        # Operaciones morfológicas para limpiar ruido
        kernel = np.ones((2, 2), np.uint8)
        abierto = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel)

        # Añadir preprocesamiento específico para texto manuscrito
        # Dilatación para conectar componentes de caracteres manuscritos
        kernel_dil = np.ones((2, 2), np.uint8)
        dilatado = cv2.dilate(abierto, kernel_dil, iterations=1)

        return dilatado, gris, mejorado
    except Exception as e:
        logger.error(f"Error en el preprocesamiento de imagen: {str(e)}")
        raise


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # Umbral binario para detectar el texto
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Corrige el ángulo
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def detectar_regiones(imagen, imagen_orig=None):
    """
    Detecta dinámicamente las regiones de interés en el certificado electoral.
    Busca texto clave y define regiones relativas a ese texto.
    """
    try:
        regiones = {}
        alto, ancho = imagen.shape[:2]

        # Obtener todo el texto y posiciones con configuración para documentos
        # Usar un PSM más adecuado para detección de texto en documentos estructurados
        config = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
        datos = pytesseract.image_to_data(imagen, lang='spa', config=config, output_type=Output.DICT)

        # Visualización para debugging (opcional)
        imagen_debug = None
        if imagen_orig is not None:
            imagen_debug = imagen_orig.copy()

        # Localizar identificadores de texto clave
        n_boxes = len(datos['text'])
        cedula_box = None
        nombres_box = None
        electoral_box = None
        fecha_box = None

        # Lista de palabras clave para buscar con mayor tolerancia
        palabras_cedula = ['dula', 'Cédula', 'CEDULA', 'CÉDULA', 'C.C.', 'CC', 'No.', 'DOCUMENTO']
        palabras_nombres = ['Nombres', 'NOMBRES', 'Apellidos', 'APELLIDOS', 'NOMBRE', 'APELLIDO']

        # Mejorar la búsqueda de texto con tolerancia a errores OCR
        for i in range(n_boxes):
            texto = datos['text'][i].strip().upper()
            conf = int(datos['conf'][i])

            # Solo considerar texto con cierta confianza (ajustar según resultados)
            if conf < 40 or not texto:
                continue

            # Visualización para debugging
            if imagen_debug is not None and texto:
                x, y, w, h = datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i]
                cv2.rectangle(imagen_debug, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(imagen_debug, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Buscar textos clave con mayor tolerancia
            if 'CERTIFICADO' in texto and ('ELECTORAL' in texto or 'VOTACION' in texto or 'VOTACIÓN' in texto):
                electoral_box = (datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i])

                # Buscar fecha cerca del título
                fecha_match = re.search(r'\d{1,2}\s+DE\s+[A-ZÁ-Ú]{3,9}\s+DE\s+\d{4}', texto)
                if fecha_match:
                    fecha_box = electoral_box

            # Buscar mención de cédula con mayor tolerancia
            if any(palabra in texto for palabra in palabras_cedula):
                cedula_box = (datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i])
                logger.info(f"Encontrada referencia a cédula: '{texto}' en posición {cedula_box}")

            # Buscar mención de nombres con mayor tolerancia
            if any(palabra in texto for palabra in palabras_nombres):
                nombres_box = (datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i])
                logger.info(f"Encontrada referencia a nombres: '{texto}' en posición {nombres_box}")

            # Buscar posibles fechas directamente
            if re.search(r'\d{1,2}[\s/.-]+\w{3,9}[\s/.-]+\d{4}', texto):
                fecha_box = (datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i])
                logger.info(f"Encontrada posible fecha: '{texto}' en posición {fecha_box}")

        # Si no se encontraron algunas regiones clave, intentar con detección de líneas
        if not cedula_box or not nombres_box:
            # Intentar detectar líneas horizontales que suelen separar campos en documentos
            bordes = cv2.Canny(imagen, 50, 150, apertureSize=3)
            lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=100, minLineLength=ancho * 0.3, maxLineGap=20)

            if lineas is not None and imagen_debug is not None:
                # Ordenar líneas por coordenada Y
                lineas_ordenadas = sorted(lineas, key=lambda x: x[0][1])

                # Visualizar líneas detectadas
                for linea in lineas_ordenadas:
                    x1, y1, x2, y2 = linea[0]
                    cv2.line(imagen_debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Si tenemos suficientes líneas, podemos estimar regiones basadas en ellas
                if len(lineas_ordenadas) >= 3:
                    # Las regiones suelen estar entre líneas consecutivas
                    for i in range(1, min(4, len(lineas_ordenadas))):
                        y_prev = lineas_ordenadas[i - 1][0][1]
                        y_curr = lineas_ordenadas[i][0][1]

                        # Definir regiones basadas en la posición relativa entre líneas
                        if not cedula_box and i == 1:
                            regiones["numero"] = (int(ancho * 0.5), y_prev + 5, int(ancho * 0.4), y_curr - y_prev - 10)
                        elif not nombres_box and i == 2:
                            regiones["nombres"] = (
                            int(ancho * 0.25), y_prev + 5, int(ancho * 0.6), y_curr - y_prev - 10)

        # Definir regiones basadas en texto encontrado
        # Región para número de documento (a la derecha de "Cédula")
        if cedula_box:
            x, y, w, h = cedula_box
            # Ajustar para buscar el número más a la derecha y con un área más amplia
            regiones["numero"] = (x + w, y - 5, ancho - (x + w) - 10, h * 2)
            logger.info(f"Región para número definida en: {regiones['numero']}")
        else:
            # Región de respaldo para número de cédula - ampliada para capturar mejor
            regiones["numero"] = (int(ancho * 0.5), int(alto * 0.3), int(ancho * 0.45), int(alto * 0.15))
            logger.info("Usando región de respaldo para número de documento")

        # Región para nombres (debajo de "Nombres y Apellidos")
        if nombres_box:
            x, y, w, h = nombres_box
            # Ajustar para buscar más abajo del campo nombres
            regiones["nombres"] = (x, y + h, w * 2, h * 2)
            logger.info(f"Región para nombres definida en: {regiones['nombres']}")
        else:
            # Región de respaldo para nombres - ampliada
            regiones["nombres"] = (int(ancho * 0.2), int(alto * 0.4), int(ancho * 0.7), int(alto * 0.15))
            logger.info("Usando región de respaldo para nombres")

        # Región para fecha (dentro de la línea "CERTIFICADO ELECTORAL")
        if fecha_box:
            x, y, w, h = fecha_box
            # Ampliar región para fecha
            regiones["fecha"] = (x, y, w + 200, h * 2)
        elif electoral_box:
            x, y, w, h = electoral_box
            regiones["fecha"] = (x, y, w + 200, h * 2)
        else:
            # Región de respaldo para fecha - ampliada y búsqueda en la parte superior
            regiones["fecha"] = (int(ancho * 0.3), int(alto * 0.1), int(ancho * 0.6), int(alto * 0.2))
            logger.info("Usando región de respaldo para fecha")

        # Guardar imagen de debug si es necesario
        if imagen_debug is not None:
            for nombre, (x, y, w, h) in regiones.items():
                cv2.rectangle(imagen_debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(imagen_debug, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite("debug_regiones.jpg", imagen_debug)
            logger.info("Imagen de debug guardada como 'debug_regiones.jpg'")

        return regiones
    except Exception as e:
        logger.error(f"Error al detectar regiones: {str(e)}")
        # Regiones de respaldo más amplias
        alto, ancho = imagen.shape[:2]
        return {
            "numero": (int(ancho * 0.5), int(alto * 0.3), int(ancho * 0.45), int(alto * 0.15)),
            "nombres": (int(ancho * 0.2), int(alto * 0.4), int(ancho * 0.7), int(alto * 0.15)),
            "fecha": (int(ancho * 0.3), int(alto * 0.1), int(ancho * 0.6), int(alto * 0.2))
        }


def extraer_texto_de_region(imagen, region, imagenes_proc=None):
    """
    Extrae texto de una región específica con procesamiento mejorado.
    Prueba múltiples configuraciones de OCR y técnicas de preprocesamiento.
    """
    try:
        x, y, w, h = region

        # Asegurar que las coordenadas estén dentro de los límites de la imagen
        h_img, w_img = imagen.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        # Extraer región de interés
        roi = imagen[y:y + h, x:x + w]

        # Guardar ROI para debugging
        cv2.imwrite(f"debug_roi_{x}_{y}.jpg", roi)
        logger.info(f"ROI guardada: debug_roi_{x}_{y}.jpg")

        # Preparar lista de imágenes a probar
        imagenes_a_probar = [roi]

        # Si se proporcionaron imágenes preprocesadas, usarlas también
        if imagenes_proc is not None:
            for img_proc in imagenes_proc:
                if img_proc is not None:
                    roi_proc = img_proc[y:y + h, x:x + w]
                    imagenes_a_probar.append(roi_proc)
                    # Guardar ROI procesada para debugging
                    cv2.imwrite(f"debug_roi_proc_{x}_{y}_{len(imagenes_a_probar)}.jpg", roi_proc)

        # Configuraciones OCR para diferentes tipos de texto
        configs = [
            '--oem 3 --psm 6',  # Configuración estándar
            '--oem 3 --psm 4',  # Para texto de una línea
            '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúÁÉÍÓÚñÑ ',
            # Para palabras individuales
            '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',  # Solo para dígitos (útil para cédulas)
            '--oem 3 --psm 3',  # Auto-detección de columnas
            '--oem 3 --psm 11',  # Palabras dispersas
        ]

        # Probar todas las combinaciones de imágenes y configuraciones
        resultados = []

        for i, img in enumerate(imagenes_a_probar):
            for config in configs:
                try:
                    # Aplicar blur gaussiano para suavizar bordes (ayuda con texto manuscrito)
                    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
                    # Aplicar también sharpen para mayor claridad
                    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    img_sharp = cv2.filter2D(img, -1, kernel_sharpen)

                    # Intentar con imagen original, con blur y con sharpening
                    for j, img_test in enumerate([img, img_blur, img_sharp]):
                        texto = pytesseract.image_to_string(img_test, lang='spa', config=config).strip()
                        if texto:  # Solo agregar si se extrajo algún texto
                            resultados.append(texto)
                            logger.debug(f"Texto extraído ({i},{j},{config}): {texto}")
                except Exception as e:
                    logger.debug(f"Error en OCR: {str(e)}")
                    continue

        # Si no se encontró texto, devolver cadena vacía
        if not resultados:
            return ""

        # Devolver el resultado más largo (generalmente el más completo)
        mejor_resultado = max(resultados, key=len)
        logger.info(f"Mejor resultado OCR: {mejor_resultado}")
        return mejor_resultado

    except Exception as e:
        logger.error(f"Error al extraer texto de región: {str(e)}")
        return ""


def extraer_numero_documento(texto):
    """
    Extrae el número de documento del texto con regex mejorado.
    Maneja errores comunes de OCR.
    """
    try:
        # Limpiar texto primero (errores comunes de OCR)
        texto = texto.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')

        logger.info(f"Extrayendo número de documento de: '{texto}'")

        # Probar diferentes patrones
        patrones = [
            r'\d{5,}',  # Al menos 5 dígitos seguidos
            r'\d[\d\s]{5,}',  # Dígitos con posibles espacios
            r'[0-9,.]{5,}',  # Dígitos con posibles comas/puntos
            r'[0-9]{1,3}[,.][0-9]{3}[,.][0-9]{3}',  # Formato con separadores
            r'No\.\s*\d+',  # Número precedido por "No."
            r'C\.?C\.?\s*\d+',  # Número precedido por "C.C."
        ]

        for patron in patrones:
            match = re.search(patron, texto)
            if match:
                # Limpiar el resultado (quitar caracteres que no sean dígitos)
                resultado = re.sub(r'[^0-9]', '', match.group(0))
                if len(resultado) >= 5:  # Longitud razonable para un número de documento
                    logger.info(f"Número de documento encontrado: {resultado}")
                    return resultado

        # Si no se encontró con patrones, intentar extraer cualquier secuencia de dígitos
        todos_numeros = re.findall(r'\d+', texto)
        if todos_numeros:
            # Tomar el número más largo
            mejor_numero = max(todos_numeros, key=len)
            if len(mejor_numero) >= 5:
                logger.info(f"Número de documento extraído (método alternativo): {mejor_numero}")
                return mejor_numero

        logger.warning("No se pudo extraer número de documento")
        return ''
    except Exception as e:
        logger.error(f"Error al extraer número de documento: {str(e)}")
        return ''


def extraer_nombre(texto):
    """
    Extrae nombre del texto con filtrado.
    """
    try:
        logger.info(f"Extrayendo nombre de: '{texto}'")

        # Eliminar números y caracteres especiales, manteniendo solo letras y espacios
        limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', texto)
        # Eliminar espacios extra
        limpio = re.sub(r'\s+', ' ', limpio).strip()

        # Filtrar palabras cortas o que podrían ser ruido
        palabras = limpio.split()
        palabras_filtradas = [p for p in palabras if len(p) > 1]

        resultado = ' '.join(palabras_filtradas)
        logger.info(f"Nombre extraído: '{resultado}'")
        return resultado
    except Exception as e:
        logger.error(f"Error al extraer nombre: {str(e)}")
        return ''


def extraer_fecha(texto):
    """
    Extrae fecha con reconocimiento de patrones mejorado.
    """
    try:
        logger.info(f"Extrayendo fecha de: '{texto}'")

        # Probar diferentes patrones de fecha
        patrones = [
            r'\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚ]{3,9}\s+DE\s+\d{4}',  # 29 DE MAYO DE 2022
            r'\d{1,2}\s+[A-ZÁÉÍÓÚ]{3,9}\s+\d{4}',  # 29 MAYO 2022
            r'\d{1,2}/\d{1,2}/\d{4}',  # 29/05/2022
            r'\d{1,2}-\d{1,2}-\d{4}',  # 29-05-2022
            r'\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚ]+',  # 29 DE MAYO (parcial)
            r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}'  # Formatos numéricos variados
        ]

        for patron in patrones:
            match = re.search(patron, texto.upper())
            if match:
                fecha_texto = match.group(0)
                logger.info(f"Fecha encontrada: {fecha_texto}")
                return fecha_texto

        # Si no se encuentra con los patrones principales, buscar componentes
        # de fecha por separado y tratar de reconstruir
        dia = re.search(r'\b\d{1,2}\b', texto)
        mes_texto = re.search(r'\b(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)[A-Z]*\b', texto.upper())
        año = re.search(r'\b(20|19)\d{2}\b', texto)

        if dia and mes_texto and año:
            fecha_reconstruida = f"{dia.group(0)} {mes_texto.group(0)} {año.group(0)}"
            logger.info(f"Fecha reconstruida: {fecha_reconstruida}")
            return fecha_reconstruida

        logger.warning("No se pudo extraer fecha")
        return ''
    except Exception as e:
        logger.error(f"Error al extraer fecha: {str(e)}")
        return ''


def analizar_fecha(texto_fecha):
    """
    Analiza el texto de fecha y lo convierte a formato estándar.
    """
    try:
        if not texto_fecha:
            return ''

        logger.info(f"Analizando fecha: '{texto_fecha}'")

        # Eliminar 'DE' y limpiar
        texto_fecha = texto_fecha.upper().replace('DE', '').strip()
        partes_fecha = re.split(r'\s+', texto_fecha)

        if len(partes_fecha) >= 3:
            dia = partes_fecha[0].zfill(2)
            nombre_mes = partes_fecha[1][:3]  # Tomar primeras 3 letras
            año = partes_fecha[-1]

            if nombre_mes in MESES_ES:
                mes = MESES_ES[nombre_mes]
                fecha_iso = f"{año}-{mes}-{dia}"
                logger.info(f"Fecha analizada: {fecha_iso}")
                return fecha_iso

        # Intentar formato DD/MM/YYYY
        match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', texto_fecha)
        if match:
            dia, mes, año = match.groups()
            fecha_iso = f"{año}-{mes.zfill(2)}-{dia.zfill(2)}"
            logger.info(f"Fecha analizada (formato numérico): {fecha_iso}")
            return fecha_iso
    except Exception as e:
        logger.error(f"Error al analizar fecha: {str(e)}")

    return ''


def extraer_datos(imagen):
    """
    Extrae todos los datos de la imagen con procesamiento mejorado.
    """
    try:
        # Preprocesar la imagen
        preprocesada, gris, mejorada = preprocesar_imagen(imagen)

        # Guardar imágenes procesadas para debugging
        cv2.imwrite("debug_preprocesada.jpg", preprocesada)
        cv2.imwrite("debug_gris.jpg", gris)
        cv2.imwrite("debug_mejorada.jpg", mejorada)
        logger.info("Imágenes preprocesadas guardadas para debugging")

        # Detectar regiones
        regiones = detectar_regiones(gris, imagen)

        extraido = {}

        # Extraer texto de cada región con múltiples técnicas
        for campo, region in regiones.items():
            texto = extraer_texto_de_region(gris, region, [preprocesada, mejorada])
            extraido[campo] = texto
            logger.info(f"Campo '{campo}' extraído: '{texto}'")

        # Procesar texto extraído
        if 'numero' in extraido:
            extraido["numero"] = extraer_numero_documento(extraido["numero"])

        if 'nombres' in extraido:
            extraido["nombres"] = extraer_nombre(extraido["nombres"])

        if 'fecha' in extraido:
            texto_fecha = extraer_fecha(extraido["fecha"])
            extraido["fecha"] = texto_fecha
            # Opcional: Analizar a formato estándar
            extraido["fecha_iso"] = analizar_fecha(texto_fecha)

        return extraido
    except Exception as e:
        logger.error(f"Error al extraer datos: {str(e)}")
        return {}


def validar_certificado_electoral(ruta_pdf, datos_input):
    """
    Valida un certificado electoral en PDF contra datos de entrada.

    Args:
        ruta_pdf: Ruta al archivo PDF del certificado electoral
        datos_input: Diccionario con datos esperados ("numero", "nombres", "fecha")

    Returns:
        Un diccionario con resultados de validación y texto extraído
    """
    try:
        logger.info(f"Validando certificado electoral: {ruta_pdf}")
        logger.info(f"Datos de entrada para validación: {datos_input}")

        # Convertir PDF a imagen
        imagen = pdf_a_imagen(ruta_pdf)

        if imagen is None:
            return {"error": "No se pudo cargar la imagen del PDF"}, None

        imagen = deskew_image(imagen)

        # Guardar imagen original para referencia
        cv2.imwrite("debug_original.jpg", imagen)
        logger.info("Imagen original guardada como 'debug_original.jpg'")

        # Extraer datos
        extraido = extraer_datos(imagen)
        logger.info(f"Datos extraídos: {extraido}")

        # Limpiar datos de entrada para comparación
        numero_input = re.sub(r'\D', '', datos_input.get("numero", ""))
        nombres_input = datos_input.get("nombres", "").lower().strip()
        fecha_input = datos_input.get("fecha", "").lower().strip()

        # Comparar datos extraídos con datos de entrada
        numero_encontrado = False
        if extraido.get("numero") and numero_input:
            # Permitir coincidencia parcial para números (al menos 5 dígitos consecutivos)
            numero_extraido = extraido.get("numero", "")
            # Comprobar si los últimos 5+ dígitos coinciden (más flexible)
            if len(numero_extraido) >= 5 and len(numero_input) >= 5:
                # Verificar si los últimos dígitos coinciden (más común en cédulas)
                num_digitos = min(len(numero_extraido), len(numero_input))
                if numero_extraido[-num_digitos:] == numero_input[-num_digitos:]:
                    numero_encontrado = True
                # También verificar coincidencia en cualquier parte
                elif numero_input in numero_extraido or numero_extraido in numero_input:
                    numero_encontrado = True

        # Comparación más flexible para nombres
        nombres_extraidos = extraido.get("nombres", "").lower()
        nombres_encontrado = False

        # Si hay un nombre extraído, verificar si contiene partes del nombre de entrada
        if nombres_extraidos and nombres_input:
            # Dividir en palabras para comparación parcial
            palabras_input = [p for p in nombres_input.split() if len(p) > 2]  # Ignorar palabras muy cortas
            palabras_extraidas = nombres_extraidos.split()

            # Verificar si hay coincidencias parciales
            coincidencias = 0
            for palabra in palabras_input:
                if any(palabra in extraida or extraida in palabra for extraida in palabras_extraidas):
                    coincidencias += 1

            # Considerar encontrado si hay suficientes coincidencias
            nombres_encontrado = coincidencias >= min(1, len(palabras_input) // 2)

        # Comparación más flexible para fechas
        fecha_extraida = extraido.get("fecha", "").lower()
        fecha_encontrada = False

        if fecha_extraida and fecha_input:
            # Normalizar formato de fecha para comparación
            fecha_extraida_norm = re.sub(r'[^0-9]', '', fecha_extraida)
            fecha_input_norm = re.sub(r'[^0-9]', '', fecha_input)

            # Verificar si contiene los dígitos de la fecha
            if fecha_input_norm and fecha_extraida_norm:
                # Considerar encontrado si hay suficientes dígitos coincidentes
                dígitos_coincidentes = sum(1 for d in fecha_input_norm if d in fecha_extraida_norm)
                fecha_encontrada = dígitos_coincidentes >= min(4, len(fecha_input_norm) // 2)

        resultado = {
            "numero_encontrado": numero_encontrado,
            "nombres_encontrado": nombres_encontrado,
            "fecha_encontrada": fecha_encontrada,
            "es_valido": numero_encontrado and nombres_encontrado and fecha_encontrada,
            "texto_extraido": extraido
        }

        logger.info(f"Resultados de validación: {resultado}")
        return resultado, imagen

    except Exception as e:
        logger.error(f"Error al validar certificado: {str(e)}")
        return {"error": str(e)}, None


def guardar_debug(imagen, resultado, ruta_salida="debug_resultado.txt"):
    """
    Guarda información de debugging y visualización del resultado.

    Args:
        imagen: Imagen original del certificado
        resultado: Resultado de la validación
        ruta_salida: Ruta para guardar el archivo de texto de debug
    """
    try:
        # Crear una copia de la imagen para visualización
        img_debug = imagen.copy()

        # Obtener datos extraídos
        extraido = resultado.get("texto_extraido", {})

        # Crear texto de resumen
        resumen = []
        resumen.append(f"RESUMEN DE VALIDACIÓN:")
        resumen.append(f"--------------------")
        resumen.append(f"Número encontrado: {resultado.get('numero_encontrado', False)}")
        resumen.append(f"Nombres encontrados: {resultado.get('nombres_encontrado', False)}")
        resumen.append(f"Fecha encontrada: {resultado.get('fecha_encontrada', False)}")
        resumen.append(f"Validación completa: {resultado.get('es_valido', False)}")
        resumen.append(f"--------------------")
        resumen.append(f"DATOS EXTRAÍDOS:")
        resumen.append(f"Número: {extraido.get('numero', '')}")
        resumen.append(f"Nombres: {extraido.get('nombres', '')}")
        resumen.append(f"Fecha: {extraido.get('fecha', '')}")
        resumen.append(f"Fecha ISO: {extraido.get('fecha_iso', '')}")

        # Guardar resumen en archivo de texto
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write('\n'.join(resumen))

        logger.info(f"Información de debug guardada en {ruta_salida}")

        # Crear imagen con resultados visuales
        alto, ancho = img_debug.shape[:2]

        # Añadir texto con resultados en la imagen
        y_offset = 30
        for linea in resumen:
            cv2.putText(img_debug, linea, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Guardar imagen con resultados
        cv2.imwrite("debug_resultado_visual.jpg", img_debug)
        logger.info("Imagen con resultados guardada como 'debug_resultado_visual.jpg'")

        return True
    except Exception as e:
        logger.error(f"Error al guardar información de debug: {str(e)}")
        return False


# Función para ejecutar la validación con imágenes de debug
def validar_con_debug(ruta_pdf, datos_input):
    """
    Ejecuta la validación y genera archivos de debug para diagnóstico.

    Args:
        ruta_pdf: Ruta al archivo PDF del certificado electoral
        datos_input: Diccionario con datos esperados para validación

    Returns:
        Resultado de la validación
    """
    # Configurar nivel de logging para más detalles durante el debug
    logger.setLevel(logging.DEBUG)

    # Ejecutar validación
    resultado, imagen = validar_certificado_electoral(ruta_pdf, datos_input)

    # Si hay error o no hay imagen, retornar el resultado
    if "error" in resultado or imagen is None:
        return resultado

    # Guardar información de debug
    guardar_debug(imagen, resultado)

    return resultado
