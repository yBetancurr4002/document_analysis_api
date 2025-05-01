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

# Cargar el modelo de TrOCR para reconocimiento de texto manuscrito
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_disponible = True
    logger.info("Modelo TrOCR cargado correctamente")
except Exception as e:
    logger.warning(f"No se pudo cargar el modelo TrOCR: {str(e)}")
    trocr_disponible = False

# Diccionario de meses en español para análisis de fechas
MESES_ES = {
    'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04', 'MAY': '05', 'JUN': '06',
    'JUL': '07', 'AGO': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
}


def reconocer_texto_manuscrito(imagen):
    """
    Utiliza el modelo TrOCR para reconocer texto manuscrito.
    """
    if not trocr_disponible:
        return ""

    try:
        # Convertir imagen CV2 a formato PIL
        if len(imagen.shape) == 3:  # Si es una imagen a color
            imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        else:  # Si es una imagen en escala de grises
            imagen_pil = Image.fromarray(imagen)

        # Preprocesar la imagen para el modelo
        pixel_values = processor(imagen_pil, return_tensors="pt").pixel_values

        # Generar predicción
        generated_ids = model.generate(pixel_values)
        texto_generado = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Texto manuscrito reconocido: {texto_generado}")
        return texto_generado
    except Exception as e:
        logger.error(f"Error al reconocer texto manuscrito: {str(e)}")
        return ""


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
    Aplica técnicas avanzadas de preprocesamiento para mejorar la extracción de texto,
    especialmente optimizado para certificados electorales.
    """
    try:
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Crear copias para diferentes técnicas de preprocesamiento
        resultados = []

        # 1. Umbralización adaptativa
        thresh1 = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        resultados.append(thresh1)

        # 2. Umbralización Otsu
        _, thresh2 = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resultados.append(thresh2)

        # 3. Mejora de contraste con ecualización de histograma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        mejorado = clahe.apply(gris)
        _, thresh3 = cv2.threshold(mejorado, 150, 255, cv2.THRESH_BINARY_INV)
        resultados.append(thresh3)

        # 4. Mejora para texto manuscrito - uso de técnicas de realce de bordes
        kernel_edge = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gris, -1, kernel_edge)
        _, thresh4 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resultados.append(thresh4)

        # Crear imagen combinada (O lógico de todas las técnicas)
        combinada = resultados[0].copy()
        for img in resultados[1:]:
            combinada = cv2.bitwise_or(combinada, img)

        # Operaciones morfológicas para limpiar ruido
        kernel = np.ones((2, 2), np.uint8)
        abierto = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel)

        # Dilatación para conectar componentes de caracteres manuscritos
        kernel_dil = np.ones((2, 2), np.uint8)
        dilatado = cv2.dilate(abierto, kernel_dil, iterations=1)

        return dilatado, gris, mejorado, sharpened
    except Exception as e:
        logger.error(f"Error en el preprocesamiento de imagen: {str(e)}")
        raise


def deskew_image(image):
    """
    Corrige la inclinación de la imagen para mejorar la extracción de texto.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        # Umbral binario para detectar el texto
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            logger.warning("No se encontraron puntos para calcular la inclinación")
            return image

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

        logger.info(f"Imagen corregida con ángulo: {angle}")
        return deskewed
    except Exception as e:
        logger.error(f"Error al corregir inclinación: {str(e)}")
        return image


def detectar_regiones_certificado_electoral(imagen, imagen_orig=None):
    """
    Detecta regiones específicas para certificados electorales.
    Utiliza plantilla predefinida basada en el formato estándar.
    """
    try:
        alto, ancho = imagen.shape[:2]

        # Regiones predefinidas basadas en certificado electoral estándar
        # Estas coordenadas están optimizadas para el formato mostrado en la imagen de ejemplo
        regiones = {
            # Región para número de cédula (justo después de "Cédula No.")
            "numero": (int(ancho * 0.45), int(alto * 0.3), int(ancho * 0.5), int(alto * 0.1)),

            # Región para nombres y apellidos (después de "Nombres y Apellidos")
            "nombres": (int(ancho * 0.45), int(alto * 0.4), int(ancho * 0.5), int(alto * 0.1)),

            # Región para fecha (dentro del título "CERTIFICADO ELECTORAL ELECCIONES...")
            "fecha": (int(ancho * 0.3), int(alto * 0.25), int(ancho * 0.6), int(alto * 0.08))
        }

        # Visualización para debugging
        if imagen_orig is not None:
            imagen_debug = imagen_orig.copy()
            for nombre, (x, y, w, h) in regiones.items():
                cv2.rectangle(imagen_debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(imagen_debug, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite("debug_regiones_certificado.jpg", imagen_debug)
            logger.info("Imagen de debug guardada como 'debug_regiones_certificado.jpg'")

        return regiones
    except Exception as e:
        logger.error(f"Error al detectar regiones: {str(e)}")
        # Regiones de respaldo
        alto, ancho = imagen.shape[:2]
        return {
            "numero": (int(ancho * 0.45), int(alto * 0.3), int(ancho * 0.5), int(alto * 0.1)),
            "nombres": (int(ancho * 0.45), int(alto * 0.4), int(ancho * 0.5), int(alto * 0.1)),
            "fecha": (int(ancho * 0.3), int(alto * 0.25), int(ancho * 0.6), int(alto * 0.1))
        }


def extraer_texto_combinado(imagen, region, imagenes_proc=None):
    """
    Combina OCR tradicional con reconocimiento de texto manuscrito.
    """
    try:
        x, y, w, h = region

        # Asegurar que las coordenadas estén dentro de los límites
        h_img, w_img = imagen.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        # Extraer región de interés
        roi = imagen[y:y + h, x:x + w]

        # Guardar ROI para debugging
        cv2.imwrite(f"debug_roi_{x}_{y}.jpg", roi)

        # 1. Intentar con reconocimiento de texto manuscrito si está disponible
        texto_manuscrito = reconocer_texto_manuscrito(roi)

        # 2. También intentar con OCR tradicional
        # Configuraciones OCR para diferentes escenarios
        configs = [
            '--oem 3 --psm 6',  # Configuración estándar
            '--oem 3 --psm 7',  # Para líneas de texto
            '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'  # Solo para dígitos
        ]

        resultados_ocr = []

        # Procesar todas las imágenes disponibles
        imagenes_a_probar = [roi]
        if imagenes_proc:
            for img_proc in imagenes_proc:
                if img_proc is not None:
                    roi_proc = img_proc[y:y + h, x:x + w]
                    imagenes_a_probar.append(roi_proc)
                    cv2.imwrite(f"debug_roi_proc_{x}_{y}_{len(imagenes_a_probar)}.jpg", roi_proc)

        # Probar OCR en todas las imágenes
        for img in imagenes_a_probar:
            for config in configs:
                try:
                    texto = pytesseract.image_to_string(img, lang='spa', config=config).strip()
                    if texto:
                        resultados_ocr.append(texto)
                except Exception as e:
                    logger.debug(f"Error en OCR: {str(e)}")

        # Elegir el mejor resultado
        mejor_resultado_ocr = ""
        if resultados_ocr:
            mejor_resultado_ocr = max(resultados_ocr, key=len)

        # Combinar resultados dando prioridad al reconocimiento manuscrito si existe
        resultado_final = texto_manuscrito if texto_manuscrito else mejor_resultado_ocr

        logger.info(f"Texto extraído (combinado): {resultado_final}")
        return resultado_final

    except Exception as e:
        logger.error(f"Error al extraer texto combinado: {str(e)}")
        return ""


def extraer_numero_documento_mejorado(texto):
    """
    Extrae número de documento con manejo mejorado para formato colombiano.
    """
    try:
        # Eliminar texto no deseado
        texto = re.sub(r'No\.?|C\.?C\.?|Cédula|CÉDULA|CEDULA', '', texto, flags=re.IGNORECASE)

        # Limpiar texto (reemplazar caracteres confundidos)
        texto = texto.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').strip()

        # Intentar extraer patrones típicos de cédulas colombianas
        patrones = [
            r'\d{1,3}[\,\.]\d{3}[\,\.]\d{3}',  # formato con separadores (1.234.567)
            r'\d{7,11}',  # dígitos consecutivos (formato sin separadores)
        ]

        for patron in patrones:
            match = re.search(patron, texto)
            if match:
                # Limpiar separadores
                numero = re.sub(r'[^\d]', '', match.group(0))
                if len(numero) >= 7:  # Las cédulas colombianas tienen al menos 7 dígitos
                    logger.info(f"Número de documento encontrado: {numero}")
                    return numero

        # Si llegamos aquí, intentar extraer cualquier secuencia de números
        numeros = re.findall(r'\d+', texto)
        if numeros:
            # Unir grupos de números si están separados
            if len(numeros) > 1:
                numero_combinado = ''.join(numeros)
                if len(numero_combinado) >= 7:
                    return numero_combinado

            # O tomar el número más largo
            mejor_numero = max(numeros, key=len)
            if len(mejor_numero) >= 7:
                return mejor_numero

        logger.warning(f"No se pudo extraer número de documento de: '{texto}'")
        return texto if len(re.sub(r'[^\d]', '', texto)) >= 7 else ''

    except Exception as e:
        logger.error(f"Error al extraer número de documento: {str(e)}")
        return ''


def extraer_nombre_mejorado(texto):
    """
    Extrae y normaliza nombre con manejo mejorado para manuscritos.
    """
    try:
        # Eliminar texto que no forma parte del nombre
        texto = re.sub(r'nombres|apellidos|nombre|apellido', '', texto, flags=re.IGNORECASE)

        # Eliminar caracteres no alfabéticos excepto espacios
        limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]', '', texto)

        # Eliminar espacios extra
        limpio = re.sub(r'\s+', ' ', limpio).strip()

        # Convertir a formato de nombre (primera letra de cada palabra en mayúscula)
        if limpio:
            nombre_normalizado = ' '.join(p.capitalize() for p in limpio.split())
            logger.info(f"Nombre extraído: '{nombre_normalizado}'")
            return nombre_normalizado

        logger.warning(f"No se pudo extraer nombre de: '{texto}'")
        return texto.strip()

    except Exception as e:
        logger.error(f"Error al extraer nombre: {str(e)}")
        return texto.strip()


def extraer_fecha_electoral(texto):
    """
    Extrae fecha específicamente de certificados electorales colombianos.
    Optimizado para formatos como "29 DE MAYO DE 2022"
    """
    try:
        # Para certificados electorales, generalmente la fecha está en el título
        # Buscar patrón específico para elecciones
        match_elecciones = re.search(r'ELECCIONES\s+(\d{1,2})\s+DE\s+([A-ZÁ-Ú]{3,9})\s+DE\s+(\d{4})', texto.upper())
        if match_elecciones:
            dia, mes, año = match_elecciones.groups()
            logger.info(f"Fecha electoral encontrada: {dia} DE {mes} DE {año}")
            return f"{dia} DE {mes} DE {año}"

        # Patrones alternativos
        patrones = [
            r'(\d{1,2})\s+DE\s+([A-ZÁ-Ú]{3,9})\s+DE\s+(\d{4})',  # 29 DE MAYO DE 2022
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # 29/05/2022
            r'(\d{1,2})-(\d{1,2})-(\d{4})'  # 29-05-2022
        ]

        for patron in patrones:
            match = re.search(patron, texto.upper())
            if match:
                grupos = match.groups()
                if len(grupos) == 3:
                    if '/' in patron or '-' in patron:
                        return f"{grupos[0]}/{grupos[1]}/{grupos[2]}"
                    else:
                        return f"{grupos[0]} DE {grupos[1]} DE {grupos[2]}"

        logger.warning(f"No se pudo extraer fecha de: '{texto}'")
        return ''

    except Exception as e:
        logger.error(f"Error al extraer fecha: {str(e)}")
        return ''


def validar_certificado_electoral_mejorado(ruta_pdf, datos_input):
    """
    Valida un certificado electoral con algoritmos mejorados.
    """
    try:
        logger.info(f"Validando certificado electoral: {ruta_pdf}")
        logger.info(f"Datos de entrada para validación: {datos_input}")

        # Convertir PDF a imagen
        imagen = pdf_a_imagen(ruta_pdf)
        if imagen is None:
            return {"error": "No se pudo cargar la imagen del PDF"}, None

        # Corregir inclinación
        imagen = deskew_image(imagen)

        # Guardar imagen original para referencia
        cv2.imwrite("debug_original.jpg", imagen)

        # Aplicar preprocesamiento avanzado
        preprocesada, gris, mejorada, realzada = preprocesar_imagen(imagen)

        # Guardar imágenes procesadas para debugging
        cv2.imwrite("debug_preprocesada.jpg", preprocesada)
        cv2.imwrite("debug_gris.jpg", gris)
        cv2.imwrite("debug_mejorada.jpg", mejorada)
        cv2.imwrite("debug_realzada.jpg", realzada)

        # Detectar regiones específicas para certificado electoral
        regiones = detectar_regiones_certificado_electoral(gris, imagen)

        # Extraer texto de cada región usando métodos combinados
        extraido = {}
        for campo, region in regiones.items():
            texto = extraer_texto_combinado(gris, region, [preprocesada, mejorada, realzada])
            extraido[campo] = texto
            logger.info(f"Campo '{campo}' extraído: '{texto}'")

        # Procesar texto extraído con funciones mejoradas
        if 'numero' in extraido:
            extraido["numero"] = extraer_numero_documento_mejorado(extraido["numero"])

        if 'nombres' in extraido:
            extraido["nombres"] = extraer_nombre_mejorado(extraido["nombres"])

        if 'fecha' in extraido:
            extraido["fecha"] = extraer_fecha_electoral(extraido["fecha"])

        # Validación mejorada contra datos de entrada
        # Limpiar datos de entrada para comparación
        numero_input = re.sub(r'\D', '', datos_input.get("numero", ""))
        nombres_input = datos_input.get("nombres", "").lower().strip()
        fecha_input = datos_input.get("fecha", "").lower().strip()

        # Comparación de número de documento con mayor flexibilidad
        numero_extraido = extraido.get("numero", "")
        numero_encontrado = False

        if numero_extraido and numero_input:
            # Para cédulas, a veces hay diferencias en separadores o longitud
            # Eliminar todos los no-dígitos para comparación
            num_ext_limpio = re.sub(r'\D', '', numero_extraido)
            num_inp_limpio = re.sub(r'\D', '', numero_input)

            # Verificar coincidencia en los últimos dígitos (más común en validación)
            longitud_min = min(len(num_ext_limpio), len(num_inp_limpio))
            if longitud_min >= 5:  # Al menos verificar 5 dígitos
                # Comparar los últimos dígitos (más confiables)
                ultimos_digitos = min(longitud_min, 7)  # Usar hasta 7 últimos dígitos
                if num_ext_limpio[-ultimos_digitos:] == num_inp_limpio[-ultimos_digitos:]:
                    numero_encontrado = True
                # O verificar si uno contiene al otro
                elif num_ext_limpio in num_inp_limpio or num_inp_limpio in num_ext_limpio:
                    numero_encontrado = True

        # Comparación de nombres con mayor tolerancia (para manuscritos)
        nombres_extraidos = extraido.get("nombres", "").lower()
        nombres_encontrado = False

        if nombres_extraidos and nombres_input:
            # Dividir en palabras individuales (nombres/apellidos)
            palabras_input = [p for p in nombres_input.lower().split() if len(p) > 2]
            palabras_extraidas = [p for p in nombres_extraidos.lower().split() if len(p) > 2]

            # Calcular similitud basada en palabras coincidentes
            coincidencias = 0
            for p_in in palabras_input:
                for p_ex in palabras_extraidas:
                    # Considerar coincidencia si hay suficiente similitud
                    if p_in in p_ex or p_ex in p_in or (
                            len(p_in) > 3 and len(p_ex) > 3 and
                            (p_in[:3] == p_ex[:3] or p_in[-3:] == p_ex[-3:])
                    ):
                        coincidencias += 1
                        break

            # Considerar válido si hay suficientes coincidencias
            umbral_coincidencia = max(1, len(palabras_input) // 2)  # Al menos la mitad o una palabra
            nombres_encontrado = coincidencias >= umbral_coincidencia

        # Comparación de fecha
        fecha_extraida = extraido.get("fecha", "").lower()
        fecha_encontrada = False

        if fecha_extraida and fecha_input:
            # Normalizar fechas - extraer números
            fecha_ext_nums = re.findall(r'\d+', fecha_extraida)
            fecha_inp_nums = re.findall(r'\d+', fecha_input)

            if len(fecha_ext_nums) >= 2 and len(fecha_inp_nums) >= 2:
                # Verificar si coinciden día y año (los más distintivos)
                coincidencias_fecha = sum(1 for n in fecha_ext_nums if n in fecha_inp_nums)
                fecha_encontrada = coincidencias_fecha >= 2

            # También verificar si hay coincidencia en texto de mes
            if not fecha_encontrada and "mayo" in fecha_input.lower() and "mayo" in fecha_extraida.lower():
                fecha_encontrada = True

        # Resultado final
        resultado = {
            "numero_encontrado": numero_encontrado,
            "nombres_encontrado": nombres_encontrado,
            "fecha_encontrada": fecha_encontrada,
            "es_valido": numero_encontrado and nombres_encontrado,  # Ser más flexible con la fecha
            "texto_extraido": extraido,
            "confianza": {
                "numero": "alto" if numero_encontrado else "bajo",
                "nombres": "alto" if nombres_encontrado else "bajo",
                "fecha": "alto" if fecha_encontrada else "bajo"
            }
        }

        logger.info(f"Resultados de validación: {resultado}")
        return resultado, imagen

    except Exception as e:
        logger.error(f"Error al validar certificado: {str(e)}")
        return {"error": str(e)}, None


def guardar_debug_mejorado(imagen, resultado, ruta_salida="debug_resultado.txt"):
    """
    Guarda información detallada de debugging y visualización del resultado.
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

        if "confianza" in resultado:
            resumen.append(f"--------------------")
            resumen.append(f"NIVELES DE CONFIANZA:")
            for campo, nivel in resultado["confianza"].items():
                resumen.append(f"{campo}: {nivel}")

        # Guardar resumen en archivo de texto
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write('\n'.join(resumen))

        logger.info(f"Información de debug guardada en {ruta_salida}")

        # Crear imagen con resultados visuales
        alto, ancho = img_debug.shape[:2]

        # Añadir texto con resultados en la imagen
        y_offset = 30
        for linea in resumen[:8]:  # Mostrar solo las líneas principales
            cv2.putText(img_debug, linea, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Guardar imagen con resultados
        cv2.imwrite("debug_resultado_visual.jpg", img_debug)
        logger.info("Imagen de resultados guardada como 'debug_resultado_visual.jpg'")

    except Exception as e:
        logger.error(f"Error al guardar debug: {str(e)}")


def generar_reporte_excel(resultados, ruta_salida="resultados_validacion.xlsx"):
    """
    Genera un reporte Excel con los resultados de validación para múltiples certificados.
    """
    try:
        # Crear DataFrame con resultados
        datos = []
        for archivo, resultado in resultados.items():
            extraido = resultado.get("texto_extraido", {})
            datos.append({
                "Archivo": os.path.basename(archivo),
                "Número Documento": extraido.get("numero", ""),
                "Nombres y Apellidos": extraido.get("nombres", ""),
                "Fecha Electoral": extraido.get("fecha", ""),
                "Número Válido": "Sí" if resultado.get("numero_encontrado", False) else "No",
                "Nombres Válidos": "Sí" if resultado.get("nombres_encontrado", False) else "No",
                "Fecha Válida": "Sí" if resultado.get("fecha_encontrada", False) else "No",
                "Validación Completa": "Sí" if resultado.get("es_valido", False) else "No",
                "Confianza Número": resultado.get("confianza", {}).get("numero", ""),
                "Confianza Nombres": resultado.get("confianza", {}).get("nombres", ""),
                "Confianza Fecha": resultado.get("confianza", {}).get("fecha", "")
            })

        # Crear DataFrame y ajustar anchos de columna
        df = pd.DataFrame(datos)

        # Guardar a Excel
        with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Resultados")
            hoja = writer.sheets["Resultados"]

            # Ajustar anchos de columna
            for columna in hoja.columns:
                max_length = 0
                columna_letra = columna[0].column_letter
                for celda in columna:
                    try:
                        if len(str(celda.value)) > max_length:
                            max_length = len(str(celda.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                hoja.column_dimensions[columna_letra].width = adjusted_width

        logger.info(f"Reporte Excel guardado en {ruta_salida}")
        return True
    except Exception as e:
        logger.error(f"Error al generar reporte Excel: {str(e)}")
        return False


def procesar_directorio(directorio, datos_validacion, guardar_debug=True):
    """
    Procesa todos los PDFs en un directorio para validación masiva.
    """
    try:
        resultados = {}
        archivos_pdf = [f for f in os.listdir(directorio) if f.lower().endswith('.pdf')]

        if not archivos_pdf:
            logger.warning(f"No se encontraron archivos PDF en {directorio}")
            return resultados

        logger.info(f"Procesando {len(archivos_pdf)} archivos PDF encontrados")

        for archivo in archivos_pdf:
            ruta_completa = os.path.join(directorio, archivo)
            logger.info(f"Procesando archivo: {archivo}")

            try:
                resultado, imagen = validar_certificado_electoral_mejorado(ruta_completa, datos_validacion)
                resultados[ruta_completa] = resultado

                if guardar_debug and imagen is not None:
                    # Crear directorio de resultados si no existe
                    dir_resultados = os.path.join(directorio, "resultados")
                    os.makedirs(dir_resultados, exist_ok=True)

                    # Nombre base para archivos de debug
                    nombre_base = os.path.splitext(archivo)[0]
                    ruta_debug = os.path.join(dir_resultados, f"{nombre_base}_debug.txt")
                    guardar_debug_mejorado(imagen, resultado, ruta_debug)
            except Exception as e:
                logger.error(f"Error al procesar {archivo}: {str(e)}")
                resultados[ruta_completa] = {"error": str(e)}

        # Generar reporte Excel consolidado
        if resultados:
            ruta_excel = os.path.join(directorio, "reporte_validacion.xlsx")
            generar_reporte_excel(resultados, ruta_excel)

        return resultados
    except Exception as e:
        logger.error(f"Error al procesar directorio {directorio}: {str(e)}")
        return {"error": str(e)}


def main():
    """
    Función principal que permite procesar un directorio de certificados electorales.
    """
    try:
        # Configuración de logging
        directorio_logs = "logs"
        os.makedirs(directorio_logs, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(directorio_logs, f"validacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        print("Sistema de Validación de Certificados Electorales")
        print("================================================")

        # Solicitar datos para validación
        print("\nIngrese los datos para validación:")
        numero = input("Número de documento: ")
        nombres = input("Nombres y apellidos: ")
        fecha = input("Fecha electoral (ej: 29 DE MAYO DE 2022): ")

        datos_validacion = {
            "numero": numero,
            "nombres": nombres,
            "fecha": fecha
        }

        # Solicitar directorio con certificados
        directorio = input("\nDirectorio con certificados PDF: ")
        if not os.path.isdir(directorio):
            print(f"Error: El directorio {directorio} no existe.")
            return

        print("\nProcesando certificados, por favor espere...")
        resultados = procesar_directorio(directorio, datos_validacion)

        # Mostrar resumen
        print("\nResumen de resultados:")
        total = len(resultados)
        validos = sum(1 for r in resultados.values() if r.get("es_valido", False))
        con_errores = sum(1 for r in resultados.values() if "error" in r)

        print(f"Total procesados: {total}")
        print(f"Certificados válidos: {validos}")
        print(f"Certificados inválidos: {total - validos - con_errores}")
        print(f"Errores de procesamiento: {con_errores}")

        print(f"\nSe ha generado un reporte Excel en {os.path.join(directorio, 'reporte_validacion.xlsx')}")
        print("Las imágenes y archivos de debug se encuentran en la carpeta 'resultados'")

    except Exception as e:
        logger.error(f"Error en función principal: {str(e)}")
        print(f"Ha ocurrido un error: {str(e)}")


if __name__ == "__main__":
    main()