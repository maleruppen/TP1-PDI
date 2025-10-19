import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Definición de función que se usará para analizar cada celda. 
def analizar_celda(celda_img, th_min_area=30, th_max_area=3000, space_threshold=6) -> tuple:
        """
        Analiza una imagen de una celda para detectar caracteres y estimar la cantidad de palabras.

        Parámetros: \n
        celda_img : np.ndarray
            Imagen en escala de grises de la celda a analizar. \n
        th_min_area : int, opcional
            Área mínima de un componente conectado para considerarlo un carácter válido (default: 30). \n
        th_max_area : int, opcional
            Área máxima de un componente conectado para considerarlo un carácter válido (default: 3000). \n
        space_threshold : int, opcional
            Distancia en píxeles entre componentes que se considera como separación entre palabras (default: 6). \n
        Retorna la tupla (caracteres_validos,cantidad_palabras)
        """

        padding = 3
        h, w = celda_img.shape
        if h <= 2 * padding or w <= 2 * padding:
            return 0, 0
        celda_recortada = celda_img[padding:h - padding, padding:w - padding]
        _, binary = cv2.threshold(celda_recortada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S) # Conectividad de 8 vecinos para revisar píxeles diagonales. 
        ix_area = (stats[:, -1] > th_min_area) & (stats[:, -1] < th_max_area) # Se filtran filas con componentes con áreas demasiado grandes o demasiado pequeñas. 
        stats = stats[ix_area]
        caracteres_validos = num_labels - 1 # Cantidad de componentes detectadas menos el fondo.
        if caracteres_validos == 0:
            return 0, 0
        sorted_stats = stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]
        palabras = 1
        # Cálculo de distancias usando boundig box de las componentes. 
        for i in range(caracteres_validos - 1):
            x_i = sorted_stats[i, cv2.CC_STAT_LEFT]
            w_i = sorted_stats[i, cv2.CC_STAT_WIDTH]
            x_j = sorted_stats[i + 1, cv2.CC_STAT_LEFT]
            distancia = x_j - (x_i + w_i)
            if distancia > space_threshold:
                palabras += 1
        return caracteres_validos, palabras

# Función utilizada para validar los formularios según las reglas determinadas.
def validar_formulario(chars_nombre, words_nombre,
                           chars_edad, words_edad,
                           chars_mail, words_mail,
                           chars_legajo, words_legajo,
                           chars_com, words_com,
                           chars_p1s, chars_p1n,
                           chars_p2s, chars_p2n,
                           chars_p3s, chars_p3n) -> dict:
        """Realiza las verificaciones de formato de cada campo 
        basado en la cantidad de caracteres y palabras contenidos en cada uno."""
        val_nom = (words_nombre >= 2) and (chars_nombre <= 25)
        val_edad = (chars_edad > 1 and chars_edad <= 3) and (words_edad == 1)
        val_mail = (words_mail == 1) and (chars_mail <= 25)
        val_legajo = (chars_legajo == 8) and (words_legajo == 1)
        val_com = (words_com >= 1) and (chars_com <= 25)
        val_p1 = (chars_p1s == 1) ^ (chars_p1n == 1) # Se usa el operador XOR que devuelve True si solo una de las condiciones es verdadera.
        val_p2 = (chars_p2s == 1) ^ (chars_p2n == 1)
        val_p3 = (chars_p3s == 1) ^ (chars_p3n == 1)
        form_completo = (val_nom and val_edad and val_mail and val_legajo and val_com and val_p1 and val_p2 and val_p3)
        return {
            "Nombre y apellido": val_nom,
            "Edad": val_edad,
            "Mail": val_mail,
            "Legajo": val_legajo,
            "Pregunta 1": val_p1,
            "Pregunta 2": val_p2,
            "Pregunta 3": val_p3,
            "Comentarios": val_com,
            "FORMULARIO_COMPLETO_OK": form_completo
        }


# Se enumeran todos los formularios a procesar. 
formularios = [
    'formulario_01.png',
    'formulario_02.png',
    'formulario_03.png',
    'formulario_04.png',
    'formulario_05.png'
]


resultados_globales = []
imagenes_nombre = []   # para guardar crops del campo "Nombre y Apellido"
estados_nombre = []    # para guardar si fue correcto o no 

for formulario in formularios:
    print(f"\n============================")
    print(f"Procesamiento de {formulario}")
    print(f"============================")

    # Detección de filas usando binarización y umbralado.
    img = cv2.imread(formulario, cv2.IMREAD_GRAYSCALE)
    img_zeros = img < 120
    img_row_zeros = img_zeros.sum(axis=1)
    th_row = 0.6 * np.max(img_row_zeros)
    rows_detect = img_row_zeros > th_row
    row_changes = np.where(np.diff(rows_detect.astype(int)) != 0)[0]

    horizontal_lines = []
    for i in range(0, len(row_changes) - 1, 2):
        center = (row_changes[i] + row_changes[i + 1]) // 2
        horizontal_lines.append(center)

    # Detección de columnas usando binarización y umbralado.
    img_col_zeros = img_zeros.sum(axis=0)
    th_col = 0.35 * np.max(img_col_zeros)
    cols_detect = img_col_zeros > th_col
    col_changes = np.where(np.diff(cols_detect.astype(int)) != 0)[0]

    vertical_lines = []
    for i in range(0, len(col_changes) - 1, 2):
        center = (col_changes[i] + col_changes[i + 1]) // 2
        vertical_lines.append(center)

    try:
        # Procesamiento de campos principales.
        y1_nom, y2_nom = horizontal_lines[1], horizontal_lines[2]
        x1_nom, x2_nom = vertical_lines[1], vertical_lines[3]
        celda_nombre = img[y1_nom:y2_nom, x1_nom:x2_nom]
        chars_nombre, words_nombre = analizar_celda(celda_nombre, th_min_area=20)

        y1_edad, y2_edad = horizontal_lines[2], horizontal_lines[3]
        x1_edad, x2_edad = vertical_lines[1], vertical_lines[3]
        celda_edad = img[y1_edad:y2_edad, x1_edad:x2_edad]
        chars_edad, words_edad = analizar_celda(celda_edad, th_min_area=30)

        y1_mail, y2_mail = horizontal_lines[3], horizontal_lines[4]
        celda_mail = img[y1_mail:y2_mail, x1_nom:x2_nom]
        chars_mail, words_mail = analizar_celda(celda_mail, th_min_area=5)

        y1_leg, y2_leg = horizontal_lines[4], horizontal_lines[5]
        celda_legajo = img[y1_leg:y2_leg, x1_nom:x2_nom]
        chars_legajo, words_legajo = analizar_celda(celda_legajo, th_min_area=10)

        y1_com, y2_com = horizontal_lines[-2], horizontal_lines[-1]
        celda_com = img[y1_com:y2_com, x1_nom:x2_nom]
        chars_com, words_com = analizar_celda(celda_com, th_min_area=10)

        # Procesamiento de preguntas
        y1_p1s, y2_p1s = horizontal_lines[6], horizontal_lines[7]
        y1_p2s, y2_p2s = horizontal_lines[7], horizontal_lines[8]
        y1_p3s, y2_p3s = horizontal_lines[8], horizontal_lines[9]
        x1s, x2s = vertical_lines[1], vertical_lines[2]
        x1n, x2n = vertical_lines[2], vertical_lines[3]

        celda_p1s = img[y1_p1s:y2_p1s, x1s:x2s]
        celda_p1n = img[y1_p1s:y2_p1s, x1n:x2n]
        celda_p2s = img[y1_p2s:y2_p2s, x1s:x2s]
        celda_p2n = img[y1_p2s:y2_p2s, x1n:x2n]
        celda_p3s = img[y1_p3s:y2_p3s, x1s:x2s]
        celda_p3n = img[y1_p3s:y2_p3s, x1n:x2n]

        chars_p1s, _ = analizar_celda(celda_p1s, th_min_area=10)
        chars_p1n, _ = analizar_celda(celda_p1n, th_min_area=10)
        chars_p2s, _ = analizar_celda(celda_p2s, th_min_area=10)
        chars_p2n, _ = analizar_celda(celda_p2n, th_min_area=10)
        chars_p3s, _ = analizar_celda(celda_p3s, th_min_area=10)
        chars_p3n, _ = analizar_celda(celda_p3n, th_min_area=10)

    except IndexError:
        print(f"Error procesando {formulario}: no se detectaron suficientes líneas.")
        continue

    
    
    resultados = validar_formulario(
        chars_nombre, words_nombre,
        chars_edad, words_edad,
        chars_mail, words_mail,
        chars_legajo, words_legajo,
        chars_com, words_com,
        chars_p1s, chars_p1n,
        chars_p2s, chars_p2n,
        chars_p3s, chars_p3n
    )

    resultados["Archivo"] = formulario
    resultados_globales.append(resultados)
    imagenes_nombre.append(celda_nombre)
    estados_nombre.append(resultados["FORMULARIO_COMPLETO_OK"])

    # Mostrar por pantalla
    print("\n--- RESULTADOS DE LA VALIDACIÓN ---")
    for campo, estado in resultados.items():
        if campo != "Archivo":
            estado_str = "OK" if estado else "MAL"
            print(f"> {campo}: {estado_str}")

# Generación de imagen final con los resultados del procesamiento. 
plt.figure(figsize=(15, 8))
plt.suptitle("Informe General - Formularios Correctos / Incorrectos", fontsize=18)

for i, (img_nom, estado, res) in enumerate(zip(imagenes_nombre, estados_nombre, resultados_globales)):
    plt.subplot(1, len(formularios), i + 1)
    plt.imshow(img_nom, cmap='gray')
    color = 'green' if estado else 'red'
    plt.title(f"{res['Archivo']}\n{'OK' if estado else 'MAL'}", color=color)
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Creación de csv.
with open("resultados_formularios.csv", "w", newline="", encoding="utf-8") as csvfile:
    # Definimos las columnas en el orden requerido
    fieldnames = [
        "ID",
        "Nombre y Apellido",
        "Edad",
        "Mail",
        "Legajo",
        "Pregunta 1",
        "Pregunta 2",
        "Pregunta 3",
        "Comentarios"
    ]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for res in resultados_globales:
        # Extraer ID desde el nombre del archivo (ej: formulario_03.png -> 03)
        id_form = res["Archivo"].split("_")[1].split(".")[0]

        # Convertir True/False a OK/MAL según el formato requerido
        fila = {
            "ID": id_form,
            "Nombre y Apellido": "OK" if res["Nombre y apellido"] else "MAL",
            "Edad": "OK" if res["Edad"] else "MAL",
            "Mail": "OK" if res["Mail"] else "MAL",
            "Legajo": "OK" if res["Legajo"] else "MAL",
            "Pregunta 1": "OK" if res["Pregunta 1"] else "MAL",
            "Pregunta 2": "OK" if res["Pregunta 2"] else "MAL",
            "Pregunta 3": "OK" if res["Pregunta 3"] else "MAL",
            "Comentarios": "OK" if res["Comentarios"] else "MAL"
        }

        writer.writerow(fila)






















