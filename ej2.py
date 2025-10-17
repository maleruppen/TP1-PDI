import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---  Detección de Filas  ---
img = cv2.imread('formulario_02.png', cv2.IMREAD_GRAYSCALE) 
img_zeros = img < 120
img_row_zeros = img_zeros.sum(axis=1)

th_row = 0.6 * np.max(img_row_zeros)
rows_detect = img_row_zeros > th_row
row_changes = np.where(np.diff(rows_detect.astype(int)) != 0)[0]

horizontal_lines = []
for i in range(0, len(row_changes) - 1, 2):
    center = (row_changes[i] + row_changes[i+1]) // 2
    horizontal_lines.append(center)


# ===================================================================
# ---  Detección de Columnas -----
# ===================================================================

# Usamos la misma técnica de proyección, pero en el eje 0 (columnas)
img_col_zeros = img_zeros.sum(axis=0)

# Usamos un umbral. 0.6 es un buen punto de partida.
th_col = 0.35 * np.max(img_col_zeros) 
cols_detect = img_col_zeros > th_col
col_changes = np.where(np.diff(cols_detect.astype(int)) != 0)[0]

vertical_lines = []
for i in range(0, len(col_changes) - 1, 2):
    center = (col_changes[i] + col_changes[i+1]) // 2
    vertical_lines.append(center)

print(f"Líneas Horizontales detectadas (Y): {horizontal_lines}")
print(f"Líneas Verticales detectadas (X): {vertical_lines}")


# ===================================================================
# --- Visualización de Columnas Recortadas ---
# ===================================================================

plt.figure(figsize=(15, 8))
plt.suptitle("Columnas Detectadas y Recortadas", fontsize=20)

# Iterar a través de las líneas verticales
# Asumiendo que detecta 4 líneas, resultando en 3 columnas
num_columnas = len(vertical_lines) - 1
for i in range(num_columnas):
    # Definir las coordenadas X de inicio y fin de la columna
    x1 = vertical_lines[i]
    x2 = vertical_lines[i+1]
    
    # Recortar la columna de la imagen original
    # Usamos [:, x1:x2] para tomar todas las filas
    columna_recortada = img[:, x1:x2]
    
    # Crear un subplot para cada columna
    plt.subplot(1, num_columnas, i + 1)
    plt.imshow(columna_recortada, cmap='gray')
    plt.title(f"Columna {i+1} (X: {x1} a {x2})")
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ===================================================================
# --- Visualización de Filas  ---
# ===================================================================
plt.figure(figsize=(15, 8))
plt.suptitle("Filas Detectadas y Recortadas", fontsize=20)
for i in range(len(horizontal_lines) - 1):
    y1 = horizontal_lines[i]
    y2 = horizontal_lines[i+1]
    fila_recortada = img[y1:y2, :]
    plt.subplot(2, 5, i + 1)
    plt.imshow(fila_recortada, cmap='gray')
    plt.title(f"Fila {i+1} (Y: {y1} a {y2})")
    plt.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




# --- 3. Función de Análisis  ---
def analizar_celda(celda_img, th_min_area=30, th_max_area=3000, space_threshold=6):
    """
    Analiza una celda.
    - Filtra por área MÍNIMA (ruido) y MÁXIMA (manchas/líneas).
    """
    # quita un poco de los bordes
    padding = 3
    h, w = celda_img.shape
    if h <= 2*padding or w <= 2*padding: return 0
    celda_recortada = celda_img[padding:h-padding, padding:w-padding]

    #umbralado
    _, binary = cv2.threshold(celda_recortada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    
    # Filtro de área (# --- Filtrado vectorizado por área
    ix_area = (stats[:, -1] > th_min_area) & (stats[:, -1] < th_max_area)
    stats = stats[ix_area, :]      # Filtramos solo las componentes válidas
    caracteres_validos = len(stats -1) # resto 1 para ignorar el fondo (label 0)


    # Si no hay caracteres válidos, retornamos 0 caracteres y 0 palabras.
    if caracteres_validos == 0:
        return 0, 0
    
    # --- Lógica de Detección de Palabras basada en Distancia ---

    # Ordenar las componentes de izquierda a derecha (basado en pos_x, índice 0)
    sorted_stats = stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]
    
    # Calcular palabras
    # Si hay caracteres válidos, hay al menos una palabra.
    palabras = 1 
    
    # Iteramos hasta el penúltimo carácter para comparar pares
    for i in range(caracteres_validos - 1):
        # Posición X del borde izquierdo de la letra actual
        x_i = sorted_stats[i, cv2.CC_STAT_LEFT]
        #Ancho de la letra actual
        w_i = sorted_stats[i, cv2.CC_STAT_WIDTH] 
        
        # Posición X del borde izquierdo de la letra siguiente
        x_j = sorted_stats[i+1, cv2.CC_STAT_LEFT]
        
        # Distancia horizontal: Inicio de j - (Inicio de i + Ancho de i)
        distancia = x_j - (x_i + w_i)

        # Si la distancia supera el umbral, contamos una nueva palabra
        if distancia > space_threshold:
            palabras += 1
            
    # 7. Retornar caracteres y palabras
    return caracteres_validos, palabras

# ------
try:
    print("\n--- PASO A PASO: Analizando Campos de Texto ---")
    
    # --- "Nombre y Apellido" (Fila 2, Campo Ancho) ---
    y1_nom, y2_nom = horizontal_lines[1], horizontal_lines[2]
    x1_nom_1, x2_nom_1 = vertical_lines[1], vertical_lines[3]
    celda_nombre_1 = img[y1_nom:y2_nom, x1_nom_1:x2_nom_1]
    chars_nombre = analizar_celda(celda_nombre_1, th_min_area=20) 
    
    # --- "Edad" (Fila 3, Campo Angosto) ---
    y1_edad, y2_edad = horizontal_lines[2], horizontal_lines[3]
    x1_edad, x2_edad = vertical_lines[1], vertical_lines[3]
    celda_edad = img[y1_edad:y2_edad, x1_edad:x2_edad]
    chars_edad = analizar_celda(celda_edad, th_min_area=30)

    # --- "Mail" (Fila 4, Campo Ancho) ---
    y1_mail, y2_mail = horizontal_lines[3], horizontal_lines[4]
    x1_mail_1, x2_mail_1 = vertical_lines[1], vertical_lines[3]
    celda_mail_1 = img[y1_mail:y2_mail, x1_mail_1:x2_mail_1]
    chars_mail = analizar_celda(celda_mail_1, th_min_area=5) 
    
    # --- "Legajo" (Fila 5, Campo Angosto) ---
    y1_leg, y2_leg = horizontal_lines[4], horizontal_lines[5]
    x1_leg, x2_leg = vertical_lines[1], vertical_lines[3]
    celda_legajo = img[y1_leg:y2_leg, x1_leg:x2_leg]
    chars_legajo = analizar_celda(celda_legajo, th_min_area=10)

    # --- "comentarios"  ---
    y1_com, y2_com = horizontal_lines[-2], horizontal_lines[-1]
    x1_com, x2_com = vertical_lines[1], vertical_lines[3]
    celda_com = img[y1_com:y2_com, x1_com:x2_com]
    chars_com = analizar_celda(celda_com, th_min_area=10)

    # --- "p1-si"  ---
    y1_p1s, y2_p1s = horizontal_lines[6], horizontal_lines[7]
    x1_p1s, x2_p1s = vertical_lines[1], vertical_lines[2]
    celda_p1s = img[y1_p1s:y2_p1s, x1_p1s:x2_p1s]
    chars_p1s = analizar_celda(celda_p1s, th_min_area=10)

    # --- "p1-no"  ---
    y1_p1n, y2_p1n = horizontal_lines[6], horizontal_lines[7]
    x1_p1n, x2_p1n = vertical_lines[2], vertical_lines[3]
    celda_p1n = img[y1_p1n:y2_p1n, x1_p1n:x2_p1n]
    chars_p1n = analizar_celda(celda_p1n, th_min_area=10)

    # --- "p2-si"  ---
    y1_p2s, y2_p2s = horizontal_lines[7], horizontal_lines[8]
    x1_p2s, x2_p2s = vertical_lines[1], vertical_lines[2]
    celda_p2s = img[y1_p2s:y2_p2s, x1_p2s:x2_p2s]
    chars_p2s = analizar_celda(celda_p2s, th_min_area=10)

    # --- "p2-no"  ---
    y1_p2n, y2_p2n = horizontal_lines[7], horizontal_lines[8]
    x1_p2n, x2_p2n = vertical_lines[2], vertical_lines[3]
    celda_p2n = img[y1_p2n:y2_p2n, x1_p2n:x2_p2n]
    chars_p2n = analizar_celda(celda_p2n, th_min_area=10)

    # --- "p3-si"  ---
    y1_p3s, y2_p3s = horizontal_lines[8], horizontal_lines[9]
    x1_p3s, x2_p3s = vertical_lines[1], vertical_lines[2]
    celda_p3s = img[y1_p3s:y2_p3s, x1_p3s:x2_p3s]
    chars_p3s = analizar_celda(celda_p3s, th_min_area=10)

    # --- "p1-no"  ---
    y1_p3n, y2_p3n = horizontal_lines[8], horizontal_lines[9]
    x1_p3n, x2_p3n = vertical_lines[2], vertical_lines[3]
    celda_p3n = img[y1_p3n:y2_p3n, x1_p3n:x2_p3n]
    chars_p3n = analizar_celda(celda_p3n, th_min_area=10)

    #plt.figure(), plt.imshow(celda_p1n, cmap='gray'), plt.show() 
    
    # --- Mostrar Resultados ---
    plt.figure(figsize=(12, 8))
    plt.suptitle("Análisis de Celdas de Texto (Corregido)", fontsize=16)
    
    plt.subplot(4, 4, 1)
    plt.imshow(img[y1_nom:y2_nom, vertical_lines[1]:vertical_lines[3]], cmap='gray')
    plt.title(f"'Nombre y Apellido' (Detectados: {chars_nombre})")
    plt.axis('off')

    plt.subplot(4, 4, 2)
    plt.imshow(celda_edad, cmap='gray')
    plt.title(f"'Edad' (Detectados: {chars_edad})")
    plt.axis('off')

    plt.subplot(4, 4, 3)
    plt.imshow(img[y1_mail:y2_mail, vertical_lines[1]:vertical_lines[3]], cmap='gray')
    plt.title(f"'Mail' (Detectados: {chars_mail})")
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(celda_legajo, cmap='gray')
    plt.title(f"'Legajo' (Detectados: {chars_legajo})")
    plt.axis('off')

    plt.subplot(4, 4, 5)
    plt.imshow(celda_com, cmap='gray')
    plt.title(f"'comentarios' (Detectados: {chars_com})")
    plt.axis('off')

    # --- preguntas -----

    plt.subplot(4, 4, 6)
    plt.imshow(celda_p1s, cmap='gray')
    plt.title(f"'p1s' (Detectados: {chars_p1s})")
    plt.axis('off')

    plt.subplot(4, 4, 7)
    plt.imshow(celda_p1n, cmap='gray')
    plt.title(f"'p1n' (Detectados: {chars_p1n})")
    plt.axis('off')

    plt.subplot(4, 4, 8)
    plt.imshow(celda_p2s, cmap='gray')
    plt.title(f"'p2s' (Detectados: {chars_p2s})")
    plt.axis('off')

    plt.subplot(4, 4, 9)
    plt.imshow(celda_p2n, cmap='gray')
    plt.title(f"'p2n' (Detectados: {chars_p2n})")
    plt.axis('off')

    plt.subplot(4, 4, 10)
    plt.imshow(celda_p3s, cmap='gray')
    plt.title(f"'p3s' (Detectados: {chars_p3s})")
    plt.axis('off')

    plt.subplot(4, 4, 11)
    plt.imshow(celda_p3n, cmap='gray')
    plt.title(f"'p3n' (Detectados: {chars_p3n})")
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

except IndexError:
    print("\n--- ¡ERROR! ---")
    print("La detección de líneas falló. No se encontraron suficientes filas o columnas.")
    print(f"Líneas H: {len(horizontal_lines)}, Líneas V: {len(vertical_lines)}")