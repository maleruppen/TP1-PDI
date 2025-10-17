import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---  Detección de Filas  ---
img = cv2.imread('formulario_03.png', cv2.IMREAD_GRAYSCALE) 
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