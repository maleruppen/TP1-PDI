import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Tu código de Carga y Detección ---
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

# --- 2. Visualización de los Recortes (Crops) ---

# Crear una figura para mostrar todas las filas recortadas
# Se ajusta el tamaño para que se vean bien
plt.figure(figsize=(15, 8))
plt.suptitle("Filas Detectadas y Recortadas", fontsize=20)

# Iterar a través de las líneas horizontales para crear cada "rebanada" (slice)
for i in range(len(horizontal_lines) - 1):
    # Definir las coordenadas Y de inicio y fin de la fila
    y1 = horizontal_lines[i]
    y2 = horizontal_lines[i+1]
    
    # Recortar la fila de la imagen original
    # Usamos [y1:y2, :] para tomar todas las columnas entre esas dos filas
    fila_recortada = img[y1:y2, :]
    
    # Crear un subplot para cada fila
    # El layout es de 2 filas de subplots, 5 columnas
    plt.subplot(2, 5, i + 1)
    plt.imshow(fila_recortada, cmap='gray')
    plt.title(f"Fila {i+1} (Y: {y1} a {y2})")
    plt.axis('off') # Ocultar los ejes para una vista más limpia

# Ajustar el espaciado y mostrar la figura
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()





