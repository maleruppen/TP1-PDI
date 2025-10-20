import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local(img, M, N):
    """
    Ecualización local de histograma
    img : imagen en escala de grises
    M, N : dimensiones de la ventana
    """

    # Padding para los bordes
    pad_M, pad_N = M // 2, N // 2
    img_padded = cv2.copyMakeBorder(img,pad_M, pad_M, pad_N, pad_N, borderType=cv2.BORDER_REPLICATE)

    # Imagen de salida
    salida = np.zeros_like(img)

    # Recorrer la imagen pixel a pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extraer ventana
            ventana = img_padded[i:i+M, j:j+N]
            
            # Ecualización de la ventana. 
            ventana_ecualizada = cv2.equalizeHist(ventana)
            
            # El píxel central de la ventana ecualizada es nuestro nuevo valor
            salida[i,j] = ventana_ecualizada[pad_M, pad_N] 

    return salida

# Cargar imagen en escala de grises
img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

# Aplicar con distintas ventanas
res_3x3 = ecualizacion_local(img, 3, 3)
res_20x20 = ecualizacion_local(img, 20, 20)
res_200x200 = ecualizacion_local(img, 200, 200)

# Mostrar resultados (como en el apunte)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2,2,2), plt.imshow(res_3x3, cmap='gray'), plt.title("Ventana 3x3")
plt.subplot(2,2,3), plt.imshow(res_20x20, cmap='gray'), plt.title("Ventana 20x20")
plt.subplot(2,2,4), plt.imshow(res_200x200, cmap='gray'), plt.title("Ventana 200x200")
plt.savefig("ej1.png")
plt.show()
