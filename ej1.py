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

            # Histograma local
            #hist, bins = np.histogram(ventana.flatten(), 256, [0,256])
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])

            # Normalizar y calcular CDF
            histn = hist.astype(np.double) / ventana.size
            cdf = histn.cumsum()

            # Transformar píxel central
            valor = img[i,j]
            nuevo_valor = np.uint8(cdf[valor] * 255)

            salida[i,j] = nuevo_valor

    return salida

# Cargar imagen en escala de grises
img = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

# Aplicar con distintas ventanas
res_3x3 = ecualizacion_local(img, 3, 3)
res_15x15 = ecualizacion_local(img, 15, 15)
res_51x51 = ecualizacion_local(img, 60, 60)

# Mostrar resultados (como en el apunte)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2,2,2), plt.imshow(res_3x3, cmap='gray'), plt.title("Ventana 3x3")
plt.subplot(2,2,3), plt.imshow(res_15x15, cmap='gray'), plt.title("Ventana 15x15")
plt.subplot(2,2,4), plt.imshow(res_51x51, cmap='gray'), plt.title("Ventana 51x51")
plt.show()