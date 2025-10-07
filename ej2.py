import cv2
import numpy as np
import matplotlib.pyplot as plt

#CARGAR LA IMAGEN Y UMBRALIZAR
form = cv2.imread("formulario_02.png", cv2.IMREAD_GRAYSCALE)
th = 160     
img_th = (form < th).astype(np.uint8)  #binarizacion

# DETECCION DE LINEAS
sum_rows = np.sum(img_th, axis=1)
sum_cols = np.sum(img_th, axis=0)

#umbral de deteccion
#Define que una "línea" será cualquier pico en la proyección horizontal 
# que alcance al menos el 60% del pico máximo encontrado. Esto filtra el ruido 
# y el texto que no son líneas completas.
th_row = 0.6 * np.max(sum_rows) 
th_col = 0.6 * np.max(sum_cols)

#Crea una matriz booleana. True indica las filas que 
# son parte de una línea horizontal detectada.
rows_detect = sum_rows > th_row
cols_detect = sum_cols > th_col

#deteccion de bordes
#Encuentra los puntos de inicio y fin de las líneas.
row_indices = np.where(np.diff(rows_detect.astype(int)) != 0)[0] 
col_indices = np.where(np.diff(cols_detect.astype(int)) != 0)[0] 
#rows_detect.astype(int): Convierte True/False a 1/0
#np.diff(): Calcula la diferencia entre píxeles adyacentes
#np.where(...) Obtiene los índices de píxel (coordenadas Y) donde ocurre este cambio.

print("Horizontales:", row_indices)
print("Verticales:", col_indices)


#RECORTE DE REGIONES

def crop_region(img, y1, y2, x1, x2, label=""):
    ''' 
    Define una función segura para extraer sub-matrices (celdas) de la imagen 
    binarizada. Seguridad: Usa max(0, y1) y min(img.shape[0], y2) para asegurarse 
    de que las coordenadas estén siempre dentro de los límites de la imagen, 
    evitando errores de índice.
    '''
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        print(f"[⚠️] {label} quedó vacío ({y1}:{y2}, {x1}:{x2})")
    return sub

# Definir celdas (ajustado)

nombre_celda = crop_region(img_th, row_indices[1]+2, row_indices[2]-2, col_indices[2]+2, col_indices[3]-2, "Nombre")
edad_celda   = crop_region(img_th, row_indices[2]+2, row_indices[3]-2, col_indices[2]+2, col_indices[3]-2, "Edad")
mail_celda   = crop_region(img_th, row_indices[3]+2, row_indices[4]-2, col_indices[2]+2, col_indices[3]-2, "Mail")
legajo_celda = crop_region(img_th, row_indices[4]+2, row_indices[5]-2, col_indices[2]+2, col_indices[3]-2, "Legajo")
comentarios_celda = crop_region(img_th, row_indices[10]+2, row_indices[11]-2, col_indices[2]+2, col_indices[3]-2, "Comentarios")

# Preguntas (ajuste manual)
preg1_si = crop_region(img_th, row_indices[7]+2, row_indices[8]-2, col_indices[2]+2, col_indices[3]-2, "P1_SI")
preg1_no = crop_region(img_th, row_indices[7]+2, row_indices[8]-2, col_indices[3]+2, col_indices[4]-2, "P1_NO")

preg2_si = crop_region(img_th, row_indices[8]+2, row_indices[9]-2, col_indices[2]+2, col_indices[3]-2, "P2_SI")
preg2_no = crop_region(img_th, row_indices[8]+2, row_indices[9]-2, col_indices[3]+2, col_indices[4]-2, "P2_NO")

preg3_si = crop_region(img_th, row_indices[9]+2, row_indices[10]-2, col_indices[2]+2, col_indices[3]-2, "P3_SI")
preg3_no = crop_region(img_th, row_indices[9]+2, row_indices[10]-2, col_indices[3]+2, col_indices[4]-2, "P3_NO")


# Contar componentes (seguro)

def contar_caracteres(celda_binaria, area_min=10):
    if celda_binaria.size == 0:
        return 0
    celda_binaria = (celda_binaria * 255).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(celda_binaria, 8, cv2.CV_32S)
    # Filtrar por área
    stats_filtrado = stats[stats[:, -1] > area_min]
    return max(len(stats_filtrado) - 1, 0)


# Validaciones

def validar_campo(nombre, cantidad, minimo, maximo):
    estado = "OK" if minimo <= cantidad <= maximo else "MAL"
    print(f"> {nombre}: {estado} ({cantidad} caracteres)")
    return estado

def validar_pregunta(nombre, si, no):
    s = contar_caracteres(si)
    n = contar_caracteres(no)

    # Regla: exactamente una marca en total y de un solo caracter
    if (s == 1 and n == 0) or (s == 0 and n == 1):
        estado = "OK"
    else:
        estado = "MAL"

    print(f"> {nombre}: {estado} (Si={s}, No={n})")
    return estado


# Ejecutar validaciones
nombre_val = validar_campo("Nombre y apellido", contar_caracteres(nombre_celda), 2, 25)
edad_val   = validar_campo("Edad", contar_caracteres(edad_celda), 2, 3)
mail_val   = validar_campo("Mail", contar_caracteres(mail_celda), 1, 25)
legajo_val = validar_campo("Legajo", contar_caracteres(legajo_celda), 8, 8)
coment_val = validar_campo("Comentarios", contar_caracteres(comentarios_celda), 1, 25)

p1_val = validar_pregunta("Pregunta 1", preg1_si, preg1_no)
p2_val = validar_pregunta("Pregunta 2", preg2_si, preg2_no)
p3_val = validar_pregunta("Pregunta 3", preg3_si, preg3_no)

print("\n--- RESULTADOS DEL FORMULARIO ---")
print(f"Nombre y apellido: {nombre_val}")
print(f"Edad: {edad_val}")
print(f"Mail: {mail_val}")
print(f"Legajo: {legajo_val}")
print(f"Pregunta 1: {p1_val}")
print(f"Pregunta 2: {p2_val}")
print(f"Pregunta 3: {p3_val}")
print(f"Comentarios: {coment_val}")
