# PROBLEMAS CON LA DETECCION DE COLUMNAS INTERNAS DEL FORMULARIO DE LAS PREGUNTAS
# PROBLEMAS PARA LA DETECCION DE COMENTARIOS.
# ESTE CÓDIGO MANTIENE UNA ROBUSTEZ PERFECTA PARA TODOS LOS CAMPOS DE EDAD, Mail y nombre
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen y umbralizar
form = cv2.imread("formulario_05.png", cv2.IMREAD_GRAYSCALE)
th = 160
img_th = (form < th).astype(np.uint8) #binarizacion

# Detección de líneas
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

# Obtener cambios (inicio/fin de líneas)
#deteccion de bordes
#Encuentra los puntos de inicio y fin de las líneas.
row_changes = np.where(np.diff(rows_detect.astype(int)) != 0)[0]
col_changes = np.where(np.diff(cols_detect.astype(int)) != 0)[0]
#rows_detect.astype(int): Convierte True/False a 1/0
#np.diff(): Calcula la diferencia entre píxeles adyacentes
#np.where(...) Obtiene los índices de píxel (coordenadas Y) donde ocurre este cambio.

# Agrupar líneas consecutivas (tomar el centro)
horizontal_lines = []
for i in range(0, len(row_changes)-1, 2):
    center = (row_changes[i] + row_changes[i+1]) // 2
    horizontal_lines.append(center)

vertical_lines = []
for i in range(0, len(col_changes)-1, 2):
    center = (col_changes[i] + col_changes[i+1]) // 2
    vertical_lines.append(center)

print(f"Líneas horizontales: {horizontal_lines}")
print(f"Líneas verticales: {vertical_lines}")

# ========== VALIDACIÓN DE NOMBRE Y APELLIDO ==========
print("\n" + "="*50)
print("VALIDACIÓN: NOMBRE Y APELLIDO")
print("="*50)

# Extraer el campo "Nombre y apellido" (fila 1, columna 2)
padding = 3 
#Define una pequeña separación (en píxeles) que se va a aplicar 
#en todos los lados para no recortar justo en el borde de la celda. 
#3 es un valor arbitrario: agrega 3 píxeles dentro del área definida 
# por las líneas para evitar incluir bordes, marcas o ruido.
y1_nombre = horizontal_lines[1] + padding #toma la seg línea hor, representa el borde sup de la fila 1.
y2_nombre = horizontal_lines[2] - padding #borde inferior de la fila 1
x1_nombre = vertical_lines[1] + padding #borde izquierdo de la columna 2. padding para no incluir la línea vertical en el recorte
x2_nombre = vertical_lines[2] - padding if len(vertical_lines) > 2 else vertical_lines[-1]
#el borde derecho de la columna 2, con padding hacia adentro. Si NO existe una tercera línea, 
# entonces usa la última coordenada disponible como extremo derecho.

# Recortar y obtener el nombre. 
nombre_region = form[y1_nombre:y2_nombre, x1_nombre:x2_nombre] 
#nueva imagen (submatriz) que contiene sólo la región interior de la celda correspondiente al campo “Nombre y apellido”.

# Obtener la imagen binarizada.(Píxeles que estén dentro de las letras pasan a valer 1)
_, binary_nombre = cv2.threshold(nombre_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos (posibles caracteres)
contours_nombre, _ = cv2.findContours(binary_nombre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos encontrados: {len(contours_nombre)}")

# Filtrar contornos pequeños (ruido) y obtener bounding boxes
min_area = 5  # Reducido aún más para detectar puntos
caracteres_nombre = []
for cnt in contours_nombre:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_nombre.append((x, y, w, h))

# Ordenar caracteres de izquierda a derecha
caracteres_nombre = sorted(caracteres_nombre, key=lambda c: c[0])

print(f"Caracteres detectados: {len(caracteres_nombre)}")

# Visualizar los caracteres detectados
if len(caracteres_nombre) > 0:
    num_chars = len(caracteres_nombre)
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
    
    for i, (x, y, w, h) in enumerate(caracteres_nombre):
        char_img = nombre_region[y:y+h, x:x+w]
        
        if num_chars == 1:
            axes.imshow(char_img, cmap='gray')
            axes.set_title(f'Char {i+1}')
            axes.axis('off')
        else:
            axes[i].imshow(char_img, cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f"Total: {num_chars} caracteres")
    plt.tight_layout()
    plt.show()

# Detectar espacios entre palabras
espacios_grandes_nombre = []
for i in range(len(caracteres_nombre)-1):
    x1_fin = caracteres_nombre[i][0] + caracteres_nombre[i][2]
    x2_inicio = caracteres_nombre[i+1][0]
    espacio = x2_inicio - x1_fin
    espacios_grandes_nombre.append(espacio)

print(f"Espacios entre caracteres: {espacios_grandes_nombre}")

# Un espacio es considerado separador de palabra si es significativamente mayor
if len(espacios_grandes_nombre) > 0:
    espacio_promedio = np.mean(espacios_grandes_nombre)
    umbral_palabra = espacio_promedio * 1.5
    
    num_palabras = 1
    for espacio in espacios_grandes_nombre:
        if espacio > umbral_palabra:
            num_palabras += 1
else:
    num_palabras = 1 if len(caracteres_nombre) > 0 else 0

print(f"Palabras detectadas: {num_palabras}")

# VALIDACIÓN
total_caracteres_nombre = len(caracteres_nombre)
print(f"Total caracteres: {total_caracteres_nombre}")

validacion_nombre_ok = True

# 1. Verificar que no supere 25 caracteres
if total_caracteres_nombre > 25:
    print(f"❌ ERROR: Supera 25 caracteres (tiene {total_caracteres_nombre})")
    validacion_nombre_ok = False
else:
    print(f"✓ Longitud válida: {total_caracteres_nombre} caracteres")

# 2. Verificar mínimo 2 palabras
if num_palabras < 2:
    print(f"❌ ERROR: Debe tener al menos 2 palabras (detectadas: {num_palabras})")
    validacion_nombre_ok = False
else:
    print(f"✓ Número de palabras válido: {num_palabras} palabras")

# 3. Verificar que el campo no esté vacío
if total_caracteres_nombre == 0:
    print(f"❌ ERROR: Campo vacío")
    validacion_nombre_ok = False

# Resultado final
if validacion_nombre_ok:
    print(f"\n✅ VALIDACIÓN EXITOSA")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA")

# Visualizar la región y los caracteres detectados
vis_nombre = cv2.cvtColor(nombre_region, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in caracteres_nombre:
    cv2.rectangle(vis_nombre, (x, y), (x+w, y+h), (0, 255, 0), 1)


# ========== VALIDACIÓN DE EDAD ==========
print("\n" + "="*50)
print("VALIDACIÓN: EDAD")
print("="*50)

# Extraer el campo "Edad" (fila 2, columna 2)
y1_edad = horizontal_lines[2] + padding
y2_edad = horizontal_lines[3] - padding
x1_edad = vertical_lines[1] + padding
x2_edad = vertical_lines[2] - padding if len(vertical_lines) > 2 else vertical_lines[-1]

# Recortar región de edad
edad_region = form[y1_edad:y2_edad, x1_edad:x2_edad]

# Obtener la imagen binarizada
_, binary_edad = cv2.threshold(edad_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos (posibles caracteres)
contours_edad, _ = cv2.findContours(binary_edad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos encontrados: {len(contours_edad)}")

# Filtrar contornos pequeños (ruido) y obtener bounding boxes
caracteres_edad = []
for cnt in contours_edad:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_edad.append((x, y, w, h))

# Ordenar caracteres de izquierda a derecha
caracteres_edad = sorted(caracteres_edad, key=lambda c: c[0])

print(f"Caracteres detectados: {len(caracteres_edad)}")

# Visualizar los caracteres detectados
if len(caracteres_edad) > 0:
    num_chars = len(caracteres_edad)
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
    
    for i, (x, y, w, h) in enumerate(caracteres_edad):
        char_img = edad_region[y:y+h, x:x+w]
        
        if num_chars == 1:
            axes.imshow(char_img, cmap='gray')
            axes.set_title(f'Char {i+1}')
            axes.axis('off')
        else:
            axes[i].imshow(char_img, cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f"Total: {num_chars} caracteres")
    plt.tight_layout()
    plt.show()

# VALIDACIÓN DE EDAD
total_caracteres_edad = len(caracteres_edad)
print(f"Total caracteres: {total_caracteres_edad}")

validacion_edad_ok = True

# 1. Verificar que tenga 2 o 3 caracteres
if total_caracteres_edad < 2 or total_caracteres_edad > 3:
    print(f"❌ ERROR: Debe tener 2 o 3 caracteres (tiene {total_caracteres_edad})")
    validacion_edad_ok = False
else:
    print(f"✓ Cantidad de caracteres válida: {total_caracteres_edad}")

# 2. Verificar que NO haya espacios grandes entre caracteres (deben ser consecutivos)
if len(caracteres_edad) > 1:
    espacios_edad = []
    for i in range(len(caracteres_edad)-1):
        x1_fin = caracteres_edad[i][0] + caracteres_edad[i][2]
        x2_inicio = caracteres_edad[i+1][0]
        espacio = x2_inicio - x1_fin
        espacios_edad.append(espacio)
    
    print(f"Espacios entre caracteres: {espacios_edad}")
    
    espacio_maximo = max(espacios_edad)
    umbral_espacio = 10
    
    if espacio_maximo > umbral_espacio:
        print(f"❌ ERROR: Hay espacios entre caracteres (máximo: {espacio_maximo} píxeles)")
        print(f"   Los caracteres deben ser consecutivos sin espacios")
        validacion_edad_ok = False
    else:
        print(f"✓ Caracteres consecutivos (espacio máximo: {espacio_maximo} píxeles)")

# 3. Verificar que el campo no esté vacío
if total_caracteres_edad == 0:
    print(f"❌ ERROR: Campo vacío")
    validacion_edad_ok = False

# Resultado final
if validacion_edad_ok:
    print(f"\n✅ VALIDACIÓN EXITOSA")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA")

# Visualizar la región y los caracteres detectados
vis_edad = cv2.cvtColor(edad_region, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in caracteres_edad:
    cv2.rectangle(vis_edad, (x, y), (x+w, y+h), (0, 255, 0), 1)


# ========== VALIDACIÓN DE MAIL ==========
print("\n" + "="*50)
print("VALIDACIÓN: MAIL")
print("="*50)

# Extraer el campo "Mail" (fila 3, columna 2)
y1_mail = horizontal_lines[3] + padding
y2_mail = horizontal_lines[4] - padding
x1_mail = vertical_lines[1] + padding
x2_mail = vertical_lines[2] - padding if len(vertical_lines) > 2 else vertical_lines[-1]

# Recortar región de mail
mail_region = form[y1_mail:y2_mail, x1_mail:x2_mail]

# Obtener la imagen binarizada
_, binary_mail = cv2.threshold(mail_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos (posibles caracteres)
contours_mail, _ = cv2.findContours(binary_mail, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos encontrados: {len(contours_mail)}")

# Filtrar contornos pequeños (ruido) y obtener bounding boxes
# Para mail usamos min_area muy bajo (solo para filtrar ruido extremo)
min_area_mail = 3
caracteres_mail = []
for cnt in contours_mail:
    area = cv2.contourArea(cnt)
    if area > min_area_mail:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_mail.append((x, y, w, h))

# Ordenar caracteres de izquierda a derecha
caracteres_mail = sorted(caracteres_mail, key=lambda c: c[0])

print(f"Caracteres detectados: {len(caracteres_mail)}")

# Visualizar los caracteres detectados
if len(caracteres_mail) > 0:
    num_chars = len(caracteres_mail)
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
    
    for i, (x, y, w, h) in enumerate(caracteres_mail):
        char_img = mail_region[y:y+h, x:x+w]
        
        if num_chars == 1:
            axes.imshow(char_img, cmap='gray')
            axes.set_title(f'Char {i+1}')
            axes.axis('off')
        else:
            axes[i].imshow(char_img, cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f"Total: {num_chars} caracteres")
    plt.tight_layout()
    plt.show()

# Detectar espacios entre palabras
espacios_grandes_mail = []
for i in range(len(caracteres_mail)-1):
    x1_fin = caracteres_mail[i][0] + caracteres_mail[i][2]
    x2_inicio = caracteres_mail[i+1][0]
    espacio = x2_inicio - x1_fin
    espacios_grandes_mail.append(espacio)

print(f"Espacios entre caracteres: {espacios_grandes_mail}")

# Un espacio es considerado separador de palabra si es significativamente mayor
if len(espacios_grandes_mail) > 0:
    espacio_promedio_mail = np.mean(espacios_grandes_mail)
    espacio_mediana_mail = np.median(espacios_grandes_mail)
    espacio_maximo_mail = max(espacios_grandes_mail)
    # Usar mediana en vez de promedio para ser más robusto ante outliers
    umbral_palabra_mail = espacio_mediana_mail * 2.5
    
    num_palabras_mail = 1
    for espacio in espacios_grandes_mail:
        if espacio > umbral_palabra_mail:
            num_palabras_mail += 1
    
    print(f"Espacio promedio: {espacio_promedio_mail:.2f}, Mediana: {espacio_mediana_mail:.2f}, Espacio máximo: {espacio_maximo_mail}, Umbral: {umbral_palabra_mail:.2f}")
else:
    num_palabras_mail = 1 if len(caracteres_mail) > 0 else 0

print(f"Palabras detectadas: {num_palabras_mail}")

# VALIDACIÓN
total_caracteres_mail = len(caracteres_mail)
print(f"Total caracteres: {total_caracteres_mail}")

validacion_mail_ok = True

# 1. Verificar que no supere 25 caracteres
if total_caracteres_mail > 25:
    print(f"❌ ERROR: Supera 25 caracteres (tiene {total_caracteres_mail})")
    validacion_mail_ok = False
else:
    print(f"✓ Longitud válida: {total_caracteres_mail} caracteres")

# 2. Verificar que contenga exactamente 1 palabra
if num_palabras_mail != 1:
    print(f"❌ ERROR: Debe contener exactamente 1 palabra (detectadas: {num_palabras_mail})")
    validacion_mail_ok = False
else:
    print(f"✓ Contiene 1 palabra")

# 3. Verificar que el campo no esté vacío
if total_caracteres_mail == 0:
    print(f"❌ ERROR: Campo vacío")
    validacion_mail_ok = False

# Resultado final
if validacion_mail_ok:
    print(f"\n✅ VALIDACIÓN EXITOSA")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA")

# Visualizar la región y los caracteres detectados
vis_mail = cv2.cvtColor(mail_region, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in caracteres_mail:
    cv2.rectangle(vis_mail, (x, y), (x+w, y+h), (0, 255, 0), 1)


# ========== VALIDACIÓN DE LEGAJO ==========
print("\n" + "="*50)
print("VALIDACIÓN: LEGAJO")
print("="*50)

# Extraer el campo "Legajo" (fila 4, columna 2)
y1_legajo = horizontal_lines[4] + padding
y2_legajo = horizontal_lines[5] - padding
x1_legajo = vertical_lines[1] + padding
x2_legajo = vertical_lines[2] - padding if len(vertical_lines) > 2 else vertical_lines[-1]

# Recortar región de legajo
legajo_region = form[y1_legajo:y2_legajo, x1_legajo:x2_legajo]

# Obtener la imagen binarizada
_, binary_legajo = cv2.threshold(legajo_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos (posibles caracteres)
contours_legajo, _ = cv2.findContours(binary_legajo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos encontrados: {len(contours_legajo)}")

# Filtrar contornos pequeños (ruido) y obtener bounding boxes
caracteres_legajo = []
for cnt in contours_legajo:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_legajo.append((x, y, w, h))

# Ordenar caracteres de izquierda a derecha
caracteres_legajo = sorted(caracteres_legajo, key=lambda c: c[0])

print(f"Caracteres detectados: {len(caracteres_legajo)}")

# Visualizar los caracteres detectados
if len(caracteres_legajo) > 0:
    num_chars = len(caracteres_legajo)
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
    
    for i, (x, y, w, h) in enumerate(caracteres_legajo):
        char_img = legajo_region[y:y+h, x:x+w]
        
        if num_chars == 1:
            axes.imshow(char_img, cmap='gray')
            axes.set_title(f'Char {i+1}')
            axes.axis('off')
        else:
            axes[i].imshow(char_img, cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f"Total: {num_chars} caracteres")
    plt.tight_layout()
    plt.show()

# Detectar espacios entre palabras
espacios_grandes_legajo = []
for i in range(len(caracteres_legajo)-1):
    x1_fin = caracteres_legajo[i][0] + caracteres_legajo[i][2]
    x2_inicio = caracteres_legajo[i+1][0]
    espacio = x2_inicio - x1_fin
    espacios_grandes_legajo.append(espacio)

print(f"Espacios entre caracteres: {espacios_grandes_legajo}")

# Un espacio es considerado separador de palabra si es significativamente mayor
if len(espacios_grandes_legajo) > 0:
    espacio_promedio_legajo = np.mean(espacios_grandes_legajo)
    umbral_palabra_legajo = espacio_promedio_legajo * 3.0
    
    num_palabras_legajo = 1
    for espacio in espacios_grandes_legajo:
        if espacio > umbral_palabra_legajo:
            num_palabras_legajo += 1
    
    print(f"Espacio promedio: {espacio_promedio_legajo:.2f}, Umbral: {umbral_palabra_legajo:.2f}")
else:
    num_palabras_legajo = 1 if len(caracteres_legajo) > 0 else 0

print(f"Palabras detectadas: {num_palabras_legajo}")

# VALIDACIÓN
total_caracteres_legajo = len(caracteres_legajo)
print(f"Total caracteres: {total_caracteres_legajo}")

validacion_legajo_ok = True

# 1. Verificar que tenga exactamente 8 caracteres
if total_caracteres_legajo != 8:
    print(f"❌ ERROR: Debe tener exactamente 8 caracteres (tiene {total_caracteres_legajo})")
    validacion_legajo_ok = False
else:
    print(f"✓ Cantidad de caracteres válida: {total_caracteres_legajo}")

# 2. Verificar que contenga exactamente 1 palabra
if num_palabras_legajo != 1:
    print(f"❌ ERROR: Debe contener exactamente 1 palabra (detectadas: {num_palabras_legajo})")
    validacion_legajo_ok = False
else:
    print(f"✓ Contiene 1 palabra")

# 3. Verificar que el campo no esté vacío
if total_caracteres_legajo == 0:
    print(f"❌ ERROR: Campo vacío")
    validacion_legajo_ok = False

# Resultado final
if validacion_legajo_ok:
    print(f"\n✅ VALIDACIÓN EXITOSA")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA")

# Visualizar la región y los caracteres detectados
vis_legajo = cv2.cvtColor(legajo_region, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in caracteres_legajo:
    cv2.rectangle(vis_legajo, (x, y), (x+w, y+h), (0, 255, 0), 1)


# ========== VALIDACIÓN DE COMENTARIOS ==========
print("\n" + "="*50)
print("VALIDACIÓN: COMENTARIOS")
print("="*50)

# Extraer el campo "Comentarios" (fila 10, columna 2)
y1_comentarios = horizontal_lines[10] + padding if len(horizontal_lines) > 10 else horizontal_lines[-1] + padding
y2_comentarios = horizontal_lines[11] - padding if len(horizontal_lines) > 11 else form.shape[0] - padding
x1_comentarios = vertical_lines[1] + padding
x2_comentarios = vertical_lines[2] - padding if len(vertical_lines) > 2 else vertical_lines[-1]

# Recortar región de comentarios
comentarios_region = form[y1_comentarios:y2_comentarios, x1_comentarios:x2_comentarios]

# Obtener la imagen binarizada
_, binary_comentarios = cv2.threshold(comentarios_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos (posibles caracteres)
contours_comentarios, _ = cv2.findContours(binary_comentarios, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contornos encontrados: {len(contours_comentarios)}")

# Filtrar contornos pequeños (ruido) y obtener bounding boxes
caracteres_comentarios = []
for cnt in contours_comentarios:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_comentarios.append((x, y, w, h))

# Ordenar caracteres de izquierda a derecha
caracteres_comentarios = sorted(caracteres_comentarios, key=lambda c: c[0])

print(f"Caracteres detectados: {len(caracteres_comentarios)}")

# Visualizar los caracteres detectados
if len(caracteres_comentarios) > 0:
    num_chars = len(caracteres_comentarios)
    fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
    
    for i, (x, y, w, h) in enumerate(caracteres_comentarios):
        char_img = comentarios_region[y:y+h, x:x+w]
        
        if num_chars == 1:
            axes.imshow(char_img, cmap='gray')
            axes.set_title(f'Char {i+1}')
            axes.axis('off')
        else:
            axes[i].imshow(char_img, cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f"Total: {num_chars} caracteres")
    plt.tight_layout()
    plt.show()

# Detectar espacios entre palabras
espacios_grandes_comentarios = []
for i in range(len(caracteres_comentarios)-1):
    x1_fin = caracteres_comentarios[i][0] + caracteres_comentarios[i][2]
    x2_inicio = caracteres_comentarios[i+1][0]
    espacio = x2_inicio - x1_fin
    espacios_grandes_comentarios.append(espacio)

print(f"Espacios entre caracteres: {espacios_grandes_comentarios}")

# Un espacio es considerado separador de palabra si es significativamente mayor
if len(espacios_grandes_comentarios) > 0:
    espacio_promedio_comentarios = np.mean(espacios_grandes_comentarios)
    umbral_palabra_comentarios = espacio_promedio_comentarios * 1.5
    
    num_palabras_comentarios = 1
    for espacio in espacios_grandes_comentarios:
        if espacio > umbral_palabra_comentarios:
            num_palabras_comentarios += 1
else:
    num_palabras_comentarios = 1 if len(caracteres_comentarios) > 0 else 0

print(f"Palabras detectadas: {num_palabras_comentarios}")

# VALIDACIÓN
total_caracteres_comentarios = len(caracteres_comentarios)
print(f"Total caracteres: {total_caracteres_comentarios}")

validacion_comentarios_ok = True

# 1. Verificar que no supere 25 caracteres
if total_caracteres_comentarios > 25:
    print(f"❌ ERROR: Supera 25 caracteres (tiene {total_caracteres_comentarios})")
    validacion_comentarios_ok = False
else:
    print(f"✓ Longitud válida: {total_caracteres_comentarios} caracteres")

# 2. Verificar que contenga al menos 1 palabra
if num_palabras_comentarios < 1:
    print(f"❌ ERROR: Debe contener al menos 1 palabra (detectadas: {num_palabras_comentarios})")
    validacion_comentarios_ok = False
else:
    print(f"✓ Número de palabras válido: {num_palabras_comentarios} palabras")

# 3. Verificar que el campo no esté vacío
if total_caracteres_comentarios == 0:
    print(f"❌ ERROR: Campo vacío")
    validacion_comentarios_ok = False

# Resultado final
if validacion_comentarios_ok:
    print(f"\n✅ VALIDACIÓN EXITOSA")
else:
    print(f"\n❌ VALIDACIÓN FALLIDA")

# Visualizar la región y los caracteres detectados
vis_comentarios = cv2.cvtColor(comentarios_region, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in caracteres_comentarios:
    cv2.rectangle(vis_comentarios, (x, y), (x+w, y+h), (0, 255, 0), 1)


# ========== DETECCIÓN DE LÍNEA VERTICAL EXTRA EN PREGUNTAS ==========
# La línea que separa Si/No es corta y no se detecta con el umbral global
# Detectamos líneas verticales solo en la región de preguntas

y_inicio_preguntas = horizontal_lines[5]  # Después de la fila Si/No
y_fin_preguntas = horizontal_lines[9]     # Después de Pregunta 3

region_preguntas = img_th[y_inicio_preguntas:y_fin_preguntas, :]
sum_cols_preguntas = np.sum(region_preguntas, axis=0)

th_col_preguntas = 0.5 * np.max(sum_cols_preguntas)
cols_detect_preguntas = sum_cols_preguntas > th_col_preguntas

col_changes_preguntas = np.where(np.diff(cols_detect_preguntas.astype(int)) != 0)[0]

# Obtener centros de líneas verticales adicionales
vertical_lines_preguntas = []
for i in range(0, len(col_changes_preguntas)-1, 2):
    center = (col_changes_preguntas[i] + col_changes_preguntas[i+1]) // 2
    vertical_lines_preguntas.append(center)

print(f"Líneas verticales en región de preguntas: {vertical_lines_preguntas}")

# Combinar con las líneas verticales globales y ordenar
all_vertical_lines = sorted(list(vertical_lines) + vertical_lines_preguntas)
print(f"Todas las líneas verticales: {all_vertical_lines}")

# ========== VALIDACIÓN DE PREGUNTAS 1, 2 Y 3 ==========
print("\n" + "="*50)
print("VALIDACIÓN: PREGUNTAS 1, 2 Y 3")
print("="*50)
print(f"Total líneas horizontales: {len(horizontal_lines)}")
print(f"Total líneas verticales: {len(vertical_lines)}")

# Las preguntas tienen 3 columnas: etiqueta, Si, No
# Necesitamos validar las columnas Si (col 2) y No (col 3) para cada pregunta

validacion_pregunta1_ok = True
validacion_pregunta2_ok = True
validacion_pregunta3_ok = True

# ========== PREGUNTA 1 ==========
print("\n--- PREGUNTA 1 ---")

# Usar las líneas combinadas: [borde_izq, col_etiquetas, col_Si, col_No, borde_der]
# all_vertical_lines debería tener al menos 5 elementos ahora
y1_p1_si = horizontal_lines[6] + padding
y2_p1_si = horizontal_lines[7] - padding
x1_p1_si = all_vertical_lines[2] + padding  # Columna Si
x2_p1_si = all_vertical_lines[3] - padding  # Hasta columna No

print(f"Coordenadas P1-Si: y[{y1_p1_si}:{y2_p1_si}], x[{x1_p1_si}:{x2_p1_si}]")

p1_si_region = form[y1_p1_si:y2_p1_si, x1_p1_si:x2_p1_si]

# Binarizar
_, binary_p1_si = cv2.threshold(p1_si_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p1_si, _ = cv2.findContours(binary_p1_si, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p1_si = []
for cnt in contours_p1_si:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p1_si.append((x, y, w, h))

num_chars_p1_si = len(caracteres_p1_si)

# Extraer celda "No" de Pregunta 1
x1_p1_no = all_vertical_lines[3] + padding  # Columna No
x2_p1_no = all_vertical_lines[4] - padding  # Hasta borde derecho

p1_no_region = form[y1_p1_si:y2_p1_si, x1_p1_no:x2_p1_no]

# Binarizar
_, binary_p1_no = cv2.threshold(p1_no_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p1_no, _ = cv2.findContours(binary_p1_no, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p1_no = []
for cnt in contours_p1_no:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p1_no.append((x, y, w, h))

num_chars_p1_no = len(caracteres_p1_no)

print(f"Pregunta 1 - Si: {num_chars_p1_si} caracteres")
print(f"Pregunta 1 - No: {num_chars_p1_no} caracteres")

# Validar que haya exactamente una celda marcada con un carácter
if (num_chars_p1_si == 1 and num_chars_p1_no == 0) or (num_chars_p1_si == 0 and num_chars_p1_no == 1):
    print(f"✓ Pregunta 1 válida: una única celda marcada")
elif num_chars_p1_si == 0 and num_chars_p1_no == 0:
    print(f"❌ ERROR: Pregunta 1 sin marcar (ambas celdas vacías)")
    validacion_pregunta1_ok = False
elif num_chars_p1_si > 0 and num_chars_p1_no > 0:
    print(f"❌ ERROR: Pregunta 1 con ambas celdas marcadas")
    validacion_pregunta1_ok = False
else:
    print(f"❌ ERROR: Pregunta 1 con múltiples caracteres en una celda")
    validacion_pregunta1_ok = False


# ========== PREGUNTA 2 ==========
print("\n--- PREGUNTA 2 ---")

# Extraer celda "Si" de Pregunta 2
y1_p2_si = horizontal_lines[7] + padding
y2_p2_si = horizontal_lines[8] - padding
x1_p2_si = all_vertical_lines[2] + padding
x2_p2_si = all_vertical_lines[3] - padding

p2_si_region = form[y1_p2_si:y2_p2_si, x1_p2_si:x2_p2_si]

# Binarizar
_, binary_p2_si = cv2.threshold(p2_si_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p2_si, _ = cv2.findContours(binary_p2_si, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p2_si = []
for cnt in contours_p2_si:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p2_si.append((x, y, w, h))

num_chars_p2_si = len(caracteres_p2_si)

# Extraer celda "No" de Pregunta 2
x1_p2_no = all_vertical_lines[3] + padding
x2_p2_no = all_vertical_lines[4] - padding

p2_no_region = form[y1_p2_si:y2_p2_si, x1_p2_no:x2_p2_no]

# Binarizar
_, binary_p2_no = cv2.threshold(p2_no_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p2_no, _ = cv2.findContours(binary_p2_no, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p2_no = []
for cnt in contours_p2_no:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p2_no.append((x, y, w, h))

num_chars_p2_no = len(caracteres_p2_no)

print(f"Pregunta 2 - Si: {num_chars_p2_si} caracteres")
print(f"Pregunta 2 - No: {num_chars_p2_no} caracteres")

# Validar que haya exactamente una celda marcada con un carácter
if (num_chars_p2_si == 1 and num_chars_p2_no == 0) or (num_chars_p2_si == 0 and num_chars_p2_no == 1):
    print(f"✓ Pregunta 2 válida: una única celda marcada")
elif num_chars_p2_si == 0 and num_chars_p2_no == 0:
    print(f"❌ ERROR: Pregunta 2 sin marcar (ambas celdas vacías)")
    validacion_pregunta2_ok = False
elif num_chars_p2_si > 0 and num_chars_p2_no > 0:
    print(f"❌ ERROR: Pregunta 2 con ambas celdas marcadas")
    validacion_pregunta2_ok = False
else:
    print(f"❌ ERROR: Pregunta 2 con múltiples caracteres en una celda")
    validacion_pregunta2_ok = False


# ========== PREGUNTA 3 ==========
print("\n--- PREGUNTA 3 ---")

# Extraer celda "Si" de Pregunta 3
y1_p3_si = horizontal_lines[8] + padding
y2_p3_si = horizontal_lines[9] - padding
x1_p3_si = all_vertical_lines[2] + padding
x2_p3_si = all_vertical_lines[3] - padding

p3_si_region = form[y1_p3_si:y2_p3_si, x1_p3_si:x2_p3_si]

# Binarizar
_, binary_p3_si = cv2.threshold(p3_si_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p3_si, _ = cv2.findContours(binary_p3_si, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p3_si = []
for cnt in contours_p3_si:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p3_si.append((x, y, w, h))

num_chars_p3_si = len(caracteres_p3_si)

# Extraer celda "No" de Pregunta 3
x1_p3_no = all_vertical_lines[3] + padding
x2_p3_no = all_vertical_lines[4] - padding

p3_no_region = form[y1_p3_si:y2_p3_si, x1_p3_no:x2_p3_no]

# Binarizar
_, binary_p3_no = cv2.threshold(p3_no_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Encontrar contornos
contours_p3_no, _ = cv2.findContours(binary_p3_no, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos
caracteres_p3_no = []
for cnt in contours_p3_no:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        caracteres_p3_no.append((x, y, w, h))

num_chars_p3_no = len(caracteres_p3_no)

print(f"Pregunta 3 - Si: {num_chars_p3_si} caracteres")
print(f"Pregunta 3 - No: {num_chars_p3_no} caracteres")

# Validar que haya exactamente una celda marcada con un carácter
if (num_chars_p3_si == 1 and num_chars_p3_no == 0) or (num_chars_p3_si == 0 and num_chars_p3_no == 1):
    print(f"✓ Pregunta 3 válida: una única celda marcada")
elif num_chars_p3_si == 0 and num_chars_p3_no == 0:
    print(f"❌ ERROR: Pregunta 3 sin marcar (ambas celdas vacías)")
    validacion_pregunta3_ok = False
elif num_chars_p3_si > 0 and num_chars_p3_no > 0:
    print(f"❌ ERROR: Pregunta 3 con ambas celdas marcadas")
    validacion_pregunta3_ok = False
else:
    print(f"❌ ERROR: Pregunta 3 con múltiples caracteres en una celda")
    validacion_pregunta3_ok = False


# ========== RESUMEN FINAL ==========
print("\n" + "="*50)
print("RESUMEN FINAL DE VALIDACIONES")
print("="*50)
print(f"Nombre y Apellido: {'✅ VÁLIDO' if validacion_nombre_ok else '❌ INVÁLIDO'}")
print(f"Edad: {'✅ VÁLIDO' if validacion_edad_ok else '❌ INVÁLIDO'}")
print(f"Mail: {'✅ VÁLIDO' if validacion_mail_ok else '❌ INVÁLIDO'}")
print(f"Legajo: {'✅ VÁLIDO' if validacion_legajo_ok else '❌ INVÁLIDO'}")
print(f"Comentarios: {'✅ VÁLIDO' if validacion_comentarios_ok else '❌ INVÁLIDO'}")
print(f"Pregunta 1: {'✅ VÁLIDO' if validacion_pregunta1_ok else '❌ INVÁLIDO'}")
print(f"Pregunta 2: {'✅ VÁLIDO' if validacion_pregunta2_ok else '❌ INVÁLIDO'}")
print(f"Pregunta 3: {'✅ VÁLIDO' if validacion_pregunta3_ok else '❌ INVÁLIDO'}")

if validacion_nombre_ok and validacion_edad_ok and validacion_mail_ok and validacion_legajo_ok and validacion_comentarios_ok and validacion_pregunta1_ok and validacion_pregunta2_ok and validacion_pregunta3_ok:
    print("\n🎉 FORMULARIO COMPLETAMENTE VÁLIDO")
else:
    print("\n⚠️ FORMULARIO CON ERRORES")

# Mostrar todas las visualizaciones juntas
cv2.imshow("Caracteres Detectados - Nombre", vis_nombre)
cv2.imshow("Caracteres Detectados - Edad", vis_edad)
cv2.imshow("Caracteres Detectados - Mail", vis_mail)
cv2.imshow("Caracteres Detectados - Legajo", vis_legajo)
cv2.imshow("Caracteres Detectados - Comentarios", vis_comentarios)
cv2.waitKey(0)
cv2.destroyAllWindows()











































