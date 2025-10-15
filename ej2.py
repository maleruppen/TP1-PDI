import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_campo(form, y1, y2, x1, x2, nombre_campo, min_area=5, min_area_especial=None, 
                   num_caracteres_exacto=None, num_caracteres_rango=None, 
                   num_palabras_exacto=None, num_palabras_minimo=None,
                   caracteres_consecutivos=False, umbral_mediana=False, mostrar_letras=False):
    """
    Detecta y valida un campo del formulario.
    
    Parámetros:
    - form: imagen del formulario
    - y1, y2, x1, x2: coordenadas del campo
    - nombre_campo: nombre para mostrar en logs
    - min_area: área mínima de contorno
    - min_area_especial: área mínima especial (para mail)
    - num_caracteres_exacto: número exacto de caracteres requerido
    - num_caracteres_rango: tupla (min, max) de caracteres
    - num_palabras_exacto: número exacto de palabras
    - num_palabras_minimo: mínimo de palabras
    - caracteres_consecutivos: si True, verifica espacios pequeños
    - umbral_mediana: si True, usa mediana para detectar palabras
    - mostrar_letras: si True, muestra gráficos de caracteres individuales
    
    Retorna: (validacion_ok, caracteres, vis_region)
    """
    padding = 3
    print(f"\n{'='*50}")
    print(f"VALIDACIÓN: {nombre_campo.upper()}")
    print(f"{'='*50}")
    
    # Recortar región
    region = form[y1+padding:y2-padding, x1+padding:x2-padding]
    
    # Binarizar
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contornos encontrados: {len(contours)}")
    
    # Filtrar contornos
    area_umbral = min_area_especial if min_area_especial is not None else min_area
    caracteres = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_umbral:
            x, y, w, h = cv2.boundingRect(cnt)
            caracteres.append((x, y, w, h))
    
    # Ordenar de izquierda a derecha
    caracteres = sorted(caracteres, key=lambda c: c[0])
    print(f"Caracteres detectados: {len(caracteres)}")
    
    # Visualizar caracteres individuales (solo si mostrar_letras=True)
    if mostrar_letras and len(caracteres) > 0:
        num_chars = len(caracteres)
        fig, axes = plt.subplots(1, num_chars, figsize=(num_chars*2, 3))
        
        for i, (x, y, w, h) in enumerate(caracteres):
            char_img = region[y:y+h, x:x+w]
            
            if num_chars == 1:
                axes.imshow(char_img, cmap='gray')
                axes.set_title(f'Char {i+1}')
                axes.axis('off')
            else:
                axes[i].imshow(char_img, cmap='gray')
                axes[i].set_title(f'Char {i+1}')
                axes[i].axis('off')
        
        plt.suptitle(f"{nombre_campo} - Total: {num_chars} caracteres")
        plt.tight_layout()
        plt.show()
    
    # Detectar espacios y palabras
    espacios = []
    for i in range(len(caracteres)-1):
        x1_fin = caracteres[i][0] + caracteres[i][2]
        x2_inicio = caracteres[i+1][0]
        espacio = x2_inicio - x1_fin
        espacios.append(espacio)
    
    print(f"Espacios entre caracteres: {espacios}")
    
    # Contar palabras
    if len(espacios) > 0:
        if umbral_mediana:
            mediana = np.median(espacios)
            umbral = mediana * 2.5
            print(f"Mediana: {mediana:.2f}, Umbral: {umbral:.2f}")
        else:
            promedio = np.mean(espacios)
            umbral = promedio * 3.0 if num_palabras_exacto == 1 else promedio * 1.5
            print(f"Promedio: {promedio:.2f}, Umbral: {umbral:.2f}")
        
        num_palabras = 1
        for espacio in espacios:
            if espacio > umbral:
                num_palabras += 1
    else:
        num_palabras = 1 if len(caracteres) > 0 else 0
    
    print(f"Palabras detectadas: {num_palabras}")
    
    # VALIDACIÓN
    total_caracteres = len(caracteres)
    print(f"Total caracteres: {total_caracteres}")
    validacion_ok = True
    
    # Verificar número de caracteres
    if num_caracteres_exacto is not None:
        if total_caracteres != num_caracteres_exacto:
            print(f"❌ ERROR: Debe tener exactamente {num_caracteres_exacto} caracteres (tiene {total_caracteres})")
            validacion_ok = False
        else:
            print(f"✓ Cantidad de caracteres válida: {total_caracteres}")
    
    if num_caracteres_rango is not None:
        min_c, max_c = num_caracteres_rango
        if total_caracteres < min_c or total_caracteres > max_c:
            print(f"❌ ERROR: Debe tener entre {min_c} y {max_c} caracteres (tiene {total_caracteres})")
            validacion_ok = False
        else:
            print(f"✓ Longitud válida: {total_caracteres} caracteres")
    
    # Verificar número de palabras
    if num_palabras_exacto is not None:
        if num_palabras != num_palabras_exacto:
            print(f"❌ ERROR: Debe contener exactamente {num_palabras_exacto} palabra(s) (detectadas: {num_palabras})")
            validacion_ok = False
        else:
            print(f"✓ Contiene {num_palabras_exacto} palabra(s)")
    
    if num_palabras_minimo is not None:
        if num_palabras < num_palabras_minimo:
            print(f"❌ ERROR: Debe tener al menos {num_palabras_minimo} palabra(s) (detectadas: {num_palabras})")
            validacion_ok = False
        else:
            print(f"✓ Número de palabras válido: {num_palabras} palabras")
    
    # Verificar caracteres consecutivos (para edad)
    if caracteres_consecutivos and len(espacios) > 0:
        espacio_maximo = max(espacios)
        umbral_espacio = 10
        if espacio_maximo > umbral_espacio:
            print(f"❌ ERROR: Hay espacios entre caracteres (máximo: {espacio_maximo} píxeles)")
            validacion_ok = False
        else:
            print(f"✓ Caracteres consecutivos (espacio máximo: {espacio_maximo} píxeles)")
    
    # Verificar campo no vacío
    if total_caracteres == 0:
        print(f"❌ ERROR: Campo vacío")
        validacion_ok = False
    
    # Resultado
    if validacion_ok:
        print(f"\n✅ VALIDACIÓN EXITOSA")
    else:
        print(f"\n❌ VALIDACIÓN FALLIDA")
    
    # Crear visualización con rectángulos
    vis_region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in caracteres:
        cv2.rectangle(vis_region, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    return validacion_ok, caracteres, vis_region


# ========== CARGAR Y PROCESAR FORMULARIO ==========
form = cv2.imread("formulario_01.png", cv2.IMREAD_GRAYSCALE)
th = 160
img_th = (form < th).astype(np.uint8)

# Detección de líneas
sum_rows = np.sum(img_th, axis=1)
sum_cols = np.sum(img_th, axis=0)

th_row = 0.6 * np.max(sum_rows)
th_col = 0.6 * np.max(sum_cols)

rows_detect = sum_rows > th_row
cols_detect = sum_cols > th_col

row_changes = np.where(np.diff(rows_detect.astype(int)) != 0)[0]
col_changes = np.where(np.diff(cols_detect.astype(int)) != 0)[0]

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

# ========== VALIDAR CAMPOS DE TEXTO ==========
# Cambiar mostrar_letras=True para ver los gráficos de caracteres individuales

# Nombre y Apellido: 2+ palabras, máx 25 caracteres
validacion_nombre_ok, _, vis_nombre = detectar_campo(
    form, horizontal_lines[1], horizontal_lines[2], 
    vertical_lines[1], vertical_lines[2],
    "Nombre y Apellido",
    num_caracteres_rango=(1, 25),
    num_palabras_minimo=2,
    mostrar_letras=False
)

# Edad: 2-3 caracteres consecutivos
validacion_edad_ok, _, vis_edad = detectar_campo(
    form, horizontal_lines[2], horizontal_lines[3],
    vertical_lines[1], vertical_lines[2],
    "Edad",
    num_caracteres_rango=(2, 3),
    caracteres_consecutivos=True,
    mostrar_letras=False
)

# Mail: 1 palabra, máx 25 caracteres
validacion_mail_ok, _, vis_mail = detectar_campo(
    form, horizontal_lines[3], horizontal_lines[4],
    vertical_lines[1], vertical_lines[2],
    "Mail",
    min_area_especial=3,
    num_caracteres_rango=(1, 25),
    num_palabras_exacto=1,
    umbral_mediana=True,
    mostrar_letras=False
)

# Legajo: 8 caracteres, 1 palabra
validacion_legajo_ok, _, vis_legajo = detectar_campo(
    form, horizontal_lines[4], horizontal_lines[5],
    vertical_lines[1], vertical_lines[2],
    "Legajo",
    num_caracteres_exacto=8,
    num_palabras_exacto=1,
    mostrar_letras=False
)

# Comentarios: 1+ palabra, máx 25 caracteres
y1_com = horizontal_lines[10] if len(horizontal_lines) > 10 else horizontal_lines[-1]
y2_com = horizontal_lines[11] if len(horizontal_lines) > 11 else form.shape[0]
validacion_comentarios_ok, _, vis_comentarios = detectar_campo(
    form, y1_com, y2_com,
    vertical_lines[1], vertical_lines[2],
    "Comentarios",
    num_caracteres_rango=(1, 25),
    num_palabras_minimo=1,
    mostrar_letras=False
)

# ========== VALIDAR PREGUNTAS ==========
# Detectar líneas verticales en región de preguntas
y_inicio_preguntas = horizontal_lines[5]
y_fin_preguntas = horizontal_lines[9]

region_preguntas = img_th[y_inicio_preguntas:y_fin_preguntas, :]
sum_cols_preguntas = np.sum(region_preguntas, axis=0)

th_col_preguntas = 0.5 * np.max(sum_cols_preguntas)
cols_detect_preguntas = sum_cols_preguntas > th_col_preguntas

col_changes_preguntas = np.where(np.diff(cols_detect_preguntas.astype(int)) != 0)[0]

vertical_lines_preguntas = []
for i in range(0, len(col_changes_preguntas)-1, 2):
    center = (col_changes_preguntas[i] + col_changes_preguntas[i+1]) // 2
    vertical_lines_preguntas.append(center)

all_vertical_lines = sorted(list(vertical_lines) + vertical_lines_preguntas)
print(f"\nLíneas verticales completas: {all_vertical_lines}")

print("\n" + "="*50)
print("VALIDACIÓN: PREGUNTAS 1, 2 Y 3")
print("="*50)

def validar_pregunta(form, y1, y2, x_si_ini, x_si_fin, x_no_ini, x_no_fin, num_pregunta):
    padding = 3
    min_area = 5
    
    # Celda Si
    si_region = form[y1+padding:y2-padding, x_si_ini+padding:x_si_fin-padding]
    _, binary_si = cv2.threshold(si_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_si, _ = cv2.findContours(binary_si, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_si = sum(1 for cnt in contours_si if cv2.contourArea(cnt) > min_area)
    
    # Celda No
    no_region = form[y1+padding:y2-padding, x_no_ini+padding:x_no_fin-padding]
    _, binary_no = cv2.threshold(no_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_no, _ = cv2.findContours(binary_no, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_no = sum(1 for cnt in contours_no if cv2.contourArea(cnt) > min_area)
    
    print(f"\n--- PREGUNTA {num_pregunta} ---")
    print(f"Si: {num_si} caracteres, No: {num_no} caracteres")
    
    if (num_si == 1 and num_no == 0) or (num_si == 0 and num_no == 1):
        print(f"✓ Pregunta {num_pregunta} válida")
        return True
    elif num_si == 0 and num_no == 0:
        print(f"❌ ERROR: Pregunta {num_pregunta} sin marcar")
        return False
    elif num_si > 0 and num_no > 0:
        print(f"❌ ERROR: Pregunta {num_pregunta} con ambas celdas marcadas")
        return False
    else:
        print(f"❌ ERROR: Pregunta {num_pregunta} con múltiples caracteres")
        return False

validacion_pregunta1_ok = validar_pregunta(
    form, horizontal_lines[6], horizontal_lines[7],
    all_vertical_lines[2], all_vertical_lines[3],
    all_vertical_lines[3], all_vertical_lines[4], 1
)

validacion_pregunta2_ok = validar_pregunta(
    form, horizontal_lines[7], horizontal_lines[8],
    all_vertical_lines[2], all_vertical_lines[3],
    all_vertical_lines[3], all_vertical_lines[4], 2
)

validacion_pregunta3_ok = validar_pregunta(
    form, horizontal_lines[8], horizontal_lines[9],
    all_vertical_lines[2], all_vertical_lines[3],
    all_vertical_lines[3], all_vertical_lines[4], 3
)

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

if all([validacion_nombre_ok, validacion_edad_ok, validacion_mail_ok, validacion_legajo_ok,
        validacion_comentarios_ok, validacion_pregunta1_ok, validacion_pregunta2_ok, validacion_pregunta3_ok]):
    print("\n🎉 FORMULARIO COMPLETAMENTE VÁLIDO")
else:
    print("\n⚠️ FORMULARIO CON ERRORES")

# Mostrar visualizaciones
cv2.imshow("Caracteres - Nombre", vis_nombre)
cv2.imshow("Caracteres - Edad", vis_edad)
cv2.imshow("Caracteres - Mail", vis_mail)
cv2.imshow("Caracteres - Legajo", vis_legajo)
cv2.imshow("Caracteres - Comentarios", vis_comentarios)
cv2.waitKey(0)
cv2.destroyAllWindows()