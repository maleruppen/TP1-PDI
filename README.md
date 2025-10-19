# Trabajo Práctico 1 - Procesamiento de Imágenes 

Este repositorio contiene las soluciones para los dos ejercicios del Trabajo Práctico N° 1 de Procesamiento de Imágenes.

## 📋 Prerrequisitos

* **Python 3**: Asegúrate de tener Python 3 instalado en tu sistema. Puedes descargarlo desde [python.org](https://www.python.org/).

---

## ⚙️ Configuración del Entorno

Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto.

1.  **Crear el entorno virtual:**
    Abre una terminal o línea de comandos en la carpeta del proyecto y ejecuta:
    ```bash
    python -m venv .venv
    ```
    *(Reemplaza `.venv` con el nombre que prefieras para tu entorno si lo deseas)*.

2.  **Activar el entorno virtual:**
    * **En Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
    * **En macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    Verás el nombre del entorno (ej. `(.venv)`) al principio de la línea de comandos, indicando que está activo.

3.  **Instalar las dependencias:**
    Con el entorno activado, instala las bibliotecas necesarias:
    ```bash
    pip install numpy matplotlib opencv-contrib-python
    ```

---

## ▶️ Ejecución de los Scripts

### Ejercicio 1: Ecualización Local de Histograma (`ej1.py`)

Este script aplica la técnica de ecualización local del histograma a una imagen para resaltar detalles.

1.  **Asegúrate de tener el archivo de imagen**: El script `ej1.py` espera encontrar el archivo `Imagen_con_detalles_escondidos.tif` en la misma carpeta.
2.  **Ejecuta el script:**
    ```bash
    python ej1.py
    ```
3.  **Salida**: El script mostrará varias ventanas con la imagen original y los resultados de la ecualización local con diferentes tamaños de ventana. Cierra las ventanas para finalizar. 

### Ejercicio 2: Validación de Formularios (`ej2.py`)

Este script procesa un conjunto de imágenes de formularios (`formulario_01.png` a `formulario_06.png`), valida cada campo según reglas específicas y genera un archivo CSV con los resultados.

1.  **Asegúrate de tener los archivos de formulario**: El script `ej2.py` espera encontrar los archivos `formulario_01.png`, `formulario_02.png`, ..., `formulario_06.png` en la misma carpeta.
2.  **Ejecuta el script:**
    ```bash
    python ej2.py
    ```
3.  **Salida**:
    * **Consola**: Imprimirá el resumen de validación para cada formulario. 📝
    * **Archivo CSV**: Creará (o sobrescribirá) un archivo llamado `resultados_formularios.csv` con el estado ('OK' o 'MAL') de cada campo para cada formulario. 
    * **Ventana de Gráficos**: Mostrará una ventana con recortes de las celdas analizadas para el último formulario procesado. Cierra la ventana para finalizar.

---

##  deactivate Desactivar el Entorno

Cuando termines de trabajar, puedes desactivar el entorno virtual simplemente ejecutando en la terminal:

```bash
deactivate
