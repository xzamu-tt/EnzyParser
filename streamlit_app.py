import streamlit as st
import json
import os
import sys
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd

# Configuración de la página
st.set_page_config(layout="wide", page_title="EnzyParser Distilled Reader")

# Rutas por defecto (Ajusta según tu estructura)
DEFAULT_INPUT_DIR = Path("/Users/xzamu/Desktop/PETase-database/PDFs/NEWarticles")
DEFAULT_OUTPUT_DIR = Path("/Users/xzamu/Desktop/PETase-database/PDFs/Parsed-NEWarticles")

# --- FUNCIONES DE UTILIDAD ---

def load_paper_data(output_dir, paper_id):
    """Carga el JSON y prepara los datos en memoria."""
    json_path = Path(output_dir) / paper_id / f"{paper_id}_data.json"
    if not json_path.exists():
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_paper_data(output_dir, paper_id, data):
    """Guarda los cambios (reclasificaciones) en el disco."""
    json_path = Path(output_dir) / paper_id / f"{paper_id}_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    st.toast(f"Cambios guardados para {paper_id}", icon="💾")

def get_pdf_page_image(pdf_path, page_num):
    """Renderiza una página específica del PDF a imagen (cacheada)."""
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    except Exception as e:
        st.error(f"Error renderizando PDF: {e}")
        return None

# =============================================================================
# FUNCIÓN: crop_image
# =============================================================================
# PROPÓSITO: Recortar una región específica de una imagen de página PDF.
#
# ¿POR QUÉ ES NECESARIA LA CONVERSIÓN DE COORDENADAS?
# Docling usa un sistema de coordenadas con origen en la ESQUINA INFERIOR IZQUIERDA
# (como en matemáticas: Y aumenta hacia arriba).
#
# Pero PIL/Pillow (la librería de imágenes) usa origen en la ESQUINA SUPERIOR IZQUIERDA
# (como en pantallas: Y aumenta hacia abajo).
#
# Además, Docling usa una base de 72 DPI (puntos por pulgada), pero la imagen
# renderizada puede tener una resolución diferente, así que escalamos las coordenadas.
#
# ARGUMENTOS:
#   - full_page_img: La imagen completa de la página del PDF.
#   - bbox: Lista de 4 números [left, top, right, bottom] de Docling.
#           Nota: En Docling, "top" es el borde superior y "bottom" el inferior,
#           pero medidos desde abajo de la página.
#
# RETORNA:
#   - Una imagen recortada (solo la región del bbox).
#   - O None si algo falla.
# =============================================================================
def crop_image(full_page_img, bbox):
    """
    Recorta la imagen usando bbox [L, T, R, B] de Docling.
    Convierte del sistema de coordenadas de Docling al de PIL.
    """
    # -------------------------------------------------------------------------
    # Paso 1: Verificar que tenemos una imagen válida.
    # -------------------------------------------------------------------------
    if not full_page_img:
        return None
    
    # -------------------------------------------------------------------------
    # Paso 2: Verificar que el bbox tiene 4 valores.
    # -------------------------------------------------------------------------
    if not bbox or len(bbox) != 4:
        return None
    
    # -------------------------------------------------------------------------
    # Paso 3: Calcular factores de escala.
    # -------------------------------------------------------------------------
    # Docling asume páginas de 612x792 puntos (tamaño carta a 72 DPI).
    # La imagen real puede tener diferente resolución, así que escalamos.
    #
    # Ejemplo: Si la imagen es 1224 px de ancho y Docling usa 612 pts,
    #          entonces scale_x = 1224 / 612 = 2.0
    # -------------------------------------------------------------------------
    page_width_pts = 612.0   # Ancho estándar en puntos (72 DPI)
    page_height_pts = 792.0  # Alto estándar en puntos (72 DPI)
    
    scale_x = full_page_img.width / page_width_pts
    scale_y = full_page_img.height / page_height_pts
    
    # -------------------------------------------------------------------------
    # Paso 4: Extraer coordenadas del bbox.
    # -------------------------------------------------------------------------
    # l = left (izquierda), t = top (arriba), r = right (derecha), b = bottom (abajo)
    # -------------------------------------------------------------------------
    l, t, r, b = bbox
    
    # -------------------------------------------------------------------------
    # Paso 5: Convertir coordenadas de Docling a coordenadas de PIL.
    # -------------------------------------------------------------------------
    # Docling: Origen en esquina INFERIOR izquierda, Y crece hacia ARRIBA.
    # PIL:     Origen en esquina SUPERIOR izquierda, Y crece hacia ABAJO.
    #
    # Para convertir Y de Docling a PIL:
    #   y_pil = altura_pagina - y_docling
    #
    # Además, en el bbox de Docling:
    #   - "t" (top) es la coordenada Y del borde SUPERIOR del elemento
    #   - "b" (bottom) es la coordenada Y del borde INFERIOR del elemento
    #
    # Pero como Docling mide desde abajo, en realidad:
    #   - t tiene un valor MAYOR que b (está más arriba en la página)
    #
    # Para PIL necesitamos:
    #   - y1 = la coordenada Y más pequeña (borde superior en pantalla)
    #   - y2 = la coordenada Y más grande (borde inferior en pantalla)
    # -------------------------------------------------------------------------
    
    # Escalar coordenadas X (horizontales) - estas no cambian de orientación
    x1 = l * scale_x  # Borde izquierdo
    x2 = r * scale_x  # Borde derecho
    
    # Convertir y escalar coordenadas Y (verticales)
    # En Docling: t > b (porque t está más arriba, más lejos del origen inferior)
    # En PIL: necesitamos y1 < y2
    y1_docling = t * scale_y  # Posición Y del top en coords Docling escaladas
    y2_docling = b * scale_y  # Posición Y del bottom en coords Docling escaladas
    
    # Convertir a sistema PIL (invertir Y)
    y1_pil = full_page_img.height - y1_docling  # Top de Docling → más arriba en PIL
    y2_pil = full_page_img.height - y2_docling  # Bottom de Docling → más abajo en PIL
    
    # -------------------------------------------------------------------------
    # Paso 6: Asegurar que las coordenadas están en orden correcto.
    # -------------------------------------------------------------------------
    # PIL requiere: (left, upper, right, lower) donde left < right y upper < lower.
    # Usamos min/max para garantizar el orden correcto.
    # -------------------------------------------------------------------------
    crop_left = max(0, min(x1, x2))                        # No menor que 0
    crop_right = min(full_page_img.width, max(x1, x2))     # No mayor que ancho
    crop_upper = max(0, min(y1_pil, y2_pil))               # No menor que 0
    crop_lower = min(full_page_img.height, max(y1_pil, y2_pil))  # No mayor que alto
    
    # -------------------------------------------------------------------------
    # Paso 7: Verificar que el recorte tiene área válida.
    # -------------------------------------------------------------------------
    if crop_right <= crop_left or crop_lower <= crop_upper:
        # El recorte no tiene área (ancho o alto es 0 o negativo)
        return None
    
    # -------------------------------------------------------------------------
    # Paso 8: Realizar el recorte.
    # -------------------------------------------------------------------------
    # .crop() de PIL recibe una tupla: (left, upper, right, lower)
    # -------------------------------------------------------------------------
    return full_page_img.crop((crop_left, crop_upper, crop_right, crop_lower))


# =============================================================================
# FUNCIÓN: resolve_source_pdf_path
# =============================================================================
# PROPÓSITO: Encontrar la ruta del archivo PDF correcto para cualquier bloque.
#
# ¿POR QUÉ ES NECESARIA?
# Cuando extraemos información de un paper científico, puede venir de:
#   1. El PDF principal (ejemplo: "Almeida-2019.pdf")
#   2. Un PDF suplementario (ejemplo: "Table 1.pdf", "Supporting_Info.pdf")
#
# Cada bloque de texto/tabla/figura guarda el nombre de su archivo fuente
# en un campo llamado "source_file" o "source". Esta función busca ese
# archivo en el disco para que podamos mostrar el recorte original.
#
# ARGUMENTOS:
#   - block: Un diccionario (como una ficha) con la información del bloque.
#            Contiene campos como "source_file", "page_no", "bbox", etc.
#   - paper_id: El nombre del paper, ejemplo "Almeida-2019".
#   - input_dir: La carpeta raíz donde están todos los papers originales.
#
# RETORNA:
#   - Una cadena de texto (string) con la ruta completa al archivo PDF.
#   - O None (vacío) si el archivo no existe en el disco.
#
# EJEMPLO DE USO:
#   pdf_path = resolve_source_pdf_path(block, "Almeida-2019", "./NEWarticles")
#   # Podría retornar: "/Users/.../NEWarticles/Almeida-2019/Table 1.pdf"
# =============================================================================
def resolve_source_pdf_path(block, paper_id, input_dir):
    """
    Encuentra la ruta completa al archivo PDF fuente de un bloque.
    
    Busca primero el campo 'source_file', luego 'source'.
    Si no existe, asume que es el PDF principal del paper.
    """
    
    # -------------------------------------------------------------------------
    # PASO 1: Obtener el nombre del archivo fuente del bloque.
    # -------------------------------------------------------------------------
    # Los bloques extraídos guardan el nombre de su archivo origen.
    # Usamos ".get()" que es un método seguro - si el campo no existe,
    # retorna None en lugar de causar un error.
    #
    # El operador "or" funciona así:
    #   - Si el primer valor existe y no está vacío, lo usa.
    #   - Si no, intenta con el segundo valor.
    # Es como decir: "Dame source_file, o si no existe, dame source".
    # -------------------------------------------------------------------------
    source_file = block.get("source_file") or block.get("source")
    
    # -------------------------------------------------------------------------
    # PASO 2: Si no hay nombre de fuente, asumimos que es el PDF principal.
    # -------------------------------------------------------------------------
    # La convención es que el PDF principal tiene el mismo nombre que
    # la carpeta del paper. Por ejemplo:
    #   Carpeta: Almeida-2019/
    #   PDF:     Almeida-2019.pdf
    #
    # f"..." es una "f-string" (formatted string). Las llaves {} se
    # reemplazan por el valor de la variable. Ejemplo:
    #   Si paper_id = "Almeida-2019"
    #   Entonces f"{paper_id}.pdf" = "Almeida-2019.pdf"
    # -------------------------------------------------------------------------
    if not source_file:
        source_file = f"{paper_id}.pdf"
    
    # -------------------------------------------------------------------------
    # PASO 3: Verificar que el archivo es un PDF (no Excel, CSV, etc.)
    # -------------------------------------------------------------------------
    # Solo podemos mostrar recortes de archivos PDF.
    # .lower() convierte a minúsculas para comparar sin importar mayúsculas.
    # .endswith(".pdf") verifica si el nombre termina en ".pdf".
    # -------------------------------------------------------------------------
    if not source_file.lower().endswith(".pdf"):
        return None  # No es un PDF, no podemos mostrar recorte
    
    # -------------------------------------------------------------------------
    # PASO 4: Construir la ruta completa al archivo.
    # -------------------------------------------------------------------------
    # Path() es una clase de Python que maneja rutas de archivos.
    # El operador "/" une partes de la ruta de forma segura.
    # Funciona en Windows, Mac y Linux sin problemas.
    #
    # Ejemplo:
    #   Path("./NEWarticles") / "Almeida-2019" / "Table 1.pdf"
    #   Resultado: ./NEWarticles/Almeida-2019/Table 1.pdf
    # -------------------------------------------------------------------------
    full_path = Path(input_dir) / paper_id / source_file
    
    # -------------------------------------------------------------------------
    # PASO 5: Verificar que el archivo existe en el disco.
    # -------------------------------------------------------------------------
    # .exists() es un método de Path que retorna:
    #   - True: si el archivo está en el disco
    #   - False: si no existe
    #
    # str(full_path) convierte el objeto Path a texto simple,
    # porque otras funciones esperan texto, no objetos Path.
    # -------------------------------------------------------------------------
    if full_path.exists():
        return str(full_path)
    
    # -------------------------------------------------------------------------
    # PASO 6: El archivo no existe, retornar None.
    # -------------------------------------------------------------------------
    # None es el valor especial de Python que significa "nada" o "vacío".
    # Las funciones que llamen a esta verificarán si el resultado es None
    # para saber si deben mostrar un mensaje de error.
    # -------------------------------------------------------------------------
    return None


# --- LÓGICA DEL FEED ---


def merge_and_sort_blocks(data):
    """Fusiona textos, artefactos y TABLAS en una sola lista cronológica."""
    blocks = []
    
    # 1. Texto
    for seg in data.get("text_segments", []):
        seg["block_type"] = "text"
        blocks.append(seg)
        
    # 2. Artefactos (Figuras)
    for art in data.get("artifacts", []):
        art["block_type"] = "artifact"
        blocks.append(art)

    # 3. Tablas (PDF y Excel)
    for tbl in data.get("tables_data", []):
        tbl["block_type"] = "table"
        # Asegurar compatibilidad de ordenamiento para suplementarios (page 0)
        if "page" in tbl and "page_no" not in tbl: 
            tbl["page_no"] = tbl["page"]
        blocks.append(tbl)
        
    # Ordenar por Página -> Posición Y (Arriba a Abajo) -> Posición X
    # Nota: Usamos .get con defaults seguros para evitar crash con Excels sin bbox
    blocks.sort(key=lambda x: (
        x.get("page_no", 0), 
        x.get("bbox", [0,0,0,0])[1] if x.get("bbox") else 0, 
        x.get("bbox", [0,0,0,0])[0] if x.get("bbox") else 0
    ))
    
    return blocks


def group_blocks_by_source(data, paper_id: str):
    """
    Groups blocks by their source file, with Main PDF first.
    Returns: OrderedDict {"MAIN: Paper.pdf": [...], "SUPP: data.xlsx": [...]}
    """
    from collections import OrderedDict
    
    # Get all blocks first
    all_blocks = merge_and_sort_blocks(data)
    
    # Identify the main PDF filename
    main_pdf_name = f"{paper_id}.pdf"
    
    # Group by source
    groups = {}
    for block in all_blocks:
        # Determine source file
        source = block.get("source_file") or block.get("source") or main_pdf_name
        
        if source not in groups:
            groups[source] = []
        groups[source].append(block)
    
    # Sort: Main PDF first, then supplementary files alphabetically
    sorted_groups = OrderedDict()
    
    # Main PDF first
    if main_pdf_name in groups:
        sorted_groups[f"MAIN: {main_pdf_name}"] = groups.pop(main_pdf_name)
    
    # Then supplementary files sorted alphabetically
    for source in sorted(groups.keys()):
        sorted_groups[f"SUPP: {source}"] = groups[source]
    
    return sorted_groups

# --- COMPONENTES DE UI ---

def render_block(block, idx, pdf_path_str, paper_data, output_dir, ordered_ids, source_name="default"):
    """Renderiza una tarjeta individual (Texto, Figura o Tabla)."""
    
    block_id = block.get("id")
    is_important = block.get("has_quantitative_data", False)
    
    # Check if this block is in the expanded set
    if "expanded_ids" not in st.session_state:
        st.session_state.expanded_ids = set()
    
    is_expanded_context = block_id in st.session_state.expanded_ids
    
    if not is_important and not is_expanded_context:
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
        
        type_label = block.get('type', block['block_type']).upper()
        page_label = f"Pág {block.get('page_no', '-')}"
        
        with c1:
            if is_important:
                st.markdown(f"**:sparkles: RELEVANTE** | `{type_label}` | {page_label}")
            else:
                st.markdown(f"*:grey[Contexto Expandido]* | `{type_label}` | {page_label}")

        with c2:
            if not is_important:
                if st.button("📌 Marcar Útil", key=f"mark_{source_name}_{idx}_{block_id}"):
                    block["has_quantitative_data"] = True
                    save_paper_data(output_dir, st.session_state.selected_paper_id, paper_data)
                    st.rerun()
            else:
                if st.button("🗑️ Descartar", key=f"unmark_{source_name}_{idx}_{block_id}"):
                    block["has_quantitative_data"] = False
                    save_paper_data(output_dir, st.session_state.selected_paper_id, paper_data)
                    st.rerun()

        with c3:
            # Toggle de vista (Solo útil para texto/tablas PDF, no imágenes)
            if block["block_type"] != "artifact":
                view_mode = st.toggle("Ver Original", key=f"view_{source_name}_{idx}_{block_id}")
            else:
                view_mode = False

        # --- RENDERIZADO POR TIPO ---
        
        # CASO 1: FIGURAS (Artifacts)
        if block["block_type"] == "artifact":
            rel_path = block.get("file_path")
            if rel_path:
                # Manejo robusto de rutas relativas/absolutas
                abs_path = Path(output_dir) / st.session_state.selected_paper_id / "artifacts" / Path(rel_path).name
                if not abs_path.exists():
                    abs_path = Path(os.getcwd()) / rel_path
                if not abs_path.exists():
                    abs_path = Path(rel_path)
                
                if abs_path.exists():
                    st.image(str(abs_path), caption=block.get("caption", "Figura"), width="stretch")
                else:
                    st.error(f"🖼️ Imagen no encontrada: {rel_path}")
            
            if block.get("vlm_description"):
                with st.expander("Análisis IA"):
                    st.write(block["vlm_description"])

        # CASO 2: TABLAS (PDF y Excel)
        elif block["block_type"] == "table":
            caption = block.get("caption", "")
            notes = block.get("table_notes", "")
            
            if caption: 
                st.markdown(f"**Tabla:** {caption}")
            
            # 1. Si tenemos la imagen extraída por Docling, la mostramos PRIMERO (fuente de verdad)
            table_img_path = block.get("image_path")
            if table_img_path:
                # Intentar múltiples rutas posibles
                abs_path = Path(output_dir) / st.session_state.selected_paper_id / "artifacts" / Path(table_img_path).name
                if not abs_path.exists():
                    abs_path = Path(table_img_path)
                
                if abs_path.exists():
                    st.image(str(abs_path), caption="Recorte Exacto de Tabla", use_container_width=True)
            # -----------------------------------------------------------------
            # FALLBACK: Recorte manual del PDF cuando no hay imagen guardada.
            # -----------------------------------------------------------------
            # Esto se ejecuta solo si:
            #   - view_mode es True (el usuario activó "Ver Original")
            #   - El bloque tiene número de página (page_no)
            #   - El bloque tiene coordenadas (bbox = bounding box)
            # -----------------------------------------------------------------
            elif view_mode and block.get("page_no") and block.get("bbox"):
                # -------------------------------------------------------------
                # PASO 1: Encontrar el archivo PDF correcto para este bloque.
                # -------------------------------------------------------------
                # Usamos nuestra nueva función que busca el archivo real.
                # 
                # ARGUMENTOS:
                #   block: La información del bloque actual (tiene source_file)
                #   selected_paper_id: El nombre del paper (ej: "Almeida-2019")
                #   DEFAULT_INPUT_DIR: Carpeta donde están los papers originales
                # -------------------------------------------------------------
                actual_pdf_path = resolve_source_pdf_path(
                    block,                                    # El bloque actual
                    st.session_state.selected_paper_id,       # Nombre del paper
                    DEFAULT_INPUT_DIR                         # Carpeta de entrada
                )
                
                # -------------------------------------------------------------
                # PASO 2: Verificar que encontramos el archivo PDF.
                # -------------------------------------------------------------
                # actual_pdf_path será None si:
                #   - El archivo no existe en el disco
                #   - El archivo no es un PDF (es Excel, CSV, etc.)
                # -------------------------------------------------------------
                if actual_pdf_path:
                    # ---------------------------------------------------------
                    # PASO 3: Renderizar la página específica del PDF.
                    # ---------------------------------------------------------
                    # get_pdf_page_image() convierte una página PDF a imagen.
                    # block["page_no"] indica qué página queremos ver.
                    # ---------------------------------------------------------
                    page_img = get_pdf_page_image(actual_pdf_path, block["page_no"])
                    
                    if page_img:
                        # -----------------------------------------------------
                        # PASO 4: Recortar la imagen usando las coordenadas.
                        # -----------------------------------------------------
                        # block["bbox"] contiene [izquierda, arriba, derecha, abajo]
                        # Son las coordenadas exactas de donde está la tabla.
                        # crop_image() recorta solo esa región de la página.
                        # -----------------------------------------------------
                        crop = crop_image(page_img, block["bbox"])
                        
                        # -----------------------------------------------------
                        # PASO 5: Mostrar la imagen recortada en la pantalla.
                        # -----------------------------------------------------
                        # st.image() es una función de Streamlit que muestra imágenes.
                        # caption= es el texto que aparece debajo de la imagen.
                        # use_container_width=True hace que use todo el ancho.
                        # -----------------------------------------------------
                        st.image(crop, caption="📍 Recorte del PDF Original", use_container_width=True)
                else:
                    # ---------------------------------------------------------
                    # El archivo fuente no se encontró, mostrar advertencia.
                    # ---------------------------------------------------------
                    source_name_display = block.get("source_file") or block.get("source") or "desconocido"
                    st.warning(f"⚠️ No se encontró el archivo fuente: {source_name_display}")
            
            # 2. Toggle para ver los datos extraídos (OCR)
            with st.expander("📊 Ver Datos Extraídos (OCR)", expanded=False):
                if "data" in block and block["data"]:
                    df = pd.DataFrame(block["data"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No se pudo extraer texto estructurado de esta tabla.")
            
            if notes: 
                st.caption(f"📝 *Notas:* {notes}")

        # CASO 3: TEXTO
        elif block["block_type"] == "text":
            # -----------------------------------------------------------------
            # MODO "VER ORIGINAL": Mostrar el recorte del PDF.
            # -----------------------------------------------------------------
            # Si el usuario activó el toggle "Ver Original", mostramos la
            # región exacta del PDF donde está este texto.
            # -----------------------------------------------------------------
            if view_mode:
                # -------------------------------------------------------------
                # PASO 1: Encontrar el archivo PDF correcto para este bloque.
                # -------------------------------------------------------------
                # Los bloques de texto siempre vienen de PDFs, pero pueden ser
                # del PDF principal o de un PDF suplementario. Esta función
                # busca el archivo correcto basándose en el campo "source_file".
                # -------------------------------------------------------------
                actual_pdf_path = resolve_source_pdf_path(
                    block,                                    # El bloque de texto
                    st.session_state.selected_paper_id,       # Nombre del paper
                    DEFAULT_INPUT_DIR                         # Carpeta de entrada
                )
                
                # -------------------------------------------------------------
                # PASO 2: Si encontramos el PDF, mostrar el recorte.
                # -------------------------------------------------------------
                if actual_pdf_path:
                    # ---------------------------------------------------------
                    # PASO 3: Renderizar la página del PDF como imagen.
                    # ---------------------------------------------------------
                    # Usamos la función get_pdf_page_image que:
                    #   1. Abre el archivo PDF
                    #   2. Va a la página indicada (page_no)
                    #   3. Convierte esa página a una imagen
                    # ---------------------------------------------------------
                    page_img = get_pdf_page_image(actual_pdf_path, block["page_no"])
                    
                    if page_img:
                        # -----------------------------------------------------
                        # PASO 4: Recortar usando las coordenadas bbox.
                        # -----------------------------------------------------
                        # bbox significa "bounding box" (caja delimitadora).
                        # Es un rectángulo que define exactamente dónde está
                        # el texto en la página: [izquierda, arriba, derecha, abajo]
                        #
                        # Docling extrae estas coordenadas cuando procesa el PDF,
                        # así sabemos exactamente dónde estaba cada elemento.
                        # -----------------------------------------------------
                        crop = crop_image(page_img, block["bbox"])
                        
                        # Mostrar la imagen recortada con un caption descriptivo
                        st.image(crop, caption="📍 Recorte original del PDF")
                    else:
                        st.warning("⚠️ No se pudo renderizar la página del PDF.")
                else:
                    # ---------------------------------------------------------
                    # El archivo fuente no existe o no es un PDF.
                    # ---------------------------------------------------------
                    source_name_display = block.get("source_file") or block.get("source") or "principal"
                    st.warning(f"⚠️ No se encontró el archivo fuente: {source_name_display}")
            else:
                if is_important:
                    st.markdown(block["text"])
                else:
                    st.markdown(f"<span style='color:grey'>{block['text']}</span>", unsafe_allow_html=True)

        # Context expansion buttons
        sc1, sc2 = st.columns(2)
        
        # Find current index in ordered list
        try:
            current_idx = ordered_ids.index(block_id)
        except ValueError:
            current_idx = idx
        
        # Previous context button
        if current_idx > 0:
            prev_id = ordered_ids[current_idx - 1]
            if sc1.button("⬆️ Contexto Previo", key=f"prev_{source_name}_{idx}_{block_id}", help="Mostrar bloque anterior"):
                st.session_state.expanded_ids.add(prev_id)
                st.rerun()
        else:
            sc1.button("⬆️ Contexto Previo", key=f"prev_{source_name}_{idx}_{block_id}", disabled=True, help="No hay bloque anterior")
        
        # Next context button
        if current_idx < len(ordered_ids) - 1:
            next_id = ordered_ids[current_idx + 1]
            if sc2.button("⬇️ Contexto Posterior", key=f"next_{source_name}_{idx}_{block_id}", help="Mostrar bloque siguiente"):
                st.session_state.expanded_ids.add(next_id)
                st.rerun()
        else:
            sc2.button("⬇️ Contexto Posterior", key=f"next_{source_name}_{idx}_{block_id}", disabled=True, help="No hay bloque siguiente")


# --- GUI UTILS ---

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

def select_folder_native():
    """Abre un diálogo nativo del sistema para seleccionar carpeta."""
    if not HAS_TKINTER:
        st.error("El módulo `tkinter` no está instalado en este sistema. Usa la entrada de texto o instala python-tk.")
        return None
        
    try:
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal
        root.wm_attributes('-topmost', 1)  # Traer al frente
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_path
    except Exception as e:
        st.error(f"No se pudo abrir el selector de carpetas: {e}")
        return None

def folder_picker_ui(label: str, default_path: str, key: str) -> str:
    """Componente UI que usa el selector nativo."""
    st.markdown(f"**{label}**")
    
    col1, col2 = st.columns([0.8, 0.2])
    
    # Session state management
    state_key = f"path_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(DEFAULT_INPUT_DIR) if "input" in key else str(DEFAULT_OUTPUT_DIR)

    with col1:
        # Permite edición manual también
        new_path = st.text_input(
            "Ruta",
            value=st.session_state[state_key],
            key=f"input_{key}",
            label_visibility="collapsed"
        )
        # Update state if manually changed
        if new_path != st.session_state[state_key]:
            st.session_state[state_key] = new_path

    with col2:
        if st.button("📂 Buscar", key=f"btn_{key}", use_container_width=True):
            selected = select_folder_native()
            if selected:
                st.session_state[state_key] = selected
                st.rerun()

    return st.session_state[state_key]


def tab_execution():
    """Pestaña de Ejecución del Parser."""
    st.header("🚀 Ejecutar Parser")
    st.caption("Procesa papers desde cualquier carpeta del disco con logs en tiempo real.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_dir = folder_picker_ui(
            "📂 Carpeta de Entrada (Papers)",
            str(DEFAULT_INPUT_DIR),
            "input"
        )
    
    with col2:
        output_dir = folder_picker_ui(
            "💾 Carpeta de Salida (JSONs)",
            str(DEFAULT_OUTPUT_DIR),
            "output"
        )
    
    st.divider()
    
    # Store output dir in session for review tab
    st.session_state.output_dir = output_dir
    
    # Validation
    input_path = Path(input_dir)
    if not input_path.exists():
        st.warning(f"⚠️ La carpeta de entrada no existe: {input_dir}")
        return
    
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    st.info(f"📁 Encontrados {len(subdirs)} subdirectorios en la carpeta de entrada.")
    
    # --- STAGE SELECTION: Two-Stage Processing ---
    st.subheader("Procesamiento en Dos Etapas")
    
    # --- STAGE 1: Docling Extraction ---
    st.markdown("### ⚙️ Etapa 1: Extracción con Docling")
    st.caption("Parsea PDFs, extrae texto/figuras/tablas. **No usa LLM.**")
    
    # Mode selection for Stage 1
    mode = st.radio(
        "Modo de extracción:",
        ["📦 Todos los Artículos", "📄 Un Solo Artículo"],
        horizontal=True,
        key="stage1_mode"
    )
    
    selected_folder = None
    if mode == "📄 Un Solo Artículo":
        if not subdirs:
            st.warning("No hay subdirectorios para procesar.")
        else:
            folder_names = sorted([d.name for d in subdirs])
            selected_folder = st.selectbox(
                "Selecciona el artículo:",
                folder_names,
                key="single_folder_select"
            )
    
    # Force rerun option
    force_rerun = st.checkbox("🔄 Forzar re-procesamiento (Sobreescribir existentes)", value=False)
    
    if st.button("▶️ Ejecutar Extracción (Sin LLM)", type="secondary", use_container_width=True):
        try:
            from enzyme_parser import EnzymeParser
        except ImportError:
            st.error("❌ No se pudo importar EnzymeParser.")
            return
        
        log_container = st.empty()
        logs = []
        
        with st.spinner("Extrayendo con Docling..."):
            try:
                parser = EnzymeParser(input_dir, output_dir)
                
                if selected_folder:
                    folder_path = input_path / selected_folder
                    logs.append(f"🔬 Extrayendo artículo: {selected_folder}")
                    log_container.code("\n".join(logs), language="")
                    
                    for msg in parser._process_paper_folder_streaming(folder_path, skip_llm=True, force_rerun=force_rerun):
                        logs.append(msg)
                        log_container.code("\n".join(logs), language="")
                else:
                    for msg in parser.process_all_streaming(skip_llm=True, force_rerun=force_rerun):
                        logs.append(msg)
                        log_container.code("\n".join(logs), language="")
                
                st.success("✅ Extracción completada! Revisa los JSONs antes de clasificar.")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.divider()
    
    # --- STAGE 2: LLM Classification ---
    st.markdown("### 🧠 Etapa 2: Clasificación con LLM")
    st.caption("Ejecuta el clasificador LLM sobre JSONs existentes en la carpeta de salida.")
    
    # Count papers in output
    output_path = Path(output_dir)
    if output_path.exists():
        paper_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        st.info(f"📁 {len(paper_dirs)} papers encontrados en carpeta de salida")
    else:
        st.warning("La carpeta de salida no existe.")
        paper_dirs = []
    
    if st.button("▶️ Ejecutar Clasificación LLM", type="primary", use_container_width=True, disabled=len(paper_dirs) == 0):
        try:
            from enzyme_parser import EnzymeParser
        except ImportError:
            st.error("❌ No se pudo importar EnzymeParser.")
            return
        
        log_container = st.empty()
        logs = []
        
        with st.spinner("Clasificando con LLM..."):
            try:
                parser = EnzymeParser(input_dir, output_dir)
                
                for msg in parser.classify_existing_papers_streaming():
                    logs.append(msg)
                    log_container.code("\n".join(logs), language="")
                
                st.success("✅ Clasificación completada!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())


def tab_review():
    """Pestaña de Revisión de Papers Procesados."""
    st.header("📄 Revisar Papers")
    
    # Get output dir from session or use default
    output_dir = st.session_state.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    output_path = Path(output_dir)
    
    if not output_path.exists():
        st.warning(f"No se encuentra el directorio de salida: {output_dir}")
        st.info("💡 Primero ejecuta el parser en la pestaña 'Ejecutar'")
        return

    papers = sorted([d.name for d in output_path.iterdir() if d.is_dir()])
    
    if not papers:
        st.warning("No hay papers procesados en el directorio de salida.")
        return
    
    selected_paper = st.selectbox("Seleccionar Artículo", papers)
    
    if selected_paper:
        st.session_state.selected_paper_id = selected_paper
        data = load_paper_data(output_dir, selected_paper)
        
        if not data:
            st.error("JSON de datos no encontrado.")
            return

        # Group blocks by source file
        grouped_blocks = group_blocks_by_source(data, selected_paper)
        
        # --- STATISTICS BY SOURCE ---
        st.subheader("📊 Estadísticas por Fuente")
        
        stats_cols = st.columns(len(grouped_blocks) + 1)
        
        # Total stats
        all_blocks = merge_and_sort_blocks(data)
        total_relevant = sum(1 for b in all_blocks if b.get("has_quantitative_data"))
        stats_cols[0].metric("📈 Total Relevantes", total_relevant)
        
        # Per-source stats
        for i, (source_name, blocks) in enumerate(grouped_blocks.items(), 1):
            is_main = source_name.startswith("MAIN")
            icon = "📕" if is_main else "📎"
            relevant_count = sum(1 for b in blocks if b.get("has_quantitative_data"))
            short_name = source_name.split(": ", 1)[1] if ": " in source_name else source_name
            # Truncate long names
            if len(short_name) > 20:
                short_name = short_name[:17] + "..."
            stats_cols[i].metric(f"{icon} {short_name}", f"{relevant_count}/{len(blocks)}")
        
        # LLM Classification Summary
        if data.get("llm_classification"):
            with st.expander("🤖 Clasificación LLM", expanded=False):
                st.json(data["llm_classification"])

        st.divider()
        st.caption("📋 Contenido agrupado por archivo fuente. Solo se muestran bloques relevantes.")
        
        pdf_path = data.get("original_pdf")
        
        # Create ordered list of IDs for context navigation (global across all sources)
        all_ordered_ids = [b.get("id") for b in all_blocks]
        
        # Option to clear expanded context
        if st.session_state.get("expanded_ids"):
            if st.button("🔄 Limpiar contexto expandido", help="Ocultar todos los bloques de contexto"):
                st.session_state.expanded_ids = set()
                st.rerun()
        
        # --- RENDER GROUPED SECTIONS ---
        for source_name, blocks in grouped_blocks.items():
            is_main = source_name.startswith("MAIN")
            icon = "📕" if is_main else "📎"
            
            # Count relevant items in this section
            relevant_in_section = sum(1 for b in blocks if b.get("has_quantitative_data"))
            
            # Expander label
            label = f"{icon} {source_name} ({relevant_in_section} relevantes / {len(blocks)} total)"
            
            with st.expander(label, expanded=is_main):
                if relevant_in_section == 0:
                    st.info("No hay bloques relevantes en esta fuente.")
                else:
                    for i, block in enumerate(blocks):
                        render_block(block, i, pdf_path, data, output_dir, all_ordered_ids, source_name=source_name)



# --- MAIN APP ---

def main():
    st.sidebar.title("🧬 EnzyParser UI")
    
    # Navigation - Default to "Ejecutar" (index=0)
    tab = st.sidebar.radio(
        "Navegación",
        ["🚀 Ejecutar", "📄 Revisar"],
        index=0,  # Start on Execute tab
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    st.sidebar.caption("EnzyParser v1.0 - Curated Reader")
    
    if tab == "🚀 Ejecutar":
        tab_execution()
    else:
        tab_review()

if __name__ == "__main__":
    main()
