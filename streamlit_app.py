import streamlit as st
import json
import os
import sys
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
# =============================================================================
# VISOR DE PDF INTERACTIVO
# =============================================================================
# streamlit-pdf-viewer es un componente que permite mostrar PDFs directamente
# en Streamlit con anotaciones (bounding boxes) y callbacks cuando se clica.
# =============================================================================
from streamlit_pdf_viewer import pdf_viewer


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
# FUNCIÓN: highlight_region_on_page
# =============================================================================
# PROPÓSITO: Mostrar la página completa del PDF con un rectángulo rojo
#            resaltando la región donde está el elemento (texto, tabla, figura).
#
# ¿POR QUÉ ES MEJOR QUE SOLO RECORTAR?
# 1. Puedes ver el CONTEXTO - qué hay alrededor del elemento.
# 2. Puedes entender DÓNDE en la página está ubicado el elemento.
# 3. Es más fácil verificar que Docling extrajo la información correcta.
#
# ¿CÓMO FUNCIONA?
# 1. Toma la imagen completa de la página del PDF.
# 2. Calcula dónde dibujar el rectángulo usando las coordenadas (bbox).
# 3. Convierte las coordenadas de Docling al sistema de PIL.
# 4. Dibuja un rectángulo rojo semi-grueso alrededor de la región.
# 5. Retorna la imagen con el rectángulo dibujado.
#
# ARGUMENTOS:
#   - full_page_img: La imagen completa de la página del PDF.
#   - bbox: Lista de 4 números [left, top, right, bottom] de Docling.
#   - color: Color del rectángulo (por defecto rojo).
#   - width: Grosor de la línea del rectángulo (por defecto 4 píxeles).
#
# RETORNA:
#   - La imagen de la página completa con el rectángulo dibujado.
#   - O None si algo falla.
# =============================================================================
def highlight_region_on_page(full_page_img, bbox, color="red", width=4):
    """
    Dibuja un rectángulo resaltando una región en la imagen de la página.
    Convierte coordenadas de Docling (origen inferior-izquierdo) a PIL (origen superior-izquierdo).
    """
    # -------------------------------------------------------------------------
    # Importar ImageDraw para dibujar sobre la imagen.
    # -------------------------------------------------------------------------
    # ImageDraw es parte de PIL/Pillow y permite dibujar formas sobre imágenes.
    # Lo importamos aquí para que el código sea auto-contenido.
    # -------------------------------------------------------------------------
    from PIL import ImageDraw
    
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
    # Paso 3: Hacer una COPIA de la imagen.
    # -------------------------------------------------------------------------
    # Importante: Usamos .copy() para no modificar la imagen original.
    # Si no hacemos copia, la imagen original quedaría con los rectángulos
    # dibujados permanentemente.
    # -------------------------------------------------------------------------
    img_with_highlight = full_page_img.copy()
    
    # -------------------------------------------------------------------------
    # Paso 4: Calcular factores de escala.
    # -------------------------------------------------------------------------
    # Docling asume páginas de 612x792 puntos (tamaño carta a 72 DPI).
    # La imagen renderizada puede tener diferente resolución.
    # -------------------------------------------------------------------------
    page_width_pts = 612.0   # Ancho estándar en puntos
    page_height_pts = 792.0  # Alto estándar en puntos
    
    scale_x = img_with_highlight.width / page_width_pts
    scale_y = img_with_highlight.height / page_height_pts
    
    # -------------------------------------------------------------------------
    # Paso 5: Extraer coordenadas del bbox.
    # -------------------------------------------------------------------------
    l, t, r, b = bbox  # left, top, right, bottom
    
    # -------------------------------------------------------------------------
    # Paso 6: Convertir coordenadas de Docling a PIL.
    # -------------------------------------------------------------------------
    # Docling: Origen en esquina INFERIOR izquierda (Y crece hacia arriba)
    # PIL:     Origen en esquina SUPERIOR izquierda (Y crece hacia abajo)
    #
    # Fórmula de conversión para Y:
    #   y_pil = altura_imagen - (y_docling * escala)
    # -------------------------------------------------------------------------
    
    # Coordenadas X (horizontales) - solo escalar, no invertir
    x1 = l * scale_x
    x2 = r * scale_x
    
    # Coordenadas Y (verticales) - escalar E invertir
    # En Docling, 't' es el borde superior (valor Y más alto)
    # En Docling, 'b' es el borde inferior (valor Y más bajo)
    y1_docling = t * scale_y
    y2_docling = b * scale_y
    
    # Convertir a sistema PIL (invertir Y)
    y1_pil = img_with_highlight.height - y1_docling  # Top → arriba en pantalla
    y2_pil = img_with_highlight.height - y2_docling  # Bottom → abajo en pantalla
    
    # -------------------------------------------------------------------------
    # Paso 7: Ordenar coordenadas correctamente.
    # -------------------------------------------------------------------------
    # PIL espera: (left, top, right, bottom) donde left < right y top < bottom
    # -------------------------------------------------------------------------
    rect_left = min(x1, x2)
    rect_right = max(x1, x2)
    rect_top = min(y1_pil, y2_pil)      # El menor valor Y está arriba
    rect_bottom = max(y1_pil, y2_pil)   # El mayor valor Y está abajo
    
    # -------------------------------------------------------------------------
    # Paso 8: Asegurar que las coordenadas están dentro de la imagen.
    # -------------------------------------------------------------------------
    rect_left = max(0, rect_left)
    rect_top = max(0, rect_top)
    rect_right = min(img_with_highlight.width, rect_right)
    rect_bottom = min(img_with_highlight.height, rect_bottom)
    
    # -------------------------------------------------------------------------
    # Paso 9: Verificar que el rectángulo tiene área válida.
    # -------------------------------------------------------------------------
    if rect_right <= rect_left or rect_bottom <= rect_top:
        return img_with_highlight  # Retornar imagen sin rectángulo
    
    # -------------------------------------------------------------------------
    # Paso 10: Crear objeto para dibujar sobre la imagen.
    # -------------------------------------------------------------------------
    # ImageDraw.Draw() crea un "pincel" que puede dibujar sobre la imagen.
    # -------------------------------------------------------------------------
    draw = ImageDraw.Draw(img_with_highlight)
    
    # -------------------------------------------------------------------------
    # Paso 11: Dibujar el rectángulo.
    # -------------------------------------------------------------------------
    # .rectangle() dibuja un rectángulo.
    # - xy: Las coordenadas [(x1, y1), (x2, y2)] de las esquinas opuestas.
    # - outline: El color del borde (rojo por defecto).
    # - width: El grosor de la línea.
    # -------------------------------------------------------------------------
    draw.rectangle(
        [(rect_left, rect_top), (rect_right, rect_bottom)],
        outline=color,
        width=width
    )
    
    # -------------------------------------------------------------------------
    # Paso 12: Retornar la imagen con el rectángulo dibujado.
    # -------------------------------------------------------------------------
    return img_with_highlight


# =============================================================================
# FUNCIÓN: docling_bbox_to_pdf_annotation
# =============================================================================
# PROPÓSITO: Convertir las coordenadas de Docling al formato que usa
#            streamlit-pdf-viewer para dibujar anotaciones sobre el PDF.
#
# ¿POR QUÉ ES NECESARIA?
# Docling y streamlit-pdf-viewer usan sistemas de coordenadas diferentes:
#
# DOCLING:
#   - Origen en esquina INFERIOR IZQUIERDA
#   - bbox = [left, top, right, bottom] donde top > bottom
#   - Unidad: puntos PDF (72 por pulgada)
#
# STREAMLIT-PDF-VIEWER:
#   - Origen en esquina SUPERIOR IZQUIERDA (como PDF.js)
#   - Formato: {page, x, y, width, height, color}
#   - x, y es la esquina SUPERIOR IZQUIERDA de la anotación
#
# ARGUMENTOS:
#   - block: Diccionario con información del bloque (tiene bbox, page_no, id)
#   - page_height: Altura de la página en puntos (792 para carta estándar)
#   - color: Color del rectángulo de anotación
#
# RETORNA:
#   - Diccionario en el formato de streamlit-pdf-viewer, o None si no hay bbox
# =============================================================================
def docling_bbox_to_pdf_annotation(block, page_height=792, color="red"):
    """
    Convierte bbox de Docling [L, T, R, B] al formato de streamlit-pdf-viewer.
    """
    # -------------------------------------------------------------------------
    # Paso 1: Verificar que el bloque tiene las propiedades necesarias.
    # -------------------------------------------------------------------------
    bbox = block.get("bbox")
    page_no = block.get("page_no")
    block_id = block.get("id", "unknown")
    
    if not bbox or not page_no:
        return None
    
    if len(bbox) != 4:
        return None
    
    # -------------------------------------------------------------------------
    # Paso 2: Extraer coordenadas del bbox.
    # -------------------------------------------------------------------------
    # En Docling: l=left, t=top (Y alto), r=right, b=bottom (Y bajo)
    # -------------------------------------------------------------------------
    l, t, r, b = bbox
    
    # -------------------------------------------------------------------------
    # Paso 3: Calcular dimensiones del rectángulo.
    # -------------------------------------------------------------------------
    # Ancho = derecha - izquierda
    # Alto = top - bottom (porque en Docling t > b)
    # -------------------------------------------------------------------------
    width = abs(r - l)
    height = abs(t - b)
    
    # -------------------------------------------------------------------------
    # Paso 4: Convertir la coordenada Y al sistema de PDF.js.
    # -------------------------------------------------------------------------
    # En Docling, 't' es la posición Y del borde SUPERIOR, medida desde abajo.
    # En PDF.js, 'y' es la posición Y del borde SUPERIOR, medida desde arriba.
    #
    # Fórmula: y_pdfjs = page_height - t_docling
    #
    # Pero streamlit-pdf-viewer espera la esquina superior izquierda,
    # que es donde empieza el elemento.
    # -------------------------------------------------------------------------
    y_pdfjs = page_height - t  # Convertir Y de Docling a PDF.js
    
    # -------------------------------------------------------------------------
    # Paso 5: Construir el diccionario de anotación.
    # -------------------------------------------------------------------------
    return {
        "page": page_no,        # Número de página (1-indexed)
        "x": l,                  # Coordenada X (izquierda)
        "y": y_pdfjs,           # Coordenada Y (desde arriba de la página)
        "width": width,          # Ancho del rectángulo
        "height": height,        # Alto del rectángulo
        "color": color,          # Color del borde
        "id": block_id           # ID para identificar el bloque al hacer clic
    }

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
    is_selected = block_id == st.session_state.get("selected_block_id")
    
    if not is_important and not is_expanded_context and not is_selected:
        return

    # Si está seleccionado, usamos un contenedor visualmente distinto
    if is_selected:
        st.markdown(f"<div id='block-{block_id}'></div>", unsafe_allow_html=True) # Anchor (no funciona nativo pero es buena práctica)
        st.error(f"🔴 **BLOQUE SELECCIONADO** (ID: {block_id})")
        
    container_border = True
    
    with st.container(border=container_border):
        c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
        
        type_label = block.get('type', block['block_type']).upper()
        page_label = f"Pág {block.get('page_no', '-')}"
        
        with c1:
            if is_selected:
                st.markdown(f"**:red[🔴 SELECCIONADO]** | `{type_label}` | {page_label}")
            elif is_important:
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
            # -------------------------------------------------------------
            # Botón "Ir al PDF": Navega al PDF viewer en la posición correcta.
            # -------------------------------------------------------------
            # Solo mostramos el botón si el bloque tiene página y es de un PDF.
            # (No para Excel, CSV, etc.)
            # -------------------------------------------------------------
            source_file = block.get("source_file") or block.get("source") or ""
            page_no = block.get("page_no")
            is_pdf_source = source_file.lower().endswith(".pdf") or not source_file
            
            if page_no and is_pdf_source and block["block_type"] != "artifact":
                if st.button("📍 Ir al PDF", key=f"gopdf_{source_name}_{idx}_{block_id}"):
                    st.session_state.selected_block_id = block_id
                    st.session_state.selected_page = page_no
                    st.session_state.selected_block_source = Path(source_file).name if source_file else None
                    st.rerun()

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
            
            # Mostrar la imagen de la tabla si existe (extraída por Docling)
            table_img_path = block.get("image_path")
            if table_img_path:
                abs_path = Path(output_dir) / st.session_state.selected_paper_id / "artifacts" / Path(table_img_path).name
                if not abs_path.exists():
                    abs_path = Path(table_img_path)
                
                if abs_path.exists():
                    st.image(str(abs_path), caption="Recorte Exacto de Tabla", use_container_width=True)
            
            # Mostrar Mistral Markdown si existe (Editable)
            mistral_md = block.get("markdown_content")
            if mistral_md or block.get("has_quantitative_data"): # Show editor if available or if marked as relevant
                with st.expander("🧠 Mistral OCR (Enriquecido / Editar)", expanded=True):
                    # Editable Text Area
                    new_md = st.text_area(
                        "Contenido Markdown:",
                        value=mistral_md if mistral_md else "",
                        height=300,
                        key=f"md_edit_{source_name}_{idx}_{block_id}"
                    )
                    
                    if new_md != mistral_md:
                        if st.button("💾 Guardar Edición", key=f"save_md_{source_name}_{idx}_{block_id}"):
                            block["markdown_content"] = new_md
                            save_paper_data(output_dir, st.session_state.selected_paper_id, paper_data)
                            st.success("Guardado!")
                            st.rerun()
            
            # Mostrar datos OCR extraídos (Docling)
            with st.expander("📊 Ver Datos Extraídos (Docling)", expanded=False):
                if "data" in block and block["data"]:
                    df = pd.DataFrame(block["data"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No se pudo extraer texto estructurado de esta tabla.")
            
            if notes: 
                st.caption(f"📝 *Notas:* {notes}")

        # CASO 3: TEXTO
        elif block["block_type"] == "text":
            # Mostrar el texto extraído directamente
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
    
    st.divider()

    # --- STAGE 3: Mistral Enrichment ---
    st.markdown("### 🧪 Etapa 3: Enriquecimiento OCR (Mistral)")
    st.caption("Procesa tablas marcadas como 'Positivas' con Mistral OCR para obtener Markdown de alta fidelidad.")

    if st.button("▶️ Enriquecer Tablas con Mistral OCR", type="secondary", use_container_width=True, disabled=len(paper_dirs) == 0):
        try:
            from enzyme_parser import EnzymeParser
        except ImportError:
            st.error("❌ No se pudo importar EnzymeParser.")
            return

        log_container = st.empty()
        logs = []

        with st.spinner("Enriqueciendo tablas..."):
            try:
                parser = EnzymeParser(input_dir, output_dir)
                for msg in parser.enrich_existing_papers_streaming():
                    logs.append(msg)
                    log_container.code("\n".join(logs), language="")
                
                st.success("✨ Enriquecimiento completado!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())


@st.cache_data(show_spinner=False)
def get_cached_pdf_images(pdf_path):
    """Convierte un PDF a una lista de imágenes PIL (cacheado)."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=100) # 100 DPI es suficiente para pantalla y más rápido
    except Exception as e:
        return []

def tab_review():
    """
    ==========================================================================
    PESTAÑA DE REVISIÓN - NUEVO LAYOUT (PDF IZQUIERDA / TOOLS DERECHA)
    ==========================================================================
    MODIFICACIÓN SOLICITADA:
    Se ha invertido el layout original para coincidir con el diseño solicitado.
    
    COLUMNA IZQUIERDA (PDF):
        - Visor continuo o paginado
        - Controles de navegación estilo "Clean"
    
    COLUMNA DERECHA (TOOLS):
        - Pestañas: Parse | Split | Extract | Chat
        - "Parse" contiene la lista de bloques extraídos (lógica original)
    
    NOTA: Se mantienen todas las llamadas al backend y la lógica de datos.
    Solo se cambia la estructura visual (st.columns, st.tabs).
    ==========================================================================
    """
    st.header("📄 Revisar Papers")
    
    # -------------------------------------------------------------------------
    # Paso 1: Obtener el directorio de salida.
    # -------------------------------------------------------------------------
    # (Sin cambios en la lógica de backend)
    # -------------------------------------------------------------------------
    output_dir = st.session_state.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    output_path = Path(output_dir)
    
    if not output_path.exists():
        st.warning(f"No se encuentra el directorio de salida: {output_dir}")
        st.info("💡 Primero ejecuta el parser en la pestaña 'Ejecutar'")
        return

    # -------------------------------------------------------------------------
    # Paso 2: Listar papers disponibles.
    # -------------------------------------------------------------------------
    papers = sorted([d.name for d in output_path.iterdir() if d.is_dir()])
    
    if not papers:
        st.warning("No hay papers procesados en el directorio de salida.")
        return
    
    # -------------------------------------------------------------------------
    # Paso 3: Selector de paper.
    # -------------------------------------------------------------------------
    selected_paper = st.selectbox("Seleccionar Artículo", papers)
    
    if not selected_paper:
        return
        
    st.session_state.selected_paper_id = selected_paper
    data = load_paper_data(output_dir, selected_paper)
    
    if not data:
        st.error("JSON de datos no encontrado.")
        return

    # -------------------------------------------------------------------------
    # Paso 4: Preparar datos.
    # -------------------------------------------------------------------------
    grouped_blocks = group_blocks_by_source(data, selected_paper)
    all_blocks = merge_and_sort_blocks(data)
    total_relevant = sum(1 for b in all_blocks if b.get("has_quantitative_data"))
    
    # -------------------------------------------------------------------------
    # Paso 5: Estadísticas rápidas.
    # -------------------------------------------------------------------------
    # st.metric("📈 Total Bloques Relevantes", total_relevant) 
    # (Comentado para limpiar la UI y que se parezca más al diseño "clean")
    
    # st.divider() # Quitamos divider para limpiar
    
    # -------------------------------------------------------------------------
    # Paso 5a: Selector Global de Fuente (Sincronizado)
    # -------------------------------------------------------------------------
    source_options = list(grouped_blocks.keys())
    
    # Manejar sincronización desde botón "Ir al PDF"
    if st.session_state.get("selected_block_source"):
        target_src = st.session_state.selected_block_source
        for opt in source_options:
            if target_src in opt:
                st.session_state.global_source_selector = opt
                st.session_state.selected_block_source = None
                break
    
    if "global_source_selector" not in st.session_state:
        st.session_state.global_source_selector = source_options[0]
        
    # -------------------------------------------------------------------------
    # CONTROLES SUPERIORES (BARRA DE HERRAMIENTAS)
    # -------------------------------------------------------------------------
    # Organizados en columnas para compactar la UI.
    # -------------------------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([0.4, 0.3, 0.3], gap="small", vertical_alignment="bottom")
    
    with ctrl_col1:
        selected_source = st.selectbox(
            "📂 Documento", 
            source_options,
            key="global_source_selector",
            label_visibility="collapsed" # Más limpio
        )
    
    with ctrl_col2:
        # Toggle para modo de vista
        view_mode = st.radio(
            "Visualización",
            ["📄 Paginada", "📜 Continua"],
            index=0, 
            horizontal=True,
            key="pdf_view_mode",
            label_visibility="collapsed"
        )
        
    with ctrl_col3:
        # Ajuste de ancho de columnas (Invertido ahora: Izq PDF, Der Tools)
        # El valor representa el ancho del PDF.
        split_ratio = st.slider(
            "↔️ Ancho PDF", 
            min_value=0.2, 
            max_value=0.8, 
            value=0.6,  # Empezamos con el PDF más ancho (60%)
            step=0.05,
            label_visibility="collapsed"
        )
    
    st.markdown("---") # Separador sutil
    
    # FILTRADO GLOBAL DE BLOQUES
    grouped_blocks = {selected_source: grouped_blocks[selected_source]}
    
    # -------------------------------------------------------------------------
    # Paso 6: NUEVO LAYOUT SIDE-BY-SIDE
    # -------------------------------------------------------------------------
    # CAMBIO IMPORTANTE:
    # col_pdf (Izquierda) | col_tools (Derecha)
    # -------------------------------------------------------------------------
    col_pdf, col_tools = st.columns([split_ratio, 1 - split_ratio])
    
    # =========================================================================
    # COLUMNA IZQUIERDA: Visor de PDF Original
    # =========================================================================
    with col_pdf:
        # Estilo "Card" para el contenedor del PDF
        with st.container(border=True):
            
            # Obtener la única fuente seleccionada
            if not grouped_blocks:
                st.warning("No hay datos para esta fuente.")
            else:
                current_source_name, current_blocks = list(grouped_blocks.items())[0]
                
                # Resolver ruta al PDF
                current_pdf_path = None
                is_main = current_source_name.startswith("MAIN")
                
                if is_main:
                     main_pdf = resolve_source_pdf_path(
                        {"source_file": f"{selected_paper}.pdf"},
                        selected_paper,
                        DEFAULT_INPUT_DIR
                    )
                     if main_pdf:
                         current_pdf_path = main_pdf
                else:
                     for block in current_blocks:
                          source_file = block.get("source_file") or block.get("source")
                          if source_file and source_file.lower().endswith(".pdf"):
                               current_pdf_path = resolve_source_pdf_path(block, selected_paper, DEFAULT_INPUT_DIR)
                               break
                
                if not current_pdf_path:
                     st.info(f"ℹ️ El archivo '{current_source_name}' no es un PDF visualizable.")
                else:
                    # Título sutil
                    st.caption(f"Visualizando: {Path(current_pdf_path).name}")
                    
                    try:
                        from streamlit_image_coordinates import streamlit_image_coordinates
                        from PIL import ImageDraw
                        
                        # Carga de imágenes (Cacheada)
                        with st.spinner("Cargando PDF..."):
                            pages = get_cached_pdf_images(current_pdf_path)
                        
                        if not pages:
                            st.error("No se pudo leer el archivo PDF.")
                        else:
                            is_continuous = "Continua" in st.session_state.get("pdf_view_mode", "")
                            pages_to_render = [] 
                            
                            # ---------------------------------------------------------
                            # CONTROLES DE NAVEGACIÓN COMPACTOS (Solo Paginado)
                            # ---------------------------------------------------------
                            if not is_continuous:
                                total_pages = len(pages)
                                current_page = st.session_state.get("selected_page", 1)
                                if current_page < 1: current_page = 1
                                if current_page > total_pages: current_page = total_pages
                                st.session_state.selected_page = current_page
                                
                                # Barra de navegación centrada y minimalista
                                nc1, nc2, nc3, nc4, nc5 = st.columns([2, 1, 2, 1, 2])
                                with nc2:
                                    if st.button("❮", disabled=current_page <= 1, key="nav_prev", use_container_width=True):
                                        st.session_state.selected_page -= 1
                                        st.rerun()
                                with nc3:
                                    st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 5px'>{current_page} / {total_pages}</div>", unsafe_allow_html=True)
                                with nc4:
                                    if st.button("❯", disabled=current_page >= total_pages, key="nav_next", use_container_width=True):
                                        st.session_state.selected_page += 1
                                        st.rerun()
                                        
                                if total_pages > 0:
                                    pages_to_render.append((pages[current_page-1], current_page))
                            else:
                                # Modo continuo: Mostrar todas
                                for i, p in enumerate(pages):
                                    pages_to_render.append((p, i + 1))

                            # ---------------------------------------------------------
                            # RENDERIZADO DE PÁGINAS E INTERACCIÓN
                            # ---------------------------------------------------------
                            # Usamos un scroll container para el PDF
                            with st.container(height=800):
                                for page_img_orig, current_page_num in pages_to_render:
                                    if is_continuous:
                                        st.caption(f"Página {current_page_num}")
                                    
                                    # Preparar imagen y overlay
                                    page_img = page_img_orig.copy()
                                    img_width, img_height = page_img.size
                                    draw = ImageDraw.Draw(page_img)
                                    scale_factor = 100 / 72  
                                    pdf_page_height = img_height / scale_factor

                                    # Filtrar bloques de esta página
                                    block_coords = []
                                    for i, block in enumerate(current_blocks): # Iteramos sobre bloques FILTRADOS por fuente
                                        if (block.get("has_quantitative_data") and 
                                            block.get("bbox") and 
                                            block.get("page_no") == current_page_num):
                                            
                                            bbox = block.get("bbox")
                                            block_id = block.get("id", str(i))
                                            l, t, r, b = bbox
                                            
                                            x1 = int(l * scale_factor)
                                            y1 = int((pdf_page_height - t) * scale_factor)
                                            x2 = int(r * scale_factor)
                                            y2 = int((pdf_page_height - b) * scale_factor)
                                            
                                            block_coords.append((x1, y1, x2, y2, block_id))
                                            
                                            is_selected = block_id == st.session_state.get("selected_block_id")
                                            color = "#00FF00" if is_selected else "#FF0000"
                                            width = 4 if is_selected else 2
                                            
                                            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                                            
                                            # Label pequeño
                                            # draw.text(...) # Podríamos poner label si se desea

                                    # Render Interactivo
                                    clicked = streamlit_image_coordinates(
                                        page_img,
                                        key=f"pdf_page_{current_page_num}_{st.session_state.get('global_source_selector','default')}_mode_{is_continuous}"
                                    )
                                    
                                    if clicked is not None:
                                        cx, cy = clicked["x"], clicked["y"]
                                        for (bx1, by1, bx2, by2, bid) in block_coords:
                                            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                                                if bid != st.session_state.get("selected_block_id"):
                                                    st.session_state.selected_block_id = bid
                                                    st.rerun()
                                                break

                    except Exception as e:
                        st.error(f"Error renderizando PDF: {e}")

    # =========================================================================
    # COLUMNA DERECHA: Herramientas y Datos (Tabs)
    # =========================================================================
    with col_tools:
        # Implementación de Pestañas como en el diseño
        tabs = st.tabs(["✨ Parse", "✂️ Split", "🧪 Extract", "💬 Chat"])
        
        # --- TAB: PARSE (Lógica original de visualización de datos) ---
        with tabs[0]: 
            st.markdown("##### Extracción Estructurada") # Título menor
            
            # Selector Markdown | JSON (simulado visualmente o funcional)
            view_type = st.radio("Formato:", ["Markdown", "JSON"], horizontal=True, label_visibility="collapsed")
            
            if view_type == "JSON":
                 st.json(data, expanded=False)
            else:
                # LISTA DE BLOQUES (Scrollable)
                with st.container(height=750):
                    
                    # Inicializar estado selected
                    if "selected_block_id" not in st.session_state:
                         st.session_state.selected_block_id = None
                    
                    pdf_path = data.get("original_pdf")
                    all_ordered_ids = [b.get("id") for b in all_blocks]
                    
                    # Botón para limpiar selección
                    if st.session_state.get("selected_block_id"):
                        if st.button("Desmarcar Selección", type="secondary"):
                             st.session_state.selected_block_id = None
                             st.rerun()

                    # Iterar y mostrar bloques
                    # Nota: grouped_blocks AQUÍ todavía tiene solo la fuente seleccionada
                    # porque la filtramos arriba. Si queremos mostrar TODO el contenido
                    # independientemente del PDF que se ve, deberíamos haber guardado
                    # una copia de 'grouped_blocks' original.
                    # PERO: El diseño sugiere que el panel derecho muestra lo relacionado
                    # con el documento. Vamos a mantener la consistencia:
                    # Mostrar bloques de la fuente activa.
                    
                    for source_name, blocks in grouped_blocks.items():
                         # Renderizar cada bloque usando la función helper existente
                         for i, block in enumerate(blocks):
                                render_block(block, i, pdf_path, data, output_dir, all_ordered_ids, source_name=source_name)

        # --- TABS: PLACEHOLDERS ---
        with tabs[1]: # Split
             st.info("🚧 **Splitter**\n\nHerramienta para dividir documentos grandes en secciones lógicas.\n*(Próximamente)*")
             
        with tabs[2]: # Extract
             st.info("🚧 **Extractor**\n\nDefinición de esquemas de extracción personalizados y validación.\n*(Próximamente)*")
             
        with tabs[3]: # Chat
             st.info("🚧 **Chat con tu Data**\n\nInterfaz conversacional para interrogar al documento.\n*(Próximamente)*")






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
