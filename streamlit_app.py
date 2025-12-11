import streamlit as st
import json
import os
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd

# Configuración de la página
st.set_page_config(layout="wide", page_title="EnzyParser Distilled Reader")

# Rutas (Ajusta según tu estructura)
OUTPUT_DIR = Path("./Processed_Enzyme_Data")

# --- FUNCIONES DE UTILIDAD ---

def load_paper_data(paper_id):
    """Carga el JSON y prepara los datos en memoria."""
    json_path = OUTPUT_DIR / paper_id / f"{paper_id}_data.json"
    if not json_path.exists():
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_paper_data(paper_id, data):
    """Guarda los cambios (reclasificaciones) en el disco."""
    json_path = OUTPUT_DIR / paper_id / f"{paper_id}_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    st.toast(f"Cambios guardados para {paper_id}", icon="💾")

def get_pdf_page_image(pdf_path, page_num):
    """Renderiza una página específica del PDF a imagen (cacheada)."""
    # Nota: pdf2image usa base 1 para las páginas, igual que Docling
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    except Exception as e:
        st.error(f"Error renderizando PDF: {e}")
        return None

def crop_image(full_page_img, bbox):
    """Recorta la imagen usando bbox [L, T, R, B] de Docling (72 dpi base)."""
    # Docling suele usar coordenadas en puntos (72 dpi). 
    # pdf2image por defecto renderiza a mayor dpi (usualmente 200 o 300).
    # Necesitamos escalar el bbox.
    
    if not full_page_img: return None
    
    # Asumimos que Docling reporta bbox normalizado o en puntos de PDF standard
    # Escalamos proporcionalmente al tamaño real de la imagen renderizada
    # Docling PDF backend usa coordenadas relativas al tamaño de página original
    # Simplificación: usaremos porcentajes relativos si es posible, o escala directa.
    
    # Factor de escala simple (Ajustar según necesidad, Docling suele ser preciso en puntos)
    # Aquí asumimos que bbox viene en puntos y la imagen es high-res.
    # Un PDF letter es ~612x792 puntos.
    
    scale_x = full_page_img.width / 612.0 # Aproximación standard, ideal leer metadata de tamaño
    scale_y = full_page_img.height / 792.0
    
    l, t, r, b = bbox
    return full_page_img.crop((l * scale_x, t * scale_y, r * scale_x, b * scale_y))

# --- LÓGICA DEL FEED ---

def merge_and_sort_blocks(data):
    """
    Fusiona textos y artefactos en una sola lista cronológica.
    Esto es vital para mantener la 'cohesión' que pediste.
    """
    blocks = []
    
    # 1. Text Segments
    for seg in data.get("text_segments", []):
        seg["block_type"] = "text"
        blocks.append(seg)
        
    # 2. Artifacts (Figuras)
    for art in data.get("artifacts", []):
        art["block_type"] = "artifact"
        blocks.append(art)
        
    # Ordenar por: Página -> Posición Vertical (Top) -> Posición Horizontal (Left)
    # bbox suele ser [l, t, r, b]
    blocks.sort(key=lambda x: (x.get("page_no", 0), x.get("bbox", [0,0,0,0])[1], x.get("bbox", [0,0,0,0])[0]))
    
    return blocks

# --- COMPONENTES DE UI ---

def render_block(block, idx, pdf_path_str, paper_data):
    """Renderiza una tarjeta individual (Bloque)."""
    
    # Estado local del bloque (expandido/visualización)
    block_id = block.get("id")
    is_important = block.get("has_quantitative_data", False)
    
    # Si NO es importante y NO está expandido manualmente, no lo mostramos
    # (A menos que el usuario esté en modo "Ver Todo")
    is_expanded_context = st.session_state.get(f"expand_{block_id}", False)
    
    if not is_important and not is_expanded_context:
        return # Skip render

    # Estilo del borde: Verde para Data, Gris para Contexto
    border_color = "green" if is_important else "grey"
    
    # Contenedor de la Tarjeta
    with st.container(border=True):
        
        # 1. Cabecera de la Tarjeta (Tipo y Acciones)
        c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
        
        with c1:
            if is_important:
                st.markdown(f"**:sparkles: RELEVANTE** | `{block['type']}` | Pág {block['page_no']}")
            else:
                st.markdown(f"*:grey[Contexto Expandido]* | `{block['type']}` | Pág {block['page_no']}")

        # Botón de Reclasificación (Feedback Loop)
        with c2:
            if not is_important:
                if st.button("📌 Marcar Útil", key=f"mark_{block_id}"):
                    block["has_quantitative_data"] = True
                    save_paper_data(st.session_state.selected_paper_id, paper_data)
                    st.rerun()
            else:
                if st.button("🗑️ Descartar", key=f"unmark_{block_id}"):
                    block["has_quantitative_data"] = False
                    save_paper_data(st.session_state.selected_paper_id, paper_data)
                    st.rerun()

        # Toggle Vista (Híbrida)
        with c3:
            view_mode = st.toggle("Ver Original", key=f"view_{block_id}")

        # 2. Contenido Principal
        
        # CASO A: ARTEFACTOS VISUALES (Imágenes ya extraídas)
        if block["block_type"] == "artifact":
            # Usamos la imagen pre-guardada por Docling
            # Ajustar ruta relativa a absoluta para Streamlit
            rel_path = block.get("file_path")
            # Truco para arreglar rutas si corres streamlit desde otro dir
            abs_path = Path(os.getcwd()) / rel_path 
            
            if abs_path.exists():
                st.image(str(abs_path), caption=block.get("caption", "Figura"), use_container_width=True)
                if block.get("vlm_description"):
                    with st.expander("Análisis IA"):
                        st.write(block["vlm_description"])
            else:
                st.error(f"Imagen no encontrada: {rel_path}")

        # CASO B: SEGMENTOS DE TEXTO
        elif block["block_type"] == "text":
            
            if view_mode:
                # MODO VISUAL: Renderizar crop del PDF
                if pdf_path_str:
                    # Cachear la página entera para no recargarla por cada bloque
                    page_img = get_pdf_page_image(pdf_path_str, block["page_no"])
                    if page_img:
                        crop = crop_image(page_img, block["bbox"])
                        st.image(crop, caption="Recorte original del PDF")
                else:
                    st.warning("PDF original no encontrado para generar vista.")
            else:
                # MODO TEXTO: Markdown limpio
                # Si es contexto, usamos letra gris
                if is_important:
                    st.markdown(block["text"])
                else:
                    st.markdown(f"<span style='color:grey'>{block['text']}</span>", unsafe_allow_html=True)

        # 3. Botones de Contexto (Expandir vecinos)
        # Lógica simple: expandir el ID anterior y siguiente en la lista global
        # (Esto requiere saber el índice, que pasamos como argumento)
        # Implementación simplificada: marcamos los IDs vecinos para expandirse
        
        sc1, sc2 = st.columns(2)
        if sc1.button("⬆️ Contexto Previo", key=f"prev_{block_id}", help="Mostrar bloque anterior"):
            # En una implementación real, buscaríamos el ID del bloque idx-1
            # y pondríamos st.session_state[f"expand_{prev_id}"] = True
            st.toast("Funcionalidad de contexto en desarrollo (requiere mapeo de índices)")
            
        if sc2.button("⬇️ Contexto Posterior", key=f"next_{block_id}"):
            st.toast("Funcionalidad de contexto en desarrollo")


# --- MAIN APP ---

def main():
    st.sidebar.title("🧬 EnzyParser UI")
    
    # 1. Selector de Paper
    if not OUTPUT_DIR.exists():
        st.error(f"No se encuentra el directorio {OUTPUT_DIR}")
        return

    papers = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    selected_paper = st.sidebar.selectbox("Seleccionar Artículo", papers)
    
    if selected_paper:
        st.session_state.selected_paper_id = selected_paper
        data = load_paper_data(selected_paper)
        
        if not data:
            st.error("JSON de datos no encontrado.")
            return

        # Sidebar Stats
        n_segments = len(data.get("text_segments", []))
        n_relevant = sum(1 for s in data["text_segments"] if s.get("has_quantitative_data"))
        st.sidebar.metric("Segmentos Totales", n_segments)
        st.sidebar.metric("Datos Relevantes", n_relevant)
        st.sidebar.metric("Figuras", len(data.get("artifacts", [])))

        # 2. Main Feed
        st.title(f"📄 {selected_paper}")
        st.caption("Modo de Revisión Destilada: Solo se muestra información clasificada como cuantitativa/experimental.")
        
        # Preparar bloques
        all_blocks = merge_and_sort_blocks(data)
        
        # PDF Path (para crops on the fly)
        # Asumimos que la ruta en el JSON es correcta o relativa
        pdf_path = data.get("original_pdf")
        
        # Render Loop
        # Usamos un iterador para manejar el contexto en el futuro
        for i, block in enumerate(all_blocks):
            render_block(block, i, pdf_path, data)

if __name__ == "__main__":
    main()
