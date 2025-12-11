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
DEFAULT_INPUT_DIR = Path("./NEWarticles")
DEFAULT_OUTPUT_DIR = Path("./Processed_Enzyme_Data")

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

def crop_image(full_page_img, bbox):
    """Recorta la imagen usando bbox [L, T, R, B] de Docling (72 dpi base)."""
    if not full_page_img: return None
    
    scale_x = full_page_img.width / 612.0
    scale_y = full_page_img.height / 792.0
    
    l, t, r, b = bbox
    return full_page_img.crop((l * scale_x, t * scale_y, r * scale_x, b * scale_y))

# --- LÓGICA DEL FEED ---

def merge_and_sort_blocks(data):
    """Fusiona textos y artefactos en una sola lista cronológica."""
    blocks = []
    
    for seg in data.get("text_segments", []):
        seg["block_type"] = "text"
        blocks.append(seg)
        
    for art in data.get("artifacts", []):
        art["block_type"] = "artifact"
        blocks.append(art)
        
    blocks.sort(key=lambda x: (x.get("page_no", 0), x.get("bbox", [0,0,0,0])[1], x.get("bbox", [0,0,0,0])[0]))
    
    return blocks

# --- COMPONENTES DE UI ---

def render_block(block, idx, pdf_path_str, paper_data, output_dir, ordered_ids):
    """Renderiza una tarjeta individual (Bloque)."""
    
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
        
        with c1:
            if is_important:
                st.markdown(f"**:sparkles: RELEVANTE** | `{block['type']}` | Pág {block['page_no']}")
            else:
                st.markdown(f"*:grey[Contexto Expandido]* | `{block['type']}` | Pág {block['page_no']}")

        with c2:
            if not is_important:
                if st.button("📌 Marcar Útil", key=f"mark_{block_id}"):
                    block["has_quantitative_data"] = True
                    save_paper_data(output_dir, st.session_state.selected_paper_id, paper_data)
                    st.rerun()
            else:
                if st.button("🗑️ Descartar", key=f"unmark_{block_id}"):
                    block["has_quantitative_data"] = False
                    save_paper_data(output_dir, st.session_state.selected_paper_id, paper_data)
                    st.rerun()

        with c3:
            view_mode = st.toggle("Ver Original", key=f"view_{block_id}")

        if block["block_type"] == "artifact":
            rel_path = block.get("file_path")
            abs_path = Path(os.getcwd()) / rel_path 
            
            if abs_path.exists():
                st.image(str(abs_path), caption=block.get("caption", "Figura"), use_container_width=True)
                if block.get("vlm_description"):
                    with st.expander("Análisis IA"):
                        st.write(block["vlm_description"])
            else:
                st.error(f"Imagen no encontrada: {rel_path}")

        elif block["block_type"] == "text":
            if view_mode:
                if pdf_path_str:
                    page_img = get_pdf_page_image(pdf_path_str, block["page_no"])
                    if page_img:
                        crop = crop_image(page_img, block["bbox"])
                        st.image(crop, caption="Recorte original del PDF")
                else:
                    st.warning("PDF original no encontrado para generar vista.")
            else:
                if is_important:
                    st.markdown(block["text"])
                else:
                    st.markdown(f"<span style='color:grey'>{block['text']}</span>", unsafe_allow_html=True)

        # Context expansion buttons - now functional!
        sc1, sc2 = st.columns(2)
        
        # Find current index in ordered list
        try:
            current_idx = ordered_ids.index(block_id)
        except ValueError:
            current_idx = idx
        
        # Previous context button
        if current_idx > 0:
            prev_id = ordered_ids[current_idx - 1]
            if sc1.button("⬆️ Contexto Previo", key=f"prev_{block_id}", help="Mostrar bloque anterior"):
                st.session_state.expanded_ids.add(prev_id)
                st.rerun()
        else:
            sc1.button("⬆️ Contexto Previo", key=f"prev_{block_id}", disabled=True, help="No hay bloque anterior")
        
        # Next context button
        if current_idx < len(ordered_ids) - 1:
            next_id = ordered_ids[current_idx + 1]
            if sc2.button("⬇️ Contexto Posterior", key=f"next_{block_id}", help="Mostrar bloque siguiente"):
                st.session_state.expanded_ids.add(next_id)
                st.rerun()
        else:
            sc2.button("⬇️ Contexto Posterior", key=f"next_{block_id}", disabled=True, help="No hay bloque siguiente")


# --- GUI UTILS ---

import tkinter as tk
from tkinter import filedialog

def select_folder_native():
    """Abre un diálogo nativo del sistema para seleccionar carpeta."""
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
    
    # Run Button
    if st.button("▶️ Iniciar Procesamiento", type="primary", use_container_width=True):
        
        # Import parser here to avoid circular imports
        try:
            from enzyme_parser import EnzymeParser
        except ImportError:
            st.error("❌ No se pudo importar EnzymeParser. Asegúrate de que enzyme_parser.py está en el mismo directorio.")
            return
        
        # Create log container
        log_container = st.empty()
        logs = []
        
        with st.spinner("Procesando..."):
            try:
                parser = EnzymeParser(input_dir, output_dir)
                
                # Stream logs
                for msg in parser.process_all_streaming():
                    logs.append(msg)
                    # Update display with all logs
                    log_container.code("\n".join(logs), language="")
                    
                st.success("✅ Procesamiento completado!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {e}")
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

        # Stats
        col1, col2, col3 = st.columns(3)
        n_segments = len(data.get("text_segments", []))
        n_relevant = sum(1 for s in data["text_segments"] if s.get("has_quantitative_data"))
        col1.metric("Segmentos Totales", n_segments)
        col2.metric("Datos Relevantes", n_relevant)
        col3.metric("Figuras", len(data.get("artifacts", [])))
        
        # LLM Classification Summary
        if data.get("llm_classification"):
            with st.expander("🤖 Clasificación LLM", expanded=False):
                st.json(data["llm_classification"])

        st.divider()
        st.caption("Modo de Revisión Destilada: Solo se muestra información clasificada como cuantitativa/experimental.")
        
        all_blocks = merge_and_sort_blocks(data)
        pdf_path = data.get("original_pdf")
        
        # Create ordered list of IDs for context navigation
        ordered_ids = [b.get("id") for b in all_blocks]
        
        # Option to clear expanded context
        if st.session_state.get("expanded_ids"):
            if st.button("🔄 Limpiar contexto expandido", help="Ocultar todos los bloques de contexto"):
                st.session_state.expanded_ids = set()
                st.rerun()
        
        for i, block in enumerate(all_blocks):
            render_block(block, i, pdf_path, data, output_dir, ordered_ids)


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
