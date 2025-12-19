"""
repair_tables.py - Script de Hidratación de Imágenes de Tablas

Este script añade imágenes de tablas a los JSONs existentes sin tocar
las clasificaciones LLM (TRUE/FALSE) previas.

Funcionamiento:
1. Itera sobre la carpeta de resultados procesados (OUTPUT_DIR)
2. Para cada paper, carga el JSON existente
3. Ejecuta Docling SOLO para re-extraer imágenes de tablas del PDF original
4. Inyecta las rutas de imagen en el JSON sin modificar otros campos
5. Guarda el JSON actualizado
"""

import os
import json
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# --- CONFIGURACIÓN ---
OUTPUT_DIR = Path("/Users/xzamu/Desktop/PETase-database/PDFs/Parsed-NEWarticles")
INPUT_DIR = Path("/Users/xzamu/Desktop/PETase-database/PDFs/NEWarticles")

# Configurar Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TableRepair")


def setup_docling() -> DocumentConverter:
    """Configura Docling SOLO para extraer imágenes de tablas."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    
    # --- LA CLAVE: Activar imágenes de tablas ---
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = 3.0  # Alta calidad
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def repair_paper(paper_dir: Path, converter: DocumentConverter) -> None:
    """
    Repara un paper individual añadiendo imágenes de tablas a su JSON.
    
    Preserva todos los campos existentes, especialmente:
    - has_quantitative_data (clasificaciones LLM)
    - llm_classification
    - text_segments
    - artifacts
    """
    paper_id = paper_dir.name
    json_path = paper_dir / f"{paper_id}_data.json"
    
    if not json_path.exists():
        logger.warning(f"⚠️ JSON no encontrado para {paper_id}")
        return

    # 1. Cargar Datos Existentes (Preservando LLM tags)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Verificar si ya tiene imágenes (Idempotencia)
    # Si la primera tabla ya tiene 'image_path' con valor, asumimos que está listo
    tables_data = data.get("tables_data", [])
    if tables_data:
        first_table = tables_data[0]
        if first_table.get("image_path") is not None:
            logger.info(f"⏭️ {paper_id} ya tiene imágenes de tablas. Saltando.")
            return
    else:
        logger.info(f"ℹ️ {paper_id} no tiene tablas en el JSON. Saltando.")
        return

    # 3. Localizar PDF Original
    # Intentamos usar la ruta guardada en el JSON
    original_pdf = Path(data.get("original_pdf", ""))
    if not original_pdf.exists():
        # Fallback: Buscar en la carpeta de entrada por convención
        original_pdf = INPUT_DIR / paper_id / f"{paper_id}.pdf"
    
    if not original_pdf.exists():
        logger.error(f"❌ PDF no encontrado para {paper_id}. No se pueden regenerar imágenes.")
        logger.error(f"   Ruta esperada: {original_pdf}")
        return

    logger.info(f"🔨 Reparando tablas para: {paper_id}")

    try:
        # 4. Ejecutar Docling (Solo conversión para extraer imágenes)
        conv_res = converter.convert(str(original_pdf))
        doc = conv_res.document
        
        # Carpeta de artefactos
        artifacts_dir = paper_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        updates_count = 0
        
        # 5. Cruzar datos: Docling Tables vs. JSON Tables
        # Asumimos que el orden se mantiene (tbl_0, tbl_1...) que es determinista
        if hasattr(doc, 'tables'):
            for i, table in enumerate(doc.tables):
                # Validar que existe la entrada en el JSON
                if i < len(tables_data):
                    json_table = tables_data[i]
                    
                    # Verificar identidad básica (página)
                    doc_page = table.prov[0].page_no if table.prov else 0
                    json_page = json_table.get("page", 0)
                    
                    if json_page != doc_page and json_page != 0:
                        logger.warning(f"   ⚠️ Mismatch de página en tabla {i}: JSON={json_page}, Doc={doc_page}")

                    # 6. Guardar Imagen si existe
                    if hasattr(table, 'image') and table.image:
                        if hasattr(table.image, 'pil_image'):
                            img = table.image.pil_image
                            
                            # Nombre consistente: MAIN_PaperID_pX_tblY.png
                            filename = f"MAIN_{paper_id}_p{doc_page}_tbl{i}.png"
                            save_path = artifacts_dir / filename
                            
                            img.save(save_path)
                            
                            # 7. ACTUALIZAR JSON (Inyectar path)
                            json_table["image_path"] = str(save_path)
                            updates_count += 1
                            logger.debug(f"   ✓ Guardada: {filename}")
                else:
                    logger.warning(f"   ⚠️ Tabla {i} en Docling pero no en JSON. Posible inconsistencia.")
        
        # 8. Guardar JSON Actualizado
        if updates_count > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"✅ Guardado: {updates_count} imágenes de tablas añadidas a {paper_id}")
        else:
            logger.info(f"ℹ️ No se encontraron imágenes de tablas nuevas para {paper_id}")

    except Exception as e:
        logger.error(f"💥 Error procesando {paper_id}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Punto de entrada principal del script."""
    logger.info("🚀 Iniciando Script de Hidratación de Tablas...")
    logger.info(f"   📂 OUTPUT_DIR: {OUTPUT_DIR.absolute()}")
    logger.info(f"   📂 INPUT_DIR:  {INPUT_DIR.absolute()}")
    
    # Verificar que existen las carpetas
    if not OUTPUT_DIR.exists():
        logger.error(f"❌ Directorio de salida no existe: {OUTPUT_DIR}")
        logger.info("   Crea la carpeta o ajusta OUTPUT_DIR en el script.")
        return
    
    if not INPUT_DIR.exists():
        logger.warning(f"⚠️ Directorio de entrada no existe: {INPUT_DIR}")
        logger.info("   Se usará 'original_pdf' del JSON para localizar PDFs.")
    
    # Inicializar Docling
    logger.info("⚙️ Inicializando Docling...")
    converter = setup_docling()
    
    # Iterar sobre las carpetas PROCESADAS
    processed_folders = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    logger.info(f"📊 Encontrados {len(processed_folders)} papers para revisar.")

    for i, folder in enumerate(processed_folders):
        repair_paper(folder, converter)
        print(f"Progreso: {i+1}/{len(processed_folders)}")

    logger.info("✨ Proceso completado.")


if __name__ == "__main__":
    main()
