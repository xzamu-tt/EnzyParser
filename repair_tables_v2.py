"""
repair_tables_v2.py - Script de Hidratación Multi-Archivo

Versión mejorada que:
1. Lee el JSON y busca todas las fuentes únicas de tablas (source)
2. Busca esos archivos (Main y Suplementarios) en la carpeta del paper
3. Procesa cada PDF necesario para recuperar todas las imágenes
4. Inyecta las rutas de imagen en el JSON sin modificar otros campos

Esto llena los huecos que faltan cuando tablas provienen de PDFs suplementarios.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TableRepairV2")


def setup_docling():
    """Configura Docling específicamente para extraer imágenes de tablas."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_table_images = True  # Objetivo principal
    pipeline_options.images_scale = 3.0
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def repair_paper(paper_dir: Path, converter: DocumentConverter) -> None:
    """
    Repara un paper individual añadiendo imágenes de tablas faltantes.
    
    Soporta:
    - PDF principal (Main)
    - PDFs suplementarios (Supp)
    
    Preserva todas las clasificaciones LLM existentes.
    """
    paper_id = paper_dir.name
    json_path = paper_dir / f"{paper_id}_data.json"
    
    if not json_path.exists():
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Identificar todos los PDFs que contienen tablas SIN imagen
    pdf_sources = set()
    for tbl in data.get("tables_data", []):
        # Si ya tiene imagen, ignoramos
        if tbl.get("image_path") is None:
            source_file = tbl.get("source")
            if source_file and source_file.lower().endswith(".pdf"):
                pdf_sources.add(source_file)

    if not pdf_sources:
        logger.info(f"⏭️ {paper_id}: Todas las tablas ya tienen imágenes o no son de PDF.")
        return

    logger.info(f"🔨 Procesando {len(pdf_sources)} archivos PDF para {paper_id}...")
    artifacts_dir = paper_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    updates_count = 0

    # 2. Procesar cada PDF identificado
    for pdf_name in pdf_sources:
        # Buscar el archivo físico en la carpeta de entrada
        pdf_path = INPUT_DIR / paper_id / pdf_name
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ No se encuentra el archivo fuente: {pdf_path}")
            continue
            
        try:
            # Ejecutar Docling para este archivo específico
            logger.info(f"   -> Generando imágenes de: {pdf_name}")
            conv_res = converter.convert(str(pdf_path))
            doc = conv_res.document
            
            # Determinar si es Main o Supp para el prefijo
            is_main = (pdf_name == f"{paper_id}.pdf")
            prefix = "MAIN" if is_main else "SUPP"

            # 3. Mapear las nuevas imágenes al JSON existente
            if hasattr(doc, 'tables'):
                for i, table in enumerate(doc.tables):
                    if hasattr(table, 'image') and table.image:
                        # Buscar la entrada correspondiente en tables_data
                        doc_page = table.prov[0].page_no if table.prov else 0
                        
                        for json_tbl in data["tables_data"]:
                            # CRITERIO DE MATCH: Mismo archivo fuente y misma página
                            if (json_tbl.get("source") == pdf_name and 
                                json_tbl.get("page") == doc_page and 
                                json_tbl.get("image_path") is None):
                                
                                # Guardar imagen
                                filename = f"{prefix}_{Path(pdf_name).stem}_p{doc_page}_tbl{i}.png"
                                save_path = artifacts_dir / filename
                                table.image.pil_image.save(save_path)
                                
                                # Actualizar JSON
                                json_tbl["image_path"] = str(save_path)
                                updates_count += 1
                                logger.debug(f"      ✓ Guardada: {filename}")
                                break  # Pasar a la siguiente tabla detectada por Docling

        except Exception as e:
            logger.error(f"❌ Error procesando {pdf_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Guardar cambios si hubo actualizaciones
    if updates_count > 0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ {paper_id}: Se inyectaron {updates_count} imágenes nuevas.")
    else:
        logger.info(f"ℹ️ {paper_id}: No se encontraron imágenes nuevas para inyectar.")


def main():
    """Punto de entrada principal del script."""
    logger.info("🚀 Iniciando Script de Reparación Multi-Archivo V2...")
    logger.info(f"   📂 OUTPUT_DIR: {OUTPUT_DIR.absolute()}")
    logger.info(f"   📂 INPUT_DIR:  {INPUT_DIR.absolute()}")
    
    # Verificar que existen las carpetas
    if not OUTPUT_DIR.exists():
        logger.error(f"❌ Directorio de salida no existe: {OUTPUT_DIR}")
        return
    
    if not INPUT_DIR.exists():
        logger.error(f"❌ Directorio de entrada no existe: {INPUT_DIR}")
        return
    
    # Inicializar Docling
    logger.info("⚙️ Inicializando Docling...")
    converter = setup_docling()
    
    # Iterar sobre las carpetas PROCESADAS
    processed_folders = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    logger.info(f"📊 Encontrados {len(processed_folders)} papers para revisar.")

    for i, folder in enumerate(processed_folders, 1):
        print(f"\n[{i}/{len(processed_folders)}] ", end="")
        repair_paper(folder, converter)

    logger.info("\n✨ Proceso completado.")


if __name__ == "__main__":
    main()
