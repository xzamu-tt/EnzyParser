"""
EnzyParser - Batch Scientific Paper Processor
Uses Docling for PDF/document extraction with focus on enzyme research papers.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import typing_extensions as typing

import pandas as pd
from pydantic import BaseModel
from PIL import Image

# Nuevos imports para Gemini y Env
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar variables de entorno
load_dotenv()

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.document import TableItem, PictureItem

# Configurar Gemini si existe la key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
# logger warning moved to init or main logic to avoid immediate execution side effects if possible, 
# but user requested "Al inicio del archivo", so placing it here.
# However, logger is not defined yet. 
# User snippet has "logger.warning". "logger" relies on basicConfig which is lower down.
# I will move basicConfig UP or define logger earlier.
# The user snippet implies imports -> logger config -> gemini config.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnzyParser")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env. LLM filtering will be disabled.")


# --- Pydantic Data Models ---

class VisualArtifact(BaseModel):
    """Represents an extracted visual element (figure, table image, etc.)"""
    id: str
    type: str  # 'figure', 'table_image', 'supplementary_image'
    file_path: str  # Local path to saved .png
    page_no: int
    bbox: List[float]  # [left, top, right, bottom]
    caption: str
    vlm_description: str  # Preliminary VLM description
    source_file: str
    has_quantitative_data: Optional[bool] = None  # NUEVO CAMPO


class TextSegment(BaseModel):
    """Represents a text block with layout grounding."""
    id: str
    type: str  # 'title', 'section_header', 'text', 'list_item', 'caption', 'footnote'
    text: str
    page_no: int
    bbox: List[float]  # [left, top, right, bottom]
    source_file: str
    has_quantitative_data: Optional[bool] = None  # NUEVO CAMPO


class PaperData(BaseModel):
    """Complete paper representation with all extracted data"""
    paper_id: str  # e.g., Almeida-2019
    original_pdf: str
    metadata: Dict[str, Any]
    text_content: str  # Full markdown (kept for backward compatibility)
    text_segments: List[TextSegment] = []  # Layout-aware chunks with grounding
    artifacts: List[VisualArtifact] = []
    tables_data: List[Dict] = []
    supplementary_files_processed: List[str] = []
    llm_classification: Optional[Dict[str, Any]] = None  # New field for LLM results


# --- Definición de Tipos Optimizada (Schema Estricto y Ligero) ---
class ClassificationResult(typing.TypedDict):
    id: str
    is_data: bool  # Renombrado de 'has_quantitative_data' para ahorrar tokens


class DataClassifier:
    """
    Clasificador Binario Optimizado (Gemini Flash).
    Estrategia: High Recall, Zero Reasoning, Minimal Tokens.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash-lite-preview-02-05"):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="""ROLE: Scientific Data Sieve.
TASK: Classify text chunks as TRUE if they contain EXPERIMENTAL QUANTITATIVE DATA, or FALSE if not.

TRUE CRITERIA (Experimental Measurements):
- Kinetic/Thermodynamic values (Km, kcat, Vmax, Tm, pH optima).
- Quantitative results linked to conditions ("Activity increased by 50% at 30°C").
- Statistical data tied to results (p-values, error bars).
- Specific yields or degradation rates.

FALSE CRITERIA (Context/Methods):
- Protocols/Recipes ("Mixed 50mM buffer").
- Hardware/Software specs.
- General citations or broad introduction statements.
- Figure descriptions without explicit values (unless caption contains the data).

OUTPUT: Strict JSON list. NO reasoning. NO explanations."""
        )
        
        # Configuración para salida determinista y limpia
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[ClassificationResult]
        )

    def classify_batch(self, items: List[Dict]) -> Dict[str, bool]:
        """
        Procesa lotes minimizando overhead.
        """
        if not items:
            return {}

        try:
            # Minificamos el JSON de entrada para ahorrar tokens de input
            # Enviamos solo ID y Content, sin claves extrañas
            minified_items = [{"id": i["id"], "text": i["content"]} for i in items]
            
            response = self.model.generate_content(
                json.dumps(minified_items),
                generation_config=self.generation_config
            )
            
            results = json.loads(response.text)
            
            classification_map = {}
            for res in results:
                # Mapeamos la respuesta corta 'is_data' a la lógica interna
                item_id = res.get("id")
                is_quantitative = res.get("is_data", False)
                
                if item_id:
                    classification_map[item_id] = is_quantitative
            
            return classification_map

        except Exception as e:
            logger.error(f"❌ Batch Error: {e}")
            # Fallback seguro: Si falla, asumimos True para revisión manual (High Recall)
            return {item["id"]: True for item in items}


# --- Main Parser Class ---

class EnzymeParser:
    """
    Batch processor for scientific enzyme papers using Docling.
    
    Features:
    - Folder-based heuristics: FolderName.pdf = main paper
    - High-recall visual extraction: all figures saved as PNG
    - Supplementary file handling: PDF, Excel, images
    - Grounding: exact coordinates for traceability
    """
    
    def __init__(
        self,
        input_root: str,
        output_root: str,
        vlm_endpoint: Optional[str] = None
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.has_gemini = True
            self.classifier = DataClassifier()
            logger.info("Gemini API configured successfully")
        else:
            self.has_gemini = False
            self.classifier = None
            logger.warning("GOOGLE_API_KEY not found in .env. LLM features disabled.")

        # Configure Docling pipeline
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        
        # Enable picture extraction
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = 3.0  # High resolution for scientific graphics
        
        # VLM configuration (optional)
        if vlm_endpoint:
            try:
                from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
                self.pipeline_options.do_picture_description = True
                self.pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                    url=vlm_endpoint,
                    params={
                        "model": "local-model",
                        "temperature": 0.1,
                        "max_tokens": 200
                    },
                    prompt="Is this a scientific graph, western blot, or molecular structure? Describe axes and labels if present.",
                    timeout=30
                )
            except ImportError:
                logger.warning("PictureDescriptionApiOptions not available, skipping VLM")
                self.pipeline_options.do_picture_description = False
        else:
            self.pipeline_options.do_picture_description = False
        
        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
    
    def process_all(self) -> None:
        """Process all paper folders in input directory."""
        dirs = [d for d in self.input_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(dirs)} paper directories to process.")
        
        for paper_dir in dirs:
            try:
                self.process_paper_folder(paper_dir)
            except Exception as e:
                logger.error(f"Error processing {paper_dir.name}: {e}")

    def process_all_streaming(self, skip_llm: bool = False):
        """
        Generator version of process_all for UI integration.
        Yields log messages as strings for real-time display.
        
        Args:
            skip_llm: If True, skip LLM classification (Stage 1 mode)
        """
        dirs = [d for d in self.input_root.iterdir() if d.is_dir()]
        mode_label = "(Docling Only - No LLM)" if skip_llm else "(Full Processing)"
        yield f"📁 Found {len(dirs)} paper directories to process. {mode_label}"
        
        for i, paper_dir in enumerate(dirs, 1):
            yield f"\n--- [{i}/{len(dirs)}] Processing: {paper_dir.name} ---"
            try:
                for msg in self._process_paper_folder_streaming(paper_dir, skip_llm=skip_llm):
                    yield msg
            except Exception as e:
                yield f"❌ ERROR processing {paper_dir.name}: {e}"
        
        yield "\n✅ All papers processed!"

    def _process_paper_folder_streaming(self, folder_path: Path, skip_llm: bool = False):
        """
        Generator version of process_paper_folder for streaming logs.
        
        Args:
            folder_path: Path to paper folder
            skip_llm: If True, skip LLM classification (Stage 1 mode)
        """
        paper_id = folder_path.name
        expected_pdf_name = f"{paper_id}.pdf"
        main_pdf_path = folder_path / expected_pdf_name
        
        if not main_pdf_path.exists():
            yield f"⚠️ SKIPPING: Main PDF not found at {main_pdf_path}"
            return
        
        # Create output structure
        paper_out_dir = self.output_root / paper_id
        artifacts_dir = paper_out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        yield f"📄 Parsing PDF with Docling..."
        
        # 1. Process main PDF
        paper_data = self._parse_docling(
            main_pdf_path,
            paper_id,
            artifacts_dir,
            is_main=True
        )
        
        yield f"  ✓ Extracted {len(paper_data.text_segments)} text segments, {len(paper_data.artifacts)} figures"
        
        # 2. Process supplementary files
        supp_count = 0
        for file_path in folder_path.iterdir():
            if file_path.name == expected_pdf_name or file_path.name.startswith("."):
                continue
            self._process_supplementary(file_path, paper_data, artifacts_dir)
            supp_count += 1
        
        if supp_count > 0:
            yield f"  ✓ Processed {supp_count} supplementary files"

        # 3. LLM Filtering / Classification (SKIP IF skip_llm=True)
        if skip_llm:
            yield "⏭️ Skipping LLM classification (Stage 1 mode)"
        elif self.has_gemini and paper_data.text_content:
            yield f"🤖 Running LLM Classification..."
            classification = self.filter_paper_with_llm(paper_data.text_content)
            paper_data.llm_classification = classification
            
            if classification.get("is_enzyme_paper"):
                yield f"  ✓ [RELEVANT] Enzyme: {classification.get('enzyme_name', 'Unknown')}"
            else:
                yield f"  ⚠️ [NOT RELEVANT] Score: {classification.get('relevance_score', 0)}"
            
            # Detailed Classification
            yield f"🔬 Classifying {len(paper_data.text_segments)} segments..."
            self._classify_content(paper_data)
            n_positive = sum(1 for s in paper_data.text_segments if s.has_quantitative_data)
            yield f"  ✓ Found {n_positive} segments with quantitative data"
        else:
            yield "⚠️ Skipping LLM classification (no API key)"
        
        # 4. Save JSON output
        json_path = paper_out_dir / f"{paper_id}_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper_data.model_dump_json(indent=2))
        
        yield f"💾 Saved: {json_path}"
    
    def classify_existing_papers_streaming(self):
        """
        Stage 2: Run LLM classification on already-parsed JSON files.
        Iterates over output directory and classifies papers that haven't been classified yet.
        """
        if not self.has_gemini:
            yield "❌ Cannot run LLM classification: No GOOGLE_API_KEY configured"
            return
        
        # Find all paper directories in output
        paper_dirs = [d for d in self.output_root.iterdir() if d.is_dir()]
        yield f"📁 Found {len(paper_dirs)} paper directories in output folder."
        
        classified_count = 0
        skipped_count = 0
        
        for i, paper_dir in enumerate(paper_dirs, 1):
            paper_id = paper_dir.name
            json_path = paper_dir / f"{paper_id}_data.json"
            
            if not json_path.exists():
                yield f"⚠️ [{i}/{len(paper_dirs)}] {paper_id}: No JSON file found, skipping"
                skipped_count += 1
                continue
            
            yield f"\n--- [{i}/{len(paper_dirs)}] Classifying: {paper_id} ---"
            
            try:
                # Load existing paper data
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                paper_data = PaperData(**data)
                
                # Check if already classified
                already_classified = any(
                    s.has_quantitative_data is not None 
                    for s in paper_data.text_segments
                )
                
                if already_classified:
                    n_positive = sum(1 for s in paper_data.text_segments if s.has_quantitative_data)
                    yield f"  ⚠️ Already classified ({n_positive} positive segments). Re-classifying..."
                
                # Run paper-level classification
                if paper_data.text_content:
                    yield f"🤖 Running paper-level classification..."
                    classification = self.filter_paper_with_llm(paper_data.text_content)
                    paper_data.llm_classification = classification
                    
                    if classification.get("is_enzyme_paper"):
                        yield f"  ✓ [RELEVANT] Enzyme: {classification.get('enzyme_name', 'Unknown')}"
                    else:
                        yield f"  ⚠️ [NOT RELEVANT] Score: {classification.get('relevance_score', 0)}"
                
                # Run segment-level classification
                yield f"🔬 Classifying {len(paper_data.text_segments)} segments..."
                self._classify_content(paper_data)
                n_positive = sum(1 for s in paper_data.text_segments if s.has_quantitative_data)
                yield f"  ✓ Found {n_positive} segments with quantitative data"
                
                # Save updated JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(paper_data.model_dump_json(indent=2))
                
                yield f"💾 Saved: {json_path}"
                classified_count += 1
                
            except Exception as e:
                yield f"❌ Error classifying {paper_id}: {e}"
        
        yield f"\n✅ Classification complete! Classified: {classified_count}, Skipped: {skipped_count}"
    
    def process_paper_folder(self, folder_path: Path) -> None:
        """
        Process a single paper folder.
        
        Convention: FolderName.pdf is the main paper.
        Everything else is supplementary material.
        """
        paper_id = folder_path.name
        expected_pdf_name = f"{paper_id}.pdf"
        main_pdf_path = folder_path / expected_pdf_name
        
        if not main_pdf_path.exists():
            logger.warning(f"SKIPPING: Main PDF not found at {main_pdf_path}")
            return
        
        # Create output structure
        paper_out_dir = self.output_root / paper_id
        artifacts_dir = paper_out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing: {paper_id}")
        
        # 1. Process main PDF
        paper_data = self._parse_docling(
            main_pdf_path,
            paper_id,
            artifacts_dir,
            is_main=True
        )
        
        # 2. Process supplementary files
        for file_path in folder_path.iterdir():
            if file_path.name == expected_pdf_name or file_path.name.startswith("."):
                continue
            self._process_supplementary(file_path, paper_data, artifacts_dir)

        # 3. LLM Filtering / Classification
        if self.has_gemini and paper_data.text_content:
            logger.info(f"  -> Running LLM Classification for {paper_id}...")
            classification = self.filter_paper_with_llm(paper_data.text_content)
            paper_data.llm_classification = classification
            
            # Optional: Log if relevant
            if classification.get("is_relevant"):
                logger.info(f"  -> [RELEVANT] {classification.get('reasoning')}")
            else:
                logger.info(f"  -> [NOT RELEVANT] {classification.get('reasoning')}")
        
        # New Detailed Classification
        self._classify_content(paper_data)
        
        # 4. Save JSON output
        json_path = paper_out_dir / f"{paper_id}_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper_data.model_dump_json(indent=2))
        
        logger.info(f"✅ Completed {paper_id}. Output: {paper_out_dir}")
    
    def _find_extended_caption(self, picture, doc, tolerance: float = 100.0) -> str:
        """
        Find extended caption by looking for text blocks physically below the image.
        
        Uses geometric proximity heuristic to capture full figure legends that
        Docling may have classified as regular text instead of captions.
        
        Args:
            picture: The PictureItem from Docling
            doc: The document object
            tolerance: Maximum distance in points below the image to search
        
        Returns:
            Combined caption string (base caption + extended text)
        """
        # Get base caption from Docling
        base_caption = ""
        if hasattr(picture, 'caption_text'):
            base_caption = picture.caption_text(doc) or ""
        
        extended_text = []
        
        # Get image coordinates
        if not picture.prov:
            return base_caption
        
        pic_prov = picture.prov[0]
        pic_page = pic_prov.page_no
        pic_bbox = pic_prov.bbox
        pic_bottom = getattr(pic_bbox, 'b', 0.0)
        pic_left = getattr(pic_bbox, 'l', 0.0)
        pic_right = getattr(pic_bbox, 'r', 0.0)
        
        # Iterate over all text items on the same page
        if not hasattr(doc, 'iterate_items'):
            return base_caption
        
        try:
            for item, level in doc.iterate_items():
                # Skip non-text items
                if isinstance(item, (TableItem, PictureItem)):
                    continue
                
                if not hasattr(item, 'prov') or not item.prov:
                    continue
                
                item_prov = item.prov[0]
                
                # Only same page
                if item_prov.page_no != pic_page:
                    continue
                
                # Get text bbox
                if not hasattr(item_prov, 'bbox') or not item_prov.bbox:
                    continue
                
                text_bbox = item_prov.bbox
                text_top = getattr(text_bbox, 't', 0.0)
                text_left = getattr(text_bbox, 'l', 0.0)
                text_right = getattr(text_bbox, 'r', 0.0)
                
                # PROXIMITY CHECK:
                # 1. Text is BELOW the image (text_top > pic_bottom)
                # 2. Text is CLOSE (distance < tolerance)
                distance = text_top - pic_bottom
                
                if 0 < distance < tolerance:
                    # Check horizontal alignment to avoid capturing adjacent columns
                    text_width = text_right - text_left
                    if text_width > 0:
                        horiz_overlap = max(0, min(pic_right, text_right) - max(pic_left, text_left))
                        overlap_ratio = horiz_overlap / text_width
                        
                        # If text overlaps at least 50% with image width
                        if overlap_ratio > 0.5:
                            text = getattr(item, 'text', '')
                            if text and text.strip():
                                extended_text.append(text.strip())
        except Exception as e:
            logger.warning(f"Error finding extended caption: {e}")
        
        # Combine base caption with extended text
        if extended_text:
            full_caption = base_caption + "\n" + "\n".join(extended_text)
            return full_caption
        
        return base_caption
    
    def _find_table_caption(self, table, doc, tolerance: float = 100.0) -> str:
        """
        Busca leyendas de tablas mirando FÍSICAMENTE ARRIBA del elemento.
        En papers científicos, el título de la tabla casi siempre precede a la tabla.
        """
        # Intento 1: Docling nativo (a veces acierta)
        base_caption = ""
        if hasattr(table, 'caption_text'):
            try:
                base_caption = table.caption_text(doc) or ""
            except:
                pass
        
        # Si Docling ya lo encontró y es largo, confiamos.
        if len(base_caption) > 20:
            return base_caption

        # Intento 2: Búsqueda Geométrica Inversa (Look Upwards)
        extended_text = []
        
        if not table.prov:
            return base_caption
        
        tab_prov = table.prov[0]
        tab_page = tab_prov.page_no
        tab_top = getattr(tab_prov.bbox, 't', 0.0)  # El borde SUPERIOR de la tabla
        
        try:
            # Iteramos items de la misma página
            for item, level in doc.iterate_items():
                if isinstance(item, (TableItem, PictureItem)):
                    continue
                if not hasattr(item, 'prov') or not item.prov:
                    continue
                
                item_prov = item.prov[0]
                if item_prov.page_no != tab_page:
                    continue
                
                # Coordenadas del texto candidato
                text_bottom = getattr(item_prov.bbox, 'b', 0.0)
                
                # LÓGICA: El texto debe estar ARRIBA de la tabla
                # En coordenadas top-left (Y crece hacia abajo):
                # El bottom del texto debe ser menor que el top de la tabla
                distance = tab_top - text_bottom
                
                # Si está cerca (0 a 100px encima)
                if 0 < distance < tolerance:
                    text = getattr(item, 'text', '').strip()
                    if text:
                        # Heurística extra: Si empieza con "Table", es el ganador seguro.
                        if text.lower().startswith("table") or len(text) > 10:
                            extended_text.append(text)
        except Exception as e:
            logger.warning(f"Error finding table caption: {e}")

        return " ".join(extended_text) if extended_text else base_caption
    
    def _parse_docling(
        self,
        pdf_path: Path,
        paper_id: str,
        artifacts_dir: Path,
        is_main: bool
    ) -> PaperData:
        """Core parsing logic using Docling."""
        try:
            conv_res = self.converter.convert(str(pdf_path))
            doc = conv_res.document
        except Exception as e:
            logger.error(f"Docling conversion failed for {pdf_path.name}: {e}")
            return PaperData(
                paper_id=paper_id,
                original_pdf=str(pdf_path),
                metadata={},
                text_content=""
            )
        
        # Initialize paper data
        paper_data = PaperData(
            paper_id=paper_id,
            original_pdf=str(pdf_path),
            metadata=doc.metadata.model_dump() if hasattr(doc, 'metadata') and doc.metadata else {},
            text_content=doc.export_to_markdown() if hasattr(doc, 'export_to_markdown') else ""
        )
        
        # Extract figures/pictures
        if hasattr(doc, 'pictures'):
            for i, picture in enumerate(doc.pictures):
                try:
                    # Get rendered image
                    img = None
                    if hasattr(picture, 'image') and picture.image:
                        if hasattr(picture.image, 'pil_image'):
                            img = picture.image.pil_image
                    
                    if img:
                        prefix = "MAIN" if is_main else "SUPP"
                        page_no = picture.prov[0].page_no if picture.prov else 0
                        filename = f"{prefix}_{pdf_path.stem}_p{page_no}_fig{i}.png"
                        save_path = artifacts_dir / filename
                        img.save(save_path)
                        
                        # Extract bounding box
                        bbox = [0.0, 0.0, 0.0, 0.0]
                        if picture.prov and hasattr(picture.prov[0], 'bbox'):
                            b = picture.prov[0].bbox
                            bbox = [
                                getattr(b, 'l', 0.0),
                                getattr(b, 't', 0.0),
                                getattr(b, 'r', 0.0),
                                getattr(b, 'b', 0.0)
                            ]
                        
                        # Extract caption and VLM description
                        vlm_text = ""
                        if hasattr(picture, 'annotations') and picture.annotations:
                            vlm_text = picture.annotations[0].text if picture.annotations else ""
                        
                        # Extract caption using proximity heuristic
                        caption = self._find_extended_caption(picture, doc, tolerance=100.0)
                        
                        artifact = VisualArtifact(
                            id=f"fig_{i}",
                            type="figure",
                            file_path=str(save_path),
                            page_no=page_no,
                            bbox=bbox,
                            caption=caption,
                            vlm_description=vlm_text or "No VLM analysis",
                            source_file=pdf_path.name
                        )
                        paper_data.artifacts.append(artifact)
                except Exception as e:
                    logger.warning(f"Error extracting picture {i}: {e}")
        
        # Extract tables
        if hasattr(doc, 'tables'):
            for i, table in enumerate(doc.tables):
                try:
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe(doc)  # Pasamos 'doc' para evitar warning
                        
                        # --- CORRECCIÓN CRÍTICA ---
                        # 1. Buscamos el título ARRIBA de la tabla
                        caption = self._find_table_caption(table, doc)
                        
                        page_no = table.prov[0].page_no if table.prov else 0
                        
                        # 2. Guardamos con estructura rica e ID
                        paper_data.tables_data.append({
                            "id": f"tbl_{i}",  # ID único para referenciar
                            "type": "table",   # Para diferenciar en la UI
                            "source": pdf_path.name,
                            "page": page_no,
                            "caption": caption,
                            "data": df.to_dict(orient="records"),
                            "has_quantitative_data": True  # Asumir relevancia por defecto
                        })
                except Exception as e:
                    logger.warning(f"Error extracting table: {e}")
        
        # Extract text segments with layout grounding and PRE-FILTERING
        if hasattr(doc, 'iterate_items'):
            segment_counter = 0
            
            # DEFINIR TIPOS A IGNORAR TOTALMENTE (ruido estructural)
            IGNORED_LABELS = {'page_header', 'page_footer', 'footnote', 'reference'}
            
            try:
                for item, level in doc.iterate_items():
                    # 1. Filtro Básico de Tipo (tablas y figuras se manejan aparte)
                    if isinstance(item, (TableItem, PictureItem)):
                        continue
                    
                    # Obtener etiqueta original de Docling
                    docling_label = str(getattr(item, 'label', 'text')).lower()
                    
                    # 2. Filtro de Ruido Estructural (Headers/Footers/Footnotes)
                    if docling_label in IGNORED_LABELS:
                        continue
                    
                    # Obtener texto limpio
                    text = getattr(item, 'text', '')
                    if not text:
                        continue
                    text = text.strip()
                    
                    if not text:
                        continue
                    
                    # 3. Filtro de URLs y DOIs (siempre basura para extracción científica)
                    if "http" in text.lower() or "www." in text.lower() or "doi.org" in text.lower():
                        continue
                    
                    # 4. Filtro de Longitud (Heurística de Ruido MEJORADA)
                    # Si es texto plano y muy corto (<40 chars), suele ser basura visual
                    # PERO salvamos: Títulos, Headers, Captions, y texto que parece caption
                    
                    # --- CORRECCIÓN: Definir qué es "Sagrado" ---
                    is_title = 'title' in docling_label
                    is_header = 'header' in docling_label or 'heading' in docling_label
                    is_caption = 'caption' in docling_label  # <--- IMPORTANTE
                    
                    # El texto que empieza con "Table" o "Figure" se salva aunque Docling lo etiquete mal
                    looks_like_caption = text.lower().startswith("table") or text.lower().startswith("fig")
                    
                    # Solo borramos si es corto Y NO es nada importante
                    if len(text) < 40:
                        if not (is_title or is_header or is_caption or looks_like_caption):
                            continue
                    
                    # Get provenance (coordinates)
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    page_no = 0
                    if hasattr(item, 'prov') and item.prov:
                        prov = item.prov[0]
                        if hasattr(prov, 'bbox') and prov.bbox:
                            b = prov.bbox
                            bbox = [
                                getattr(b, 'l', 0.0),
                                getattr(b, 't', 0.0),
                                getattr(b, 'r', 0.0),
                                getattr(b, 'b', 0.0)
                            ]
                        if hasattr(prov, 'page_no'):
                            page_no = prov.page_no
                    
                    # Normalizar etiquetas para nuestro sistema
                    seg_type = "text"
                    if 'title' in docling_label:
                        seg_type = 'title'
                    elif 'section' in docling_label or 'header' in docling_label or 'heading' in docling_label:
                        seg_type = 'section_header'
                    elif 'caption' in docling_label:
                        seg_type = 'caption'
                    elif 'list' in docling_label:
                        seg_type = 'list_item'
                    else:
                        seg_type = 'text'  # Default to 'text' for cleaner output
                    
                    segment = TextSegment(
                        id=f"txt_{segment_counter}",
                        type=seg_type,
                        text=text,
                        page_no=page_no,
                        bbox=bbox,
                        source_file=pdf_path.name
                    )
                    paper_data.text_segments.append(segment)
                    segment_counter += 1
            except Exception as e:
                logger.warning(f"Error extracting text segments: {e}")
        
        return paper_data
    
    def _process_supplementary(
        self,
        file_path: Path,
        paper_data: PaperData,
        artifacts_dir: Path
    ) -> None:
        """Route supplementary files by type."""
        logger.info(f"  -> Supplementary: {file_path.name}")
        paper_data.supplementary_files_processed.append(file_path.name)
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            # Process supplementary PDF
            supp_data = self._parse_docling(
                file_path,
                paper_data.paper_id,
                artifacts_dir,
                is_main=False
            )
            paper_data.artifacts.extend(supp_data.artifacts)
            paper_data.tables_data.extend(supp_data.tables_data)
        
        elif suffix in [".xlsx", ".xls", ".csv"]:
            # Process Excel/CSV
            try:
                if suffix == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                paper_data.tables_data.append({
                    "source": file_path.name,
                    "type": "supplementary_raw_data",
                    "data": df.to_dict(orient="records")[:100]  # Limit rows
                })
            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")
        
        elif suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            # Process standalone images
            try:
                with Image.open(file_path) as img:
                    # Convert to PNG
                    filename = f"SUPP_IMG_{file_path.stem}.png"
                    save_path = artifacts_dir / filename
                    img.convert("RGB").save(save_path, "PNG")
                    
                    paper_data.artifacts.append(VisualArtifact(
                        id=file_path.stem,
                        type="supplementary_image",
                        file_path=str(save_path),
                        page_no=0,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        caption=file_path.name,
                        vlm_description="Pending analysis",
                        source_file=file_path.name
                    ))
            except Exception as e:
                logger.error(f"Error processing image {file_path.name}: {e}")

    def _classify_content(self, paper_data: PaperData):
        """Filters text segments and visual artifacts using LLM."""
        if not self.classifier:
            logger.info("Skipping LLM classification (No API Key)")
            return

        logger.info(f"Classifying {len(paper_data.text_segments)} segments and {len(paper_data.artifacts)} artifacts...")

        # 1. Preparar Payload (mezcla de texto e imágenes/captions)
        # Convertimos objetos a dicts simples para el LLM
        items_to_classify = []
        
        # Filtrar solo tipos relevantes para ahorrar tokens
        # Ignoramos 'page_footer', 'checkbox', etc.
        relevant_types = ['text', 'caption', 'section_header', 'list_item', 'table_cell'] 
        
        for seg in paper_data.text_segments:
            if seg.type in relevant_types and len(seg.text) > 20: # Ignorar textos muy cortos
                items_to_classify.append({
                    "id": seg.id,
                    "type": seg.type,
                    "content": seg.text
                })
                
        for art in paper_data.artifacts:
            # Para imágenes usamos el caption + descripción VLM si existe
            content = f"Caption: {art.caption}"
            if art.vlm_description:
                content += f" | AI Description: {art.vlm_description}"
                
            items_to_classify.append({
                "id": art.id,
                "type": art.type,
                "content": content
            })

        # 2. Procesar en Batches (Lotes de 50 items para seguridad)
        BATCH_SIZE = 50
        classification_results = {}
        
        for i in range(0, len(items_to_classify), BATCH_SIZE):
            batch = items_to_classify[i:i + BATCH_SIZE]
            logger.info(f"  -> Sending batch {i} to {i+len(batch)} to Gemini...")
            
            batch_results = self.classifier.classify_batch(batch)
            classification_results.update(batch_results)
            
            # Pequeña pausa para rate limits si no tienes tier pagado
            time.sleep(1) 

        # 3. Actualizar los objetos originales con el resultado
        count_positive = 0
        
        for seg in paper_data.text_segments:
            if seg.id in classification_results:
                seg.has_quantitative_data = classification_results[seg.id]
                if seg.has_quantitative_data: count_positive += 1
            else:
                seg.has_quantitative_data = False # Default si no se procesó

        for art in paper_data.artifacts:
            if art.id in classification_results:
                art.has_quantitative_data = classification_results[art.id]
                if art.has_quantitative_data: count_positive += 1
            else:
                art.has_quantitative_data = False

        logger.info(f"  -> Classification complete. Found {count_positive} items with data.")

    def filter_paper_with_llm(self, text_content: str) -> Dict[str, Any]:
        """
        Classifies the paper using Gemini 1.5 Flash to determine relevance to Enzyme Research.
        """
        if not self.has_gemini:
            return {"status": "skipped", "reason": "No API Key"}

        try:
            # Use Gemini 2.5 Flash Lite
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            # Construct Prompt
            prompt = """
            You are an expert scientist assisting in building a database of Enzyme Characterization papers.
            Analyze the following scientific paper text and output a JSON object with the following fields:
            
            {
                "is_enzyme_paper": boolean,  // true if paper primarily studies enzymes (characterization, discovery, engineering)
                "enzyme_name": "string or null", // Predicted main enzyme name
                "organism": "string or null",   // Source organism if applicable
                "relevance_score": float,       // 0.0 to 1.0 relevance to enzyme characterization
                "summary": "string",            // One sentence summary
                "reasoning": "string"           // Why it is or isn't relevant
            }

            Text content:
            """
            
            # For now passing full text as Docling markdown is usually reasonable in size.
            final_prompt = prompt + text_content[:500000] # Safety limit of 500k chars ~120k tokens

            response = model.generate_content(
                final_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Gemini processing failed: {e}")
            return {"status": "error", "message": str(e)}


# --- Entry Point ---

if __name__ == "__main__":
    # Configure paths
    INPUT_DIR = "./NEWarticles"
    OUTPUT_DIR = "./Processed_Enzyme_Data"
    
    # Optional: Local VLM endpoint (LM Studio / Ollama / vLLM)
    # Set to None to skip VLM descriptions
    LOCAL_VLM_URL = None  # "http://localhost:1234/v1/chat/completions"
    
    parser = EnzymeParser(INPUT_DIR, OUTPUT_DIR, vlm_endpoint=LOCAL_VLM_URL)
    parser.process_all()
