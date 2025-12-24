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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Nuevos imports para Gemini y Env
from dotenv import load_dotenv
import google.generativeai as genai
from mistral_enricher import MistralEnricher

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
DEFAULT_LLM_MODEL = "gemini-2.5-flash-lite"  # Single source of truth for model name
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
    has_quantitative_data: Optional[bool] = None


class PaperClassificationResult(BaseModel):
    """Estructura estricta para la clasificación global del paper."""
    is_enzyme_paper: bool
    enzyme_name: Optional[str] = None
    organism: Optional[str] = None
    relevance_score: float
    summary: str
    reasoning: str


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
    llm_classification: Optional[PaperClassificationResult] = None  # Modelo tipado fuerte


# --- Definición de Tipos para Schema Estricto (Sin Reasoning) ---
class ClassificationResult(typing.TypedDict):
    id: str
    is_data: bool

class DataClassifier:
    """
    Clasificador pragmático. 
    Estrategia: Lo que el modelo no confirme como FALSE, será TRUE.
    """
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="""TASK: Classify scientific text segments.
RETURN: JSON List of objects with keys 'id' and 'is_data'.
LOGIC:
- TRUE (is_data=true): Experimental results, values, units, kinetics (Km, kcat), conditions (pH, Temp), statistical data.
- FALSE (is_data=false): Methods, recipes, citations, general introduction.
OUTPUT ONLY THE JSON LIST."""
        )
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[ClassificationResult]
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def classify_batch(self, items: List[Dict]) -> Dict[str, bool]:
        """Classify batch with exponential backoff retry."""
        if not items: return {}
        
        # Minificamos el input
        minified = [{"id": i["id"], "text": i["content"]} for i in items]
        
        response = self.model.generate_content(
            json.dumps(minified),
            generation_config=self.generation_config
        )
        
        results = json.loads(response.text)
        
        # Convertimos a mapa simple
        return {res["id"]: res["is_data"] for res in results if "id" in res}


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
        
        # Enable picture and TABLE image extraction
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.generate_table_images = True  # Extract table snapshots as PNG
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
        
        # Initialize Mistral Enricher
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            self.enricher = MistralEnricher(api_key=mistral_key)
            logger.info("Mistral Enricher configured.")
        else:
            self.enricher = MistralEnricher() # Initialization will log warning
            
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

    def process_all_streaming(self, skip_llm: bool = False, force_rerun: bool = False):
        """
        Generator version of process_all for UI integration.
        Yields log messages as strings for real-time display.
        
        Args:
            skip_llm: If True, skip LLM classification (Stage 1 mode)
            force_rerun: If True, reprocess even if output already exists
        """
        dirs = [d for d in self.input_root.iterdir() if d.is_dir()]
        mode_label = "(Docling Only - No LLM)" if skip_llm else "(Full Processing)"
        rerun_label = "[FORCE RERUN]" if force_rerun else ""
        yield f"📁 Found {len(dirs)} paper directories to process. {mode_label} {rerun_label}"
        
        for i, paper_dir in enumerate(dirs, 1):
            yield f"\n--- [{i}/{len(dirs)}] Processing: {paper_dir.name} ---"
            try:
                for msg in self._process_paper_folder_streaming(paper_dir, skip_llm=skip_llm, force_rerun=force_rerun):
                    yield msg
            except Exception as e:
                yield f"❌ ERROR processing {paper_dir.name}: {e}"
        
        yield "\n✅ All papers processed!"

    def _process_paper_folder_streaming(self, folder_path: Path, skip_llm: bool = False, force_rerun: bool = False):
        """
        Generator version of process_paper_folder for streaming logs.
        
        Args:
            folder_path: Path to paper folder
            skip_llm: If True, skip LLM classification (Stage 1 mode)
            force_rerun: If True, reprocess even if output already exists
        """
        paper_id = folder_path.name
        expected_pdf_name = f"{paper_id}.pdf"
        main_pdf_path = folder_path / expected_pdf_name
        
        # Setup output paths
        paper_out_dir = self.output_root / paper_id
        json_path = paper_out_dir / f"{paper_id}_data.json"
        
        # --- CHECK DE IDEMPOTENCIA ---
        if json_path.exists() and not force_rerun:
            yield f"⏭️ SKIPPING {paper_id}: Output already exists. Use force_rerun to reprocess."
            return
        
        if not main_pdf_path.exists():
            yield f"⚠️ SKIPPING: Main PDF not found at {main_pdf_path}"
            return
        
        # Create output structure
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
                
                # --- MISTRAL ENRICHMENT (Post-Classification) ---
                if any(t.get("has_quantitative_data") for t in paper_data.tables_data):
                     yield f"🧪 Enriching tables with Mistral OCR..."
                     # Async wrapper for synchronous context (or proper async if using async framework)
                     # Since this method is a generator, we can run the loop here
                     import asyncio
                     try:
                         loop = asyncio.events.get_event_loop()
                     except RuntimeError:
                         loop = asyncio.new_event_loop()
                         asyncio.set_event_loop(loop)
                         
                     paper_data.tables_data = loop.run_until_complete(
                         self.enricher.process_paper_tables(paper_data.tables_data)
                     )
                     
                     n_enriched = sum(1 for t in paper_data.tables_data if t.get("markdown_content"))
                     yield f"  ✓ Enriched {n_enriched} tables with Markdown"

                     # Save AGAIN with enriched data
                     with open(json_path, "w", encoding="utf-8") as f:
                        f.write(paper_data.model_dump_json(indent=2))
                
                yield f"💾 Saved: {json_path}"
                classified_count += 1
                
            except Exception as e:
                yield f"❌ Error classifying {paper_id}: {e}"
        
        yield f"\n✅ Classification complete! Classified: {classified_count}, Skipped: {skipped_count}"
    
    def enrich_existing_papers_streaming(self):
        """
        Stage 3: Run Mistral OCR Enrichment on already-parsed & classified JSON files.
        Only processes tables with 'has_quantitative_data=True' and missing markdown.
        """
        if not self.has_gemini: # Mistral key is checked inside Enricher, but we can check here too
             pass
        
        # Check Mistral Key implicit via self.enricher
        if not self.enricher or not self.enricher.client:
             yield "❌ Cannot run Enrichment: Mistral Client not initialized (Check MISTRAL_API_KEY)"
             return

        paper_dirs = [d for d in self.output_root.iterdir() if d.is_dir()]
        yield f"📁 Found {len(paper_dirs)} paper directories. Scanning for tables to enrich..."
        
        enriched_count = 0
        
        import asyncio
        try:
            loop = asyncio.events.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        for i, paper_dir in enumerate(paper_dirs, 1):
            paper_id = paper_dir.name
            json_path = paper_dir / f"{paper_id}_data.json"
            
            if not json_path.exists():
                continue
            
            try:
                # Load
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                paper_data = PaperData(**data)
                
                # Check if needs enrichment
                # Criteria: has_quantitative_data=True AND markdown_content is None/Empty
                tables_to_enrich = [
                    t for t in paper_data.tables_data 
                    if t.get("has_quantitative_data") and not t.get("markdown_content")
                ]
                
                if not tables_to_enrich:
                    continue
                
                yield f"\n--- [{i}/{len(paper_dirs)}] Enriching: {paper_id} ---"
                yield f"  🧪 Found {len(tables_to_enrich)} tables pending enrichment..."
                
                # Run Enrichment
                paper_data.tables_data = loop.run_until_complete(
                    self.enricher.process_paper_tables(paper_data.tables_data)
                )
                
                n_newly_enriched = sum(1 for t in paper_data.tables_data if t.get("markdown_content") and t in tables_to_enrich) # Approximate check
                
                if n_newly_enriched > 0:
                     # Save
                     with open(json_path, "w", encoding="utf-8") as f:
                        f.write(paper_data.model_dump_json(indent=2))
                     yield f"  ✅ Enriched {n_newly_enriched} tables. Saved."
                     enriched_count += 1
                else:
                     yield f"  ⚠️ No content extracted from tables."
                
            except Exception as e:
                yield f"❌ Error processing {paper_id}: {e}"
                
        yield f"\n✨ Enrichment complete! Updated {enriched_count} papers."
    
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
    
    def _find_table_context(self, table, doc, tolerance_up: float = 100.0, tolerance_down: float = 80.0) -> Dict[str, str]:
        """
        Busca el 'Sandwich' de la tabla: Título arriba y Notas abajo.
        Retorna un dict con {'caption': str, 'notes': str}
        """
        context = {"caption": "", "notes": ""}
        
        # 1. Caption (Nativo o Búsqueda Arriba)
        base_caption = ""
        if hasattr(table, 'caption_text'):
            try:
                base_caption = table.caption_text(doc) or ""
            except:
                pass
        
        if len(base_caption) > 20:
            context["caption"] = base_caption
        elif table.prov:
            # Búsqueda manual hacia arriba
            tab_prov = table.prov[0]
            tab_top = getattr(tab_prov.bbox, 't', 0.0)
            tab_page = tab_prov.page_no
            candidates_up = []
            
            try:
                for item, _ in doc.iterate_items():
                    if isinstance(item, (TableItem, PictureItem)):
                        continue
                    if not item.prov or item.prov[0].page_no != tab_page:
                        continue
                    
                    text_bottom = getattr(item.prov[0].bbox, 'b', 0.0)
                    dist_up = tab_top - text_bottom
                    
                    if 0 < dist_up < tolerance_up:
                        text = getattr(item, 'text', '').strip()
                        if text and (text.lower().startswith("table") or len(text) > 10):
                            candidates_up.append(text)
            except Exception as e:
                logger.warning(f"Error finding table caption: {e}")
            
            # El título suele ser el último candidato (más cercano a la tabla)
            if candidates_up:
                context["caption"] = candidates_up[-1]

        # 2. Notas al pie (Búsqueda Abajo)
        if table.prov:
            tab_bottom = getattr(table.prov[0].bbox, 'b', 0.0)
            tab_page = table.prov[0].page_no
            candidates_down = []
            
            try:
                for item, _ in doc.iterate_items():
                    if isinstance(item, (TableItem, PictureItem)):
                        continue
                    if not item.prov or item.prov[0].page_no != tab_page:
                        continue
                    
                    text_top = getattr(item.prov[0].bbox, 't', 0.0)
                    dist_down = text_top - tab_bottom
                    
                    # Buscamos texto inmediatamente abajo
                    if 0 < dist_down < tolerance_down:
                        text = getattr(item, 'text', '').strip()
                        # Heurísticas para notas al pie
                        is_note = (
                            text.startswith("*") or 
                            text.lower().startswith("note") or 
                            text.lower().startswith("abbreviation") or 
                            len(text) < 150
                        )
                        if text and is_note:
                            candidates_down.append(text)
            except Exception as e:
                logger.warning(f"Error finding table notes: {e}")
            
            context["notes"] = " ".join(candidates_down)

        return context
    
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
                    # 1. Exportar datos (OCR - Intentamos obtenerlo, aunque falle)
                    try:
                        df = table.export_to_dataframe(doc)
                        data_records = df.to_dict(orient="records")
                    except Exception:
                        data_records = []

                    # 2. Guardar imagen de la tabla (si está disponible)
                    table_img_path = None
                    if hasattr(table, 'image') and table.image:
                        if hasattr(table.image, 'pil_image'):
                            img = table.image.pil_image
                            
                            prefix = "MAIN" if is_main else "SUPP"
                            page_no = table.prov[0].page_no if table.prov else 0
                            
                            filename = f"{prefix}_{pdf_path.stem}_p{page_no}_tbl{i}.png"
                            save_path = artifacts_dir / filename
                            
                            img.save(save_path)
                            table_img_path = str(save_path)
                            logger.info(f"   Guardada imagen de tabla: {filename}")

                    # 3. Contexto (Título y Notas)
                    ctx = self._find_table_context(table, doc)
                    
                    page_no = table.prov[0].page_no if table.prov else 0
                    
                    # Calcular BBox para UI
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    if table.prov:
                        b = table.prov[0].bbox
                        bbox = [
                            getattr(b, 'l', 0.0),
                            getattr(b, 't', 0.0),
                            getattr(b, 'r', 0.0),
                            getattr(b, 'b', 0.0)
                        ]
                    
                    # 4. Guardar objeto rico con imagen y datos OCR
                    paper_data.tables_data.append({
                        "id": f"tbl_{i}",
                        "type": "table",
                        "source": pdf_path.name,
                        "page": page_no,
                        "bbox": bbox,
                        "caption": ctx["caption"],
                        "table_notes": ctx["notes"],
                        "data": data_records,
                        "image_path": table_img_path,  # Póliza de seguro visual
                        "has_quantitative_data": True,
                        "markdown_content": None # Placeholder para Mistral OCR
                    })
                except Exception as e:
                    logger.warning(f"Error extracting table {i}: {e}")
        
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
                
                # Generamos un ID seguro basado en el nombre del archivo
                safe_id = f"supp_{file_path.stem.replace(' ', '_').replace('.', '_')}"
                
                # Estructura completa + Flag True para visibilidad en UI
                paper_data.tables_data.append({
                    "id": safe_id,
                    "type": "supplementary_table",
                    "source": file_path.name,
                    "page": 0,
                    "bbox": [0, 0, 0, 0],
                    "caption": f"Supplementary Data: {file_path.name}",
                    "data": df.to_dict(orient="records")[:100],  # Limit rows
                    "has_quantitative_data": True  # Obligatorio para que la UI lo muestre
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
        """
        Filtra contenido con estrategia 'Fail-Safe High Recall'.
        Cualquier fallo o silencio del LLM se asume como DATO RELEVANTE (TRUE).
        """
        if not self.classifier:
            return

        logger.info(f"⚡ Iniciando clasificación rápida con Gemini...")

        # 1. Recolectar todos los candidatos
        # (Mezclamos texto y figuras en una sola lista para eficiencia)
        candidates = []
        relevant_types = ['text', 'caption', 'section_header', 'list_item', 'table_cell'] 
        
        for seg in paper_data.text_segments:
            if seg.type in relevant_types:
                candidates.append({"id": seg.id, "content": seg.text, "ref": seg})
        
        for art in paper_data.artifacts:
            content = f"Caption: {art.caption}"
            if art.vlm_description: content += f" | AI: {art.vlm_description}"
            candidates.append({"id": art.id, "content": content, "ref": art})

        logger.info(f"   Total items a evaluar: {len(candidates)}")

        # 2. Métricas de Auditoría
        stats = {
            "sent": 0,
            "received": 0,
            "ignored_by_llm": 0, # <--- LO QUE QUIERES SABER
            "api_errors": 0
        }

        # 3. Procesar en Batches
        BATCH_SIZE = 20
        
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            stats["sent"] += len(batch)
            
            # Llamada API (Segura)
            try:
                # Quitamos la referencia al objeto antes de enviar al LLM para no romper JSON serializable
                batch_input = [{"id": item["id"], "content": item["content"]} for item in batch]
                results_map = self.classifier.classify_batch(batch_input)
            except Exception:
                results_map = {} # Simulamos respuesta vacía para activar fail-safe global
                stats["api_errors"] += 1

            stats["received"] += len(results_map)

            # 4. Asignación y Fail-Safe (Tu Lógica de Negocio)
            for item in batch:
                obj = item["ref"] # Recuperamos el objeto original (TextSegment o VisualArtifact)
                item_id = item["id"]

                if item_id in results_map:
                    # El LLM respondió explícitamente
                    obj.has_quantitative_data = results_map[item_id]
                else:
                    # El LLM IGNORÓ este ID o la API falló -> ASUMIMOS TRUE
                    obj.has_quantitative_data = True
                    stats["ignored_by_llm"] += 1 # Contamos el olvido

        # 5. Reporte Final en Consola (Lo que verás en Streamlit)
        if stats["ignored_by_llm"] > 0 or stats["api_errors"] > 0:
            logger.warning(f"⚠️ REPORTE DE INTEGRIDAD ({paper_data.paper_id}):")
            logger.warning(f"   Items Enviados: {stats['sent']}")
            logger.warning(f"   Respuestas Explícitas: {stats['received']}")
            logger.warning(f"   IGNORED BY LLM (Auto-TRUE): {stats['ignored_by_llm']} segmentos")
            if stats["api_errors"] > 0:
                logger.error(f"   Batches Fallidos: {stats['api_errors']}")
        else:
            logger.info(f"✅ Clasificación perfecta: {stats['sent']} items procesados sin fugas.")

    def filter_paper_with_llm(self, text_content: str) -> Dict[str, Any]:
        """
        Classifies the paper using Gemini to determine relevance to Enzyme Research.
        """
        if not self.has_gemini:
            return {"status": "skipped", "reason": "No API Key"}

        try:
            # Use configured LLM model
            model = genai.GenerativeModel(DEFAULT_LLM_MODEL)
            
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
            
            # Only send first ~20k chars (title + abstract + intro) for classification
            # Full text is unnecessary and wastes tokens/money
            final_prompt = prompt + text_content[:20000]

            response = model.generate_content(
                final_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            json_result = json.loads(response.text)
            
            # Validación automática al instanciar el modelo Pydantic
            # Si el LLM faltó un campo, esto lanzará error aquí mismo (Fail Fast)
            return PaperClassificationResult(**json_result).model_dump()

        except Exception as e:
            logger.error(f"Gemini processing failed: {e}")
            return {"status": "error", "message": str(e)}


# --- Entry Point ---

if __name__ == "__main__":
    # Configure paths
    INPUT_DIR = "/Users/xzamu/Desktop/PETase-database/PDFs/NEWarticles"
    OUTPUT_DIR = "/Users/xzamu/Desktop/PETase-database/PDFs/Parsed-NEWarticles"
    
    # Optional: Local VLM endpoint (LM Studio / Ollama / vLLM)
    # Set to None to skip VLM descriptions
    LOCAL_VLM_URL = None  # "http://localhost:1234/v1/chat/completions"
    
    parser = EnzymeParser(INPUT_DIR, OUTPUT_DIR, vlm_endpoint=LOCAL_VLM_URL)
    parser.process_all()
