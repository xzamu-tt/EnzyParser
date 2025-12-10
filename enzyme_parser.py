"""
EnzyParser - Batch Scientific Paper Processor
Uses Docling for PDF/document extraction with focus on enzyme research papers.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel
from PIL import Image

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnzyParser")


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


class PaperData(BaseModel):
    """Complete paper representation with all extracted data"""
    paper_id: str  # e.g., Almeida-2019
    original_pdf: str
    metadata: Dict[str, Any]
    text_content: str
    artifacts: List[VisualArtifact] = []
    tables_data: List[Dict] = []
    supplementary_files_processed: List[str] = []


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
        
        # 3. Save JSON output
        json_path = paper_out_dir / f"{paper_id}_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper_data.model_dump_json(indent=2))
        
        logger.info(f"✅ Completed {paper_id}. Output: {paper_out_dir}")
    
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
                        
                        caption = ""
                        if hasattr(picture, 'caption_text'):
                            caption = picture.caption_text(doc) or ""
                        
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
            for table in doc.tables:
                try:
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe()
                        page_no = table.prov[0].page_no if table.prov else 0
                        paper_data.tables_data.append({
                            "source": pdf_path.name,
                            "page": page_no,
                            "data": df.to_dict(orient="records")
                        })
                except Exception as e:
                    logger.warning(f"Error extracting table: {e}")
        
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
