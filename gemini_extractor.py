"""
GeminiExtractor - Stage 4: Structured Data Extraction
Uses Gemini API with token-optimized prompting to extract experimental data
from enriched Markdown tables into structured DataFrames.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv

# Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()
logger = logging.getLogger("EnzyParser")

# ============================================================================
# CONSTANTS & SCHEMAS
# ============================================================================

# Activity row schema (17 columns) - Compatible with activity_2025.csv
ACTIVITY_COLUMNS = [
    "paper_id", "enzyme_name", "organism", "seq_aa", "substrate",
    "activity_value", "activity_unit", "activity_normalized",
    "ph", "temperature_c", "time_h", "buffer",
    "reaction_volume_ml", "enzyme_loading_value", "enzyme_loading_unit",
    "substrate_amount_value", "substrate_amount_unit"
]

# Tm row schema (9 columns) - Compatible with tm_expression_2025.csv
TM_COLUMNS = [
    "paper_id", "enzyme_name", "organism", "seq_aa",
    "tm_value", "tm_unit", "method", "buffer", "notes"
]

# Load unit configuration
UNITS_CONFIG_PATH = Path(__file__).parent / "units_config.json"
if UNITS_CONFIG_PATH.exists():
    with open(UNITS_CONFIG_PATH, "r") as f:
        UNITS_CONFIG = json.load(f)
else:
    UNITS_CONFIG = {}
    logger.warning(f"units_config.json not found at {UNITS_CONFIG_PATH}")


# ============================================================================
# DATACLASSES FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class GlobalContext:
    """Paper-level metadata extracted once."""
    paper_id: str = ""
    enzyme_name: str = ""
    organism: str = ""
    seq_aa: str = ""
    substrate: str = ""
    buffer: str = ""


@dataclass
class ActivityObservation:
    """Single activity measurement."""
    block_id: str = ""
    activity_value: float = 0.0
    activity_unit: str = ""
    ph: float = 0.0
    temperature_c: float = 0.0
    time_h: float = 0.0
    substrate_amount_value: float = 0.0
    substrate_amount_unit: str = ""
    enzyme_loading_value: float = 0.0
    enzyme_loading_unit: str = ""
    reaction_volume_ml: float = 0.0


@dataclass
class TmObservation:
    """Single Tm measurement."""
    block_id: str = ""
    tm_value: float = 0.0
    tm_unit: str = "C"
    method: str = ""
    notes: str = ""


# ============================================================================
# CORE EXTRACTOR CLASS
# ============================================================================

class GeminiExtractor:
    """
    Extractor using Gemini API with token optimization.
    
    Strategies:
    1. Markdown Distillation: Map blocks to [B1], [B2] IDs
    2. Schema Minimization: Compact JSON keys (act, u, b)
    3. Global vs Local: Extract paper context once, then observations
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not found. Extraction disabled.")
            self.model = None
            return
        
        if not genai:
            logger.error("❌ google-generativeai not installed.")
            self.model = None
            return
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"✅ GeminiExtractor initialized with {model_name}")
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: MARKDOWN DISTILLATION
    # -------------------------------------------------------------------------
    
    def distill_blocks(self, tables_data: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
        """
        Convert table blocks to compact prompt format.
        
        Returns:
            (prompt_string, id_map) where id_map maps [B1] -> original block data
        """
        id_map = {}
        prompt_lines = []
        
        for i, table in enumerate(tables_data):
            # Only process tables with markdown content
            if not table.get("markdown_content"):
                continue
            
            block_id = f"B{i+1}"
            id_map[block_id] = table
            
            # Extract caption as context hint
            caption = table.get("caption", "Table")[:100]  # Limit caption length
            markdown = table.get("markdown_content", "")
            
            # Compact format: [ID]: Caption\n```markdown content```
            prompt_lines.append(f"[{block_id}]: {caption}")
            prompt_lines.append(f"```\n{markdown}\n```")
            prompt_lines.append("")  # Separator
        
        return "\n".join(prompt_lines), id_map
    
    # -------------------------------------------------------------------------
    # STRATEGY 3: GLOBAL VS LOCAL CONTEXT
    # -------------------------------------------------------------------------
    
    def extract_global_context(self, paper_data: Dict) -> GlobalContext:
        """
        Extract paper-level metadata (enzyme name, organism, etc.)
        This is done once per paper to avoid repetition in observations.
        """
        ctx = GlobalContext()
        ctx.paper_id = paper_data.get("paper_id", "")
        
        # Try to extract from LLM classification if available
        classification = paper_data.get("llm_classification") or {}
        ctx.enzyme_name = classification.get("enzyme_name", "")
        ctx.organism = classification.get("organism", "")
        
        # Try to find sequence in text content
        text_content = paper_data.get("text_content", "")
        
        # Simple heuristic: look for sequence patterns
        seq_match = re.search(r'[ACDEFGHIKLMNPQRSTVWY]{20,}', text_content)
        if seq_match:
            ctx.seq_aa = seq_match.group(0)
        
        return ctx
    
    # -------------------------------------------------------------------------
    # CORE EXTRACTION (STRATEGY 2: SCHEMA MINIMIZATION)
    # -------------------------------------------------------------------------
    
    def extract_observations(self, distilled_md: str, global_ctx: GlobalContext) -> Dict:
        """
        Send distilled markdown to Gemini and extract structured data.
        Uses compact JSON schema for token efficiency.
        """
        if not self.model:
            return {"activity": [], "tm": []}
        
        # Build system prompt with compact schema
        system_prompt = """You are a scientific data transcriber. Extract experimental measurements from the provided tables.

OUTPUT FORMAT (compact JSON):
{
  "g": {"e": "enzyme_name", "o": "organism", "s": "substrate"},
  "act": [
    {"v": 45.2, "u": "U/mg", "pH": 7.0, "T": 37, "t": 1.0, "b": "B1"},
    ...
  ],
  "tm": [
    {"v": 65.0, "u": "C", "m": "DSC", "b": "B2"},
    ...
  ]
}

RULES:
1. Do NOT convert units. Transcribe exactly as written.
2. Use block IDs [B1], [B2] in "b" field for traceability.
3. "g" = global context (enzyme, organism, substrate) - only if found.
4. "act" = activity observations (v=value, u=unit, pH, T=temperature, t=time_h, b=block).
5. "tm" = melting temperature observations (v=value, u=unit, m=method, b=block).
6. Skip empty or non-numeric values.

OUTPUT ONLY THE JSON. No explanation."""

        # Build user prompt
        user_prompt = f"""Context: Paper studying {global_ctx.enzyme_name or 'unknown enzyme'} from {global_ctx.organism or 'unknown organism'}.

Tables to extract:
{distilled_md}

Extract all numerical experimental data."""

        try:
            response = self.model.generate_content(
                [system_prompt, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            result = json.loads(response.text)
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini extraction failed: {e}")
            return {"activity": [], "tm": []}
    
    # -------------------------------------------------------------------------
    # INFLATE & NORMALIZE
    # -------------------------------------------------------------------------
    
    def inflate_results(
        self, 
        compact_json: Dict, 
        id_map: Dict[str, Dict],
        global_ctx: GlobalContext
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Expand compact JSON to full schema DataFrames.
        Apply unit normalization where possible.
        """
        # Extract global overrides from Gemini response
        g = compact_json.get("g", {})
        enzyme = g.get("e") or global_ctx.enzyme_name
        organism = g.get("o") or global_ctx.organism
        substrate = g.get("s") or global_ctx.substrate
        
        # ---- ACTIVITY DATA ----
        activity_rows = []
        for obs in compact_json.get("act", []):
            row = {
                "paper_id": global_ctx.paper_id,
                "enzyme_name": enzyme,
                "organism": organism,
                "seq_aa": global_ctx.seq_aa,
                "substrate": substrate,
                "activity_value": obs.get("v", 0.0),
                "activity_unit": obs.get("u", ""),
                "activity_normalized": self._normalize_activity(obs.get("v"), obs.get("u")),
                "ph": obs.get("pH", 0.0),
                "temperature_c": self._normalize_temperature(obs.get("T"), "C"),
                "time_h": obs.get("t", 0.0),
                "buffer": global_ctx.buffer,
                "reaction_volume_ml": obs.get("vol", 0.0),
                "enzyme_loading_value": obs.get("enz_v", 0.0),
                "enzyme_loading_unit": obs.get("enz_u", ""),
                "substrate_amount_value": obs.get("sub_v", 0.0),
                "substrate_amount_unit": obs.get("sub_u", ""),
            }
            activity_rows.append(row)
        
        # ---- TM DATA ----
        tm_rows = []
        for obs in compact_json.get("tm", []):
            row = {
                "paper_id": global_ctx.paper_id,
                "enzyme_name": enzyme,
                "organism": organism,
                "seq_aa": global_ctx.seq_aa,
                "tm_value": obs.get("v", 0.0),
                "tm_unit": obs.get("u", "C"),
                "method": obs.get("m", ""),
                "buffer": global_ctx.buffer,
                "notes": obs.get("n", ""),
            }
            tm_rows.append(row)
        
        # Create DataFrames with correct column order
        df_activity = pd.DataFrame(activity_rows, columns=ACTIVITY_COLUMNS) if activity_rows else pd.DataFrame(columns=ACTIVITY_COLUMNS)
        df_tm = pd.DataFrame(tm_rows, columns=TM_COLUMNS) if tm_rows else pd.DataFrame(columns=TM_COLUMNS)
        
        return df_activity, df_tm
    
    # -------------------------------------------------------------------------
    # NORMALIZATION HELPERS
    # -------------------------------------------------------------------------
    
    def _normalize_activity(self, value: float, unit: str) -> float:
        """Normalize activity to U/mg if possible."""
        if not value or not unit:
            return 0.0
        
        config = UNITS_CONFIG.get("activity_units", {}).get(unit)
        if config and config.get("factor"):
            return value * config["factor"]
        return value  # Return raw if can't normalize
    
    def _normalize_temperature(self, value: float, unit: str) -> float:
        """Normalize temperature to Celsius."""
        if not value:
            return 0.0
        
        config = UNITS_CONFIG.get("temperature_units", {}).get(unit)
        if config:
            return value * config.get("factor", 1.0) + config.get("offset", 0)
        return value
    
    def calculate_molecular_weight(self, seq_aa: str) -> float:
        """Calculate MW from amino acid sequence."""
        if not seq_aa:
            return 0.0
        
        aa_weights = UNITS_CONFIG.get("amino_acid_weights", {})
        water_loss = UNITS_CONFIG.get("water_loss_per_bond", 18.015)
        
        total = sum(aa_weights.get(aa.upper(), 0) for aa in seq_aa)
        # Subtract water for peptide bonds
        total -= (len(seq_aa) - 1) * water_loss
        
        return round(total, 2)
    
    # -------------------------------------------------------------------------
    # MAIN EXTRACTION PIPELINE
    # -------------------------------------------------------------------------
    
    def extract_paper(self, paper_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: Extract structured data from a paper.
        
        Returns:
            (df_activity, df_tm) DataFrames
        """
        # 1. Extract global context
        global_ctx = self.extract_global_context(paper_data)
        
        # 2. Distill tables to compact format
        tables = paper_data.get("tables_data", [])
        enriched_tables = [t for t in tables if t.get("markdown_content")]
        
        if not enriched_tables:
            logger.warning(f"⚠️ No enriched tables in {global_ctx.paper_id}")
            return pd.DataFrame(columns=ACTIVITY_COLUMNS), pd.DataFrame(columns=TM_COLUMNS)
        
        distilled_md, id_map = self.distill_blocks(enriched_tables)
        
        # 3. Extract observations via Gemini
        compact_json = self.extract_observations(distilled_md, global_ctx)
        
        # 4. Inflate to full schema
        df_activity, df_tm = self.inflate_results(compact_json, id_map, global_ctx)
        
        logger.info(f"✅ Extracted {len(df_activity)} activity + {len(df_tm)} Tm observations from {global_ctx.paper_id}")
        
        return df_activity, df_tm


# ============================================================================
# STREAMING INTERFACE FOR ENZYME_PARSER
# ============================================================================

def extract_existing_papers_streaming(output_root: Path):
    """
    Generator function to extract data from all enriched papers.
    Yields log messages for UI streaming.
    """
    extractor = GeminiExtractor()
    
    if not extractor.model:
        yield "❌ Cannot run extraction: Gemini API not configured"
        return
    
    paper_dirs = [d for d in output_root.iterdir() if d.is_dir()]
    yield f"📁 Found {len(paper_dirs)} papers. Scanning for enriched tables..."
    
    all_activity = []
    all_tm = []
    extracted_count = 0
    
    for i, paper_dir in enumerate(paper_dirs, 1):
        paper_id = paper_dir.name
        json_path = paper_dir / f"{paper_id}_data.json"
        
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            
            # Check if has enriched tables
            enriched = [t for t in paper_data.get("tables_data", []) if t.get("markdown_content")]
            if not enriched:
                continue
            
            yield f"\n--- [{i}/{len(paper_dirs)}] Extracting: {paper_id} ---"
            yield f"  🧪 {len(enriched)} enriched tables found..."
            
            df_act, df_tm = extractor.extract_paper(paper_data)
            
            if not df_act.empty:
                all_activity.append(df_act)
                yield f"  ✅ Activity: {len(df_act)} rows"
            
            if not df_tm.empty:
                all_tm.append(df_tm)
                yield f"  ✅ Tm: {len(df_tm)} rows"
            
            # Save per-paper CSVs
            if not df_act.empty:
                df_act.to_csv(paper_dir / f"{paper_id}_activity.csv", index=False)
            if not df_tm.empty:
                df_tm.to_csv(paper_dir / f"{paper_id}_tm.csv", index=False)
            
            extracted_count += 1
            
        except Exception as e:
            yield f"❌ Error extracting {paper_id}: {e}"
    
    # Save master CSVs
    if all_activity:
        master_act = pd.concat(all_activity, ignore_index=True)
        master_act.to_csv(output_root / "master_activity.csv", index=False)
        yield f"\n📊 Master Activity CSV: {len(master_act)} total rows"
    
    if all_tm:
        master_tm = pd.concat(all_tm, ignore_index=True)
        master_tm.to_csv(output_root / "master_tm.csv", index=False)
        yield f"📊 Master Tm CSV: {len(master_tm)} total rows"
    
    yield f"\n✨ Extraction complete! Processed {extracted_count} papers."
