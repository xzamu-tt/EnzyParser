import os
import base64
import asyncio
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Try importing Mistral, handle failure gracefully if not installed yet
try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

logger = logging.getLogger("EnzyParser")

class MistralEnricher:
    """
    Motor de enriquecimiento utilizando Mistral OCR para tablas.
    Procesa imágenes de tablas y extrae contenido Markdown estructurado.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_concurrency: int = 2):
        """
        Args:
            api_key: MISTRAL_API_KEY. Si es None, busca en variables de entorno.
            max_concurrency: Número máximo de peticiones simultáneas (Rate Limit).
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            logger.warning("⚠️ MISTRAL_API_KEY no encontrada. El enriquecimiento de tablas será saltado.")
            self.client = None
        else:
            if Mistral:
                self.client = Mistral(api_key=self.api_key)
                logger.info("✅ Mistral Client inicializado.")
            else:
                logger.error("❌ Librería `mistralai` no instalada. Ejecuta `pip install mistralai`.")
                self.client = None
                
        # Semáforo para controlar concurrencia (evitar 429 Too Many Requests)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _encode_image(self, image_path: str) -> str:
        """Convierte la imagen a Base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error codificando imagen {image_path}: {e}")
            return ""

    async def enrich_table(self, table_data: Dict) -> Dict:
        """
        Procesa una única tabla. Retorna la tabla actualizada con 'markdown_content'.
        
        Logica:
        1. Verifica si tiene 'image_path' y es 'has_quantitative_data'.
        2. Codifica imagen.
        3. Envía a Mistral con Caption como hint.
        """
        # Validaciones básicas
        if not self.client: 
            return table_data
            
        if not table_data.get("has_quantitative_data"):
            return table_data
            
        image_path = table_data.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return table_data
            
        # Si ya tiene contenido (por rerun), no gastar tokens
        if table_data.get("markdown_content"):
            return table_data

        async with self.semaphore:
            try:
                # 1. Preparar Base64
                base64_img = self._encode_image(image_path)
                if not base64_img:
                    return table_data
                
                # 2. Construir Prompt / Request
                # Usamos el caption como contexto para ayudar al OCR con abreviaturas
                caption_hint = table_data.get("caption", "")
                table_id = table_data.get("id", "unknown")
                
                logger.info(f"🚀 Enviando Tabla {table_id} a Mistral OCR...")
                
                # Wrapper para ejecutar llamada síncrona en hilo asíncrono si el cliente es síncrono
                # (La SDK de Python v1 suele ser síncrona o async dependiendo de uso, asumiremos sync wrappeada)
                # OJO: La SDK v1.0+ tiene clientes async? 
                # El ejemplo del usuario usa `await client.ocr.process`, implicando async nativo o wrapper.
                # Asumiremos que client.ocr.process es una coroutina o thread-safe.
                
                # Nota: Si mistralai python client es síncrono, usamos run_in_executor
                loop = asyncio.get_event_loop()
                
                def _do_request():
                    return self.client.ocr.process(
                        model="mistral-ocr-latest",
                        document={
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_img}"
                        },
                        include_image_base64=False 
                        # extract_header/footer no parecen params standard en docs python recientes, 
                        # pero seguiremos el ejemplo del usuario si aplica, o defaults seguros.
                    )

                # Ejecutar request
                ocr_response = await loop.run_in_executor(None, _do_request)
                
                # 3. Extraer Markdown
                # La respuesta suele tener .markdown o .pages[0].markdown
                # Ajustar según estructura real de respuesta de Mistral OCR API
                if hasattr(ocr_response, 'pages'):
                    # Concatenar todas las páginas (aunque es una imagen, suele ser 1 página)
                    md_result = "\n\n".join([p.markdown for p in ocr_response.pages])
                elif hasattr(ocr_response, 'markdown'):
                    md_result = ocr_response.markdown
                else:
                    md_result = str(ocr_response) # Fallback

                # 4. Guardar en estructura
                table_data["markdown_content"] = md_result
                logger.info(f"✅ Tabla {table_id} enriquecida ({len(md_result)} chars)")

            except Exception as e:
                logger.error(f"❌ Error Mistral OCR en tabla {table_data.get('id')}: {e}")
        
        return table_data

    async def process_paper_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Procesa una lista de tablas en batch (controlado por semáforo).
        Maneja la lógica de contigüidad (Tablas partidas) antes o después del OCR.
        
        Estrategia de Fase 3 (Contigüidad):
        Para simplificar y robustecer, procesaremos OCR de TODAS las partes primero.
        LUEGO, en un paso de post-procesamiento, uniremos los Markdowns.
        """
        if not self.client or not tables:
            return tables

        tasks = []
        for table in tables:
            tasks.append(self.enrich_table(table))
        
        # Ejecutar todas las tareas (limitadas por el semáforo interno)
        enriched_tables = await asyncio.gather(*tasks)
        
        # --- Fase 3: Post-OCR Merge Logic ---
        # Si detectamos tablas contiguas (Mismo caption o 'continued'), unimos sus Markdowns
        enriched_tables = self._merge_contiguous_tables(enriched_tables)
        
        return enriched_tables

    def _merge_contiguous_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Une tablas que parecen ser la continuación de la anterior.
        Heurística:
        - Tabla N está en Pág X
        - Tabla N+1 está en Pág X+1
        - Tabla N+1 tiene mismo Caption o Caption vacío o contiene "continued"
        """
        if not tables:
            return []
            
        merged = []
        skip_indices = set()
        
        # Ordenar por id o página para asegurar secuencia
        # Asumimos que la lista viene ordenada cronológicamente del parser
        sorted_tables = sorted(tables, key=lambda t: (t.get("page", 0), t.get("bbox", [0,0,0,0])[1]))
        
        for i in range(len(sorted_tables)):
            if i in skip_indices:
                continue
                
            current = sorted_tables[i]
            
            # Intentar buscar continuación en el siguiente elemento
            if i + 1 < len(sorted_tables):
                next_t = sorted_tables[i+1]
                
                # Criterios de fusión
                is_next_page = next_t.get("page", 0) == current.get("page", 0) + 1
                curr_caption = (current.get("caption") or "").lower()
                next_caption = (next_t.get("caption") or "").lower()
                
                # Check 1: Mismo caption exacto o muy similar
                same_caption = (len(curr_caption) > 10 and curr_caption == next_caption)
                
                # Check 2: "Continued" en el caption siguiente
                is_continuation = "continued" in next_caption or "con't" in next_caption
                
                if is_next_page and (same_caption or is_continuation):
                    # FUSIONAR
                    logger.info(f"🔗 Fusionando Tabla {current.get('id')} con {next_t.get('id')}")
                    
                    # Unir Markdown
                    md1 = current.get("markdown_content") or ""
                    md2 = next_t.get("markdown_content") or ""
                    
                    # Estrategia simple: Concatenar con salto
                    # Idealmente Mistral podría limpiar headers repetidos, pero concat es seguro por ahora
                    if md1 and md2:
                        combined_md = f"{md1}\n\n{md2}"
                        current["markdown_content"] = combined_md
                        
                        # Marcar el siguiente para saltarlo
                        skip_indices.add(i + 1)
            
            merged.append(current)
            
        return merged
