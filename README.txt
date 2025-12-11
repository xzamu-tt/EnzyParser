================================================================================
                            EnzyParser v1.0
         Batch Scientific Paper Processor using Docling
================================================================================

DESCRIPCIÓN - ENZYPARSER
-------------   
-----
    -----------

    FLJSDLFKJADSF
    

EnzyParser es un procesador batch de papers científicos enfocado en 
investigación enzimática. Utiliza la librería Docling de IBM para extraer
texto, tablas y figuras de PDFs y archivos suplementarios.

Características principales:
  • Procesamiento masivo de directorios (~120+ papers)
  • Lógica basada en nombre de carpeta = nombre de PDF principal
  • Extracción de TODAS las imágenes (alta recuperación para filtrado posterior)
  • Coordenadas exactas (bounding boxes) para trazabilidad
  • Soporte para archivos suplementarios (PDF, Excel, CSV, imágenes TIF/PNG/JPG)


================================================================================
ESTRUCTURA DEL PROYECTO
================================================================================

EnzyParser/
├── .venv/                    # Entorno virtual de Python
├── .gemini/
│   └── settings.json         # Configuración del servidor MCP (docling)
├── requirements.txt          # Dependencias del proyecto
├── enzyme_parser.py          # Módulo principal del parser
├── NEWarticles/              # ENTRADA: Carpetas con papers (tú la creas)
│   ├── Almeida-2019/
│   │   ├── Almeida-2019.pdf  # Paper principal (DEBE coincidir con carpeta)
│   │   ├── supp_data.pdf     # Opcional: PDFs suplementarios
│   │   ├── data.xlsx         # Opcional: Datos en Excel
│   │   └── western_blot.tif  # Opcional: Imágenes adicionales
│   ├── Chen-2020/
│   │   └── Chen-2020.pdf
│   └── ...
└── Processed_Enzyme_Data/    # SALIDA: Datos procesados (se genera automáticamente)
    ├── Almeida-2019/
    │   ├── Almeida-2019_data.json
    │   └── artifacts/
    │       ├── MAIN_Almeida-2019_p3_fig0.png
    │       ├── SUPP_supp_data_p1_fig0.png
    │       └── SUPP_IMG_western_blot.png
    └── ...


================================================================================
INSTALACIÓN
================================================================================

1. CLONAR/DESCARGAR EL PROYECTO
   Asegúrate de estar en el directorio del proyecto:
   
   cd /Users/xzamu/Projects/EnzyParser

2. CREAR ENTORNO VIRTUAL (si no existe)
   
   python3 -m venv .venv

3. ACTIVAR ENTORNO VIRTUAL
   
   En macOS/Linux:
   source .venv/bin/activate
   
   En Windows:
   .venv\Scripts\activate.bat

4. INSTALAR DEPENDENCIAS
   
   pip install -r requirements.txt
   
   Esto instalará:
   - docling (>=2.64.0)  : Motor de extracción de PDFs
   - pydantic (>=2.0)    : Modelos de datos
   - pandas (>=2.0)      : Manejo de Excel/CSV
   - pillow (>=10.0)     : Procesamiento de imágenes
   - openpyxl (>=3.1)    : Soporte para Excel .xlsx


================================================================================
PREPARAR DATOS DE ENTRADA
================================================================================

REGLA CRÍTICA:
El nombre del PDF principal DEBE coincidir EXACTAMENTE con el nombre de la 
carpeta que lo contiene.

CORRECTO:
  NEWarticles/
  └── Almeida-2019/
      └── Almeida-2019.pdf    ✓ Los nombres coinciden

INCORRECTO:
  NEWarticles/
  └── Almeida-2019/
      └── paper.pdf           ✗ No coincide, será ignorado

ARCHIVOS SUPLEMENTARIOS:
Cualquier otro archivo en la carpeta (que no sea el PDF principal) será 
tratado como material suplementario:

  • *.pdf          → Se procesa con Docling (extrae figuras y tablas)
  • *.xlsx, *.xls  → Se lee y convierte a JSON
  • *.csv          → Se lee y convierte a JSON
  • *.jpg, *.png   → Se copia a artifacts/
  • *.tif, *.tiff  → Se convierte a PNG y copia a artifacts/


================================================================================
EJECUCIÓN
================================================================================

1. ACTIVAR ENTORNO VIRTUAL (si no está activo)
   
   source .venv/bin/activate

2. EJECUTAR EL PARSER
   
   python enzyme_parser.py

3. MONITOREAR PROGRESO
   El script mostrará logs en consola:
   
   2024-12-10 16:30:00 - INFO - Found 120 paper directories to process.
   2024-12-10 16:30:01 - INFO - Processing: Almeida-2019
   2024-12-10 16:30:05 - INFO -   -> Supplementary: supp_data.pdf
   2024-12-10 16:30:08 - INFO - ✅ Completed Almeida-2019. Output: Processed_Enzyme_Data/Almeida-2019


================================================================================
ESTRUCTURA DE SALIDA
================================================================================

Para cada paper procesado se genera:

1. ARCHIVO JSON (PaperID_data.json)
   Contiene toda la información extraída:
   
   {
     "paper_id": "Almeida-2019",
     "original_pdf": "/path/to/Almeida-2019.pdf",
     "metadata": {
       "title": "...",
       "authors": "..."
     },
     "text_content": "# Título\n\nTexto completo en Markdown...",
     "artifacts": [
       {
         "id": "fig_0",
         "type": "figure",
         "file_path": "/path/to/MAIN_Almeida-2019_p3_fig0.png",
         "page_no": 3,
         "bbox": [100.5, 200.3, 500.0, 450.2],
         "caption": "Figure 1. Enzyme kinetics...",
         "vlm_description": "No VLM analysis",
         "source_file": "Almeida-2019.pdf"
       }
     ],
     "tables_data": [
       {
         "source": "Almeida-2019.pdf",
         "page": 5,
         "data": [{"Column1": "value1", "Column2": "value2"}]
       }
     ],
     "supplementary_files_processed": ["supp_data.pdf", "data.xlsx"]
   }

2. CARPETA artifacts/
   Contiene todas las imágenes extraídas:
   
   Nomenclatura:
   - MAIN_*          : Figuras del paper principal
   - SUPP_*          : Figuras de PDFs suplementarios
   - SUPP_IMG_*      : Imágenes sueltas (TIF, JPG, PNG originales)


================================================================================
CONFIGURACIÓN AVANZADA
================================================================================

CAMBIAR DIRECTORIOS DE ENTRADA/SALIDA:
Edita las variables al final de enzyme_parser.py:

   if __name__ == "__main__":
       INPUT_DIR = "./NEWarticles"           # Cambiar aquí
       OUTPUT_DIR = "./Processed_Enzyme_Data"  # Cambiar aquí
       ...

HABILITAR VLM (Vision Language Model):
Si tienes un servidor VLM local (LM Studio, Ollama, vLLM), puedes habilitar
descripciones automáticas de las figuras:

   LOCAL_VLM_URL = "http://localhost:1234/v1/chat/completions"

Esto añadirá descripciones automáticas a cada figura extraída.


================================================================================
MODELOS DE DATOS (PYDANTIC)
================================================================================

VisualArtifact:
  - id: str              Identificador único (ej: "fig_0")
  - type: str            Tipo: 'figure', 'table_image', 'supplementary_image'
  - file_path: str       Ruta local al archivo PNG extraído
  - page_no: int         Número de página donde se encontró
  - bbox: List[float]    Coordenadas [left, top, right, bottom]
  - caption: str         Pie de figura del PDF
  - vlm_description: str Descripción generada por VLM (si está habilitado)
  - source_file: str     Nombre del archivo origen

PaperData:
  - paper_id: str                       Identificador del paper
  - original_pdf: str                   Ruta al PDF original
  - metadata: Dict                      Metadatos extraídos
  - text_content: str                   Texto completo en Markdown
  - artifacts: List[VisualArtifact]     Lista de figuras/imágenes
  - tables_data: List[Dict]             Tablas extraídas
  - supplementary_files_processed: List[str]  Archivos suplementarios procesados


================================================================================
SOLUCIÓN DE PROBLEMAS
================================================================================

ERROR: "SKIPPING: Main PDF not found"
  → El nombre del PDF no coincide con el nombre de la carpeta.
  → Solución: Renombra el PDF para que coincida exactamente.

ERROR: "Docling conversion failed"
  → El PDF puede estar corrupto o protegido.
  → Prueba abrir el PDF manualmente para verificar.

ERROR al instalar dependencias:
  → Asegúrate de tener Python 3.10+ instalado.
  → En macOS, puede requerir: xcode-select --install

IMÁGENES NO SE EXTRAEN:
  → Verifica que el PDF tenga imágenes embebidas (no solo texto).
  → Algunos PDFs escaneados requieren OCR adicional.


================================================================================
PRÓXIMOS PASOS SUGERIDOS
================================================================================

1. Ejecutar con un paper de prueba para validar la extracción.

2. Revisar los artifacts/ y verificar que las figuras se extrajeron bien.

3. Usar el JSON generado para alimentar un modelo SOTA como Gemini 1.5 Pro
   para análisis más profundo de las figuras (filtrado inteligente).

4. Las coordenadas bbox permiten pintar recuadros sobre el PDF original
   para visualizar la procedencia exacta de cada figura.


================================================================================
DEPENDENCIAS
================================================================================

docling>=2.64.0      - Motor de parsing de documentos (IBM Research)
pydantic>=2.0        - Validación y serialización de datos
pandas>=2.0          - Manipulación de datos tabulares
pillow>=10.0         - Procesamiento de imágenes
openpyxl>=3.1        - Lectura de archivos Excel (.xlsx)


================================================================================
LICENCIA Y CRÉDITOS
================================================================================

EnzyParser utiliza Docling, una librería open-source de IBM Research.
https://github.com/docling-project/docling

Para más información sobre Docling:
- Documentación: https://docling-project.github.io/docling/
- Paper técnico: https://arxiv.org/abs/2408.09869


================================================================================
                              ¡Listo para procesar!
================================================================================
