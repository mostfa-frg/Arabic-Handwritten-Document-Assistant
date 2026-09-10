# Arabic Handwritten Document Assistant

An end-to-end application for converting handwritten Arabic documents into
searchable text and asking questions about their contents. The project combines
line detection, Arabic handwriting OCR, hybrid retrieval, and a Groq-hosted
language model behind a FastAPI service, with a Streamlit interface for daily
use.

## Features

- Upload handwritten Arabic document images (`PNG`, `JPG`, `JPEG`, `WEBP`, `TIF`,
  and `TIFF`).
- Detect text lines with Kraken and a Muharaf-trained segmentation model.
- Recognize Arabic handwriting with the Muharaf OCR model.
- Ask questions about extracted text or indexed documents.
- Combine semantic vector search and Arabic lexical search for retrieval.
- Use RapidFuzz for fuzzy lexical recovery and optional post-OCR correction
  suggestions.
- Display source, page, and chunk citations when available.
- Review and correct OCR text in the interface before exporting it.
- Download reviewed OCR output as UTF-8 `TXT` or right-to-left `DOCX`.
- Expose the same functionality through a documented FastAPI API.
- Use Streamlit as the primary and only web frontend.

## How it works

```text
Image
  -> document preprocessing
  -> Kraken line segmentation
  -> Muharaf Arabic OCR
  -> editable human review
  -> extracted Arabic text / DOCX
  -> Chroma + Arabic lexical retrieval
  -> Groq LLM answer with sources
```

The Streamlit frontend sends images and questions to FastAPI. OCR and retrieval
services are initialized lazily on the first request, so application startup does
not immediately load every model. The raw OCR text remains editable and is not
silently corrected before human approval.

## Recommended workflow for difficult handwriting

The system is intentionally **human-in-the-loop**. After OCR finishes, the
extracted Arabic text remains editable in the left panel. Correct uncertain words
there, select **Generate downloads from corrected text**, and then download the
reviewed TXT or DOCX file. Questions asked in the assistant use the reviewed text,
so a small amount of correction can also improve the quality of the answers.

This design is important for historical Arabic documents: OCR models learn a
distribution of scripts, image quality, and vocabulary, and cannot reliably
recover a word that is outside that distribution merely because an image is clear.
The application therefore separates automated transcription from the final
user-approved document instead of silently presenting uncertain predictions as
ground truth. The preprocessing pipeline also keeps a fallback path through the
original image when conservative enhancement produces no useful text.

The optional spell-correction helpers in `src/rag/spellcheck.py` are pure,
non-destructive utilities. They are not applied automatically to OCR output;
the raw text shown for review remains unchanged unless a user explicitly adopts
a suggested correction.

## Requirements

- Python 3.10 or newer
- A CPU with enough memory to load the OCR and embedding models
- Internet access for the first model downloads
- A Groq API key for question answering

The recommended OCR path uses CPU inference. GPU support is available in the
line-detector abstraction, but the API currently initializes the recommended
Kraken backend on CPU.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the recommended Kraken segmentation model once:

```powershell
python -m src.cv.download_muharaf_seg_model
```

The Muharaf recognition model is downloaded automatically from Hugging Face
when OCR is first used and cached by the Hugging Face Hub.

## Configuration

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b
API_BASE_URL=http://127.0.0.1:8000
```

`GROQ_API_KEY` is required for `/query` and `/query-image`. `GROQ_MODEL` is
optional and defaults to `qwen/qwen3-32b`. `API_BASE_URL` is used by the
Streamlit frontend and defaults to `http://127.0.0.1:8000`. Answers allow up to
2048 generated tokens by default.

Never commit `.env` or API keys. The repository already ignores environment
files and the local Chroma database.

## Running locally

Start FastAPI in one terminal:

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

Start Streamlit in a second terminal:

```powershell
streamlit run src/frontend/streamlit_app.py --server.port 8502
```

Open the URL printed by Streamlit, usually
`http://localhost:8502`. Upload an image, select **Process Image**, review or
edit the raw OCR result, and then ask questions in the assistant panel. The
deleted Gradio frontend is no longer supported or required.

## API reference

FastAPI provides interactive documentation at
`http://127.0.0.1:8000/docs` while the server is running.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service availability |
| `POST` | `/ocr` | Extract Arabic text from an uploaded image |
| `POST` | `/query` | Answer a question using indexed content and optional OCR text |
| `POST` | `/query-image` | OCR an image and answer a question about it |
| `POST` | `/documents/txt` | Generate a UTF-8 text download |
| `POST` | `/documents/docx` | Generate an Arabic DOCX download |

Example health check:

```powershell
curl http://127.0.0.1:8000/health
```

Example OCR request:

```powershell
curl -X POST http://127.0.0.1:8000/ocr `
  -F "image=@data/test_images/arabic_test.jpg"
```

Example question request:

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question":"ما موضوع هذه الوثيقة؟","ocr_text":"نص الوثيقة المستخرج"}'
```

The API limits uploaded images to 20 MB. Answers are constrained by the
retrieved context; when the context is insufficient, the assistant is
instructed to say so rather than inventing information.

## Document indexing

The persistent Chroma collection is stored in `data/chroma_db` and is created
automatically when the RAG service starts. Documents can be cleaned, chunked,
embedded, and indexed with the helpers in `src/rag/ingestion.py`.

Each indexed chunk stores:

- `source`: the document or file identifier
- `page_id`: the source page
- `chunk_id`: the chunk position

The API does not currently expose a document-ingestion endpoint. Use the
ingestion helpers or the project notebooks to prepare a collection before
asking questions about a document corpus.

## Project structure

```text
src/
├── api/          FastAPI routes, request models, and document exporters
├── cv/           Image preprocessing and text-line detection
├── frontend/     Streamlit application
├── ocr/          Muharaf OCR model integration
└── rag/          Chunking, embeddings, Chroma, retrieval, and answer generation
data/
├── chroma_db/   Local persistent vector store
└── test_images/ Sample and validation images
models/           Local Kraken segmentation model
notebooks/        RAG and OCR experimentation notebooks
```

## Troubleshooting

**`Kraken segmentation model not found`**  
Run `python -m src.cv.download_muharaf_seg_model` from the project root and
confirm that `models/muharaf_seg_best.mlmodel` exists.

**`GROQ_API_KEY is not set`**  
Add `GROQ_API_KEY` to `.env`, restart FastAPI, and submit the request again.

**The first OCR request is slow**  
The Muharaf model, tokenizer, and related dependencies may be downloading or
being loaded into memory. Later requests reuse the loaded service.

**The frontend cannot connect to FastAPI**  
Confirm that Uvicorn is running and that `API_BASE_URL` matches its address.

**OCR output is inaccurate**  
Use a clear, well-lit image with minimal skew and sufficient resolution. Handwriting
style, page quality, segmentation quality, and vocabulary coverage can materially
affect recognition accuracy. For historical documents, review the editable result
before exporting; the reviewed text is the intended final deliverable.

## Limitations

- The application is intended for local or trusted environments; authentication
  and rate limiting are not included.
- OCR quality depends on handwriting style, image quality, segmentation quality,
  and model coverage. It is not a substitute for human verification on historical
  or unusually styled documents.
- Question answering requires a configured Groq-compatible API.
- The default API configuration allows browser requests only from the local
  Streamlit origins.

## License

No license file is currently included. Add a license before distributing the
project or its model integrations.
