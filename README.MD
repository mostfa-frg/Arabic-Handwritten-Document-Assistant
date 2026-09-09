# Arabic Handwritten Document Assistant

Streamlit is the primary frontend. It uploads each image once to FastAPI `/ocr`,
keeps the OCR text in Streamlit session state, and sends subsequent questions to
`/query` with that extracted text. The legacy Gradio frontend remains available
but is no longer the recommended workflow.

## Run locally

Set `API_BASE_URL` when the FastAPI server is not running at its default:

```dotenv
API_BASE_URL=http://127.0.0.1:8000
```

Run the API and frontend in separate terminals:

```powershell
uvicorn src.api.main:app --reload
streamlit run src/frontend/streamlit_app.py
```