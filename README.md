# IndicRAG: Hindi PDF Question Answering

MegaMindz Hindi RAG Assignment (IndicRAG) extracts text from the supplied 22-page Hindi PDF, normalizes it, chunks each page with page and heading metadata, embeds Hindi and English with a multilingual Sentence Transformers model, and stores vectors locally in ChromaDB. Answers use retrieved text and include page, section, and chunk citations. The default answer mode is deterministic and extractive; Gemini generation is optional.

The ingestion pipeline also indexes the supplemental `.txt` files in `data/raw`, except `agni_ki_udaan_hi.txt` because that file is a separate text surrogate for the PDF. Evaluation targets and citations refer to the actual PDF pages.

## Requirements

- Windows 10 or 11
- Python 3.9 or newer (Python 3.10 or 3.11 is recommended for a fresh installation)
- Internet access for installing packages and downloading the embedding model on first use
- The supplied `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` in the project root, or in `data/raw`

The first embedding run downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` from Hugging Face. No hosted generation API is required for the default mode.

## Install on Windows

Open PowerShell in the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
```

If PowerShell blocks environment activation, either allow activation for the current process with `Set-ExecutionPolicy -Scope Process Bypass`, or run the commands directly with `.\.venv\Scripts\python.exe` instead of activating the environment.

To use the optional Gemini answer mode, copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and keep the key private. The Gemini package is installed by `requirements.txt`; without a key, the local extractive generator is used.

## Ingest the PDF

The assignment PDF is not included in the staged GitHub files. Copy `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` into the project root (next to `run.py`) or into `data/raw/`. The ingestion command checks both locations.

From the project directory:

```powershell
python run.py ingest
```

The successful output reports the loaded source count, extracted page count, chunk count, embedding model, and stored vector count. The database is regenerated under `chroma_db/`. A failed extraction does not clear an existing index. Image-only/scanned PDFs are not OCR'd and produce an actionable extraction error.

The standalone equivalent is `python ingest.py`.

## Ask Questions

```powershell
python run.py query "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया?" --top-k 3
python run.py query "Which institute did Kalam attend for aeronautical engineering?" --top-k 3
```

The output includes the retrieved page, heading, chunk ID, similarity score, answer, and citations. The answer text may remain Hindi for an English question because the default generator quotes source passages instead of translating them.

## Evaluation

Run the fixed small benchmark against the live Chroma index:

```powershell
python run.py evaluate
```

The benchmark has Hindi and English questions and an unsupported Chandrayaan-3 question. A supported hit requires both an expected PDF page and at least one evidence phrase from that page. Recall@k and MRR use supported queries only; citation accuracy checks that the inline citation matches a retrieved chunk and that the chunk contains expected page evidence. Unsupported refusal accuracy is reported separately. These are small assignment checks, not a general quality estimate.

Run local tests without downloading the embedding model:

```powershell
python -m unittest discover -s tests -v
```

The test set and expected evidence are in `data/eval_dataset.json`. In the current local run, retrieval recall@1/3/5 was 40%/80%/80%, citation accuracy was 80%, and the unsupported-query refusal accuracy was 0%. In particular, the unsupported Hindi Chandrayaan-3 case was not refused, and the five-missile English case missed its expected evidence. These measured results are disclosed as current limitations, not guaranteed results; rerun evaluation after changing the index, model, or code.

## Run the API

```powershell
python run.py serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. FastAPI documentation is available at `http://127.0.0.1:8000/docs`. The API provides `POST /api/ask`, `POST /api/ingest`, `GET /api/evaluate`, and `GET /api/stats`. Ask and evaluate return HTTP 409 with an ingestion instruction when the index is empty; `/api/ask` accepts `top_k` from 1 through 20.

## Test Queries and Expected Evidence

The test cases below were checked against text extracted from the supplied PDF. They state evidence targets, not pre-recorded model results.

| Language | Question focus | Expected PDF page and evidence |
| --- | --- | --- |
| English | Institution for aeronautical engineering | Page 6; Madras Institute of Technology (मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी) |
| Hindi | SLV-III satellite success | Page 8; Rohini satellite and July 1980 |
| Hindi | HAL prototype project | Page 7; hovercraft prototype (होवरक्राफ्ट, प्रोटोटाइप) |
| English | IGMDP missile projects | Page 9; Prithvi, Agni, Trishul, Akash, and Nag |
| Hindi | Dhawan's response to the 1979 failure | Pages 8 and 13; Satish Dhawan, responsibility, and leadership |
| Hindi | Chandrayaan-3 landing date | Unsupported by this PDF; the expected behavior is an uncited refusal |

Page values are 1-based PDF page positions, not the printed page number inside the booklet. Section metadata is the first non-empty extracted line on that PDF page, with a leading printed page number removed when present.

## Configuration

Copy `.env.example` to `.env` to override the defaults:

- `CHROMA_PERSIST_DIRECTORY`: local Chroma database path
- `CHROMA_COLLECTION_NAME`: collection name
- `EMBEDDING_MODEL_NAME`: Sentence Transformers model
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: chunker character limits
- `CONFIDENCE_THRESHOLD`: minimum cosine similarity used for confidence gating
- `GEMINI_API_KEY`: optional hosted generation key

The vector database, virtual environments, `.env` files, Python caches, and wheel files are ignored by Git. Keep `.env.example` as the non-secret configuration template. Check permission to redistribute the supplied PDF and supplemental source documents before publishing them.

## Troubleshooting and Limitations

- `ChromaDB is empty`: run `python run.py ingest` from the project directory and confirm the supplied PDF path.
- `PyMuPDF is not installed`: activate `.venv` and run `python -m pip install -r requirements.txt`.
- First query is slow or fails while loading the model: confirm internet access, then retry after the Hugging Face model download completes.
- Hindi text looks garbled in PowerShell: set `$env:PYTHONUTF8 = "1"` for the current terminal and run the CLI again.
- `No searchable text was extracted`: the current pipeline uses PDF text extraction only; it does not perform OCR.
- Similarity thresholds are heuristic and are not calibrated probabilities. Always inspect the returned retrieval trace and citations.
- Extractive answers quote the source and do not translate or synthesize complex multi-part reasoning. Optional Gemini output is accepted only when it contains valid retrieved-chunk citations, but citation checks cannot prove the semantic correctness of generated wording.
- The benchmark is small and deterministic in its scoring rules, but live retrieval results can vary with model/package versions and index contents.