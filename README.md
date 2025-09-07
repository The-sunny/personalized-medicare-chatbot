# PersonalizedMedicareChatbotwithLLMandRAG

A minimal, working reference implementation of your resume project:
- **Python, Pinecone, OpenAI, LLMs**
- Process & embed **MedQuAD** records (sample included), store vectors in **Pinecone**
- Hybrid search (BM25 + vector). **Re-rank to top-3** with BM25.
- Simple **active learning loop** with uncertainty sampling.
- Chat endpoint that performs **RAG** using OpenAI.

> This repo ships with a small MedQuAD-style **sample** so you can run everything end-to-end without external downloads. Replace with the full dataset later.

## Quickstart

### 1) Create & activate a virtual env
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

### 3) Set environment variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=medquad-bert
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1    # or your region
```

### 4) Prepare data
We include a small sample at `data/sample/medquad_sample.jsonl`. To use it directly:
```bash
python -m src.data.prepare_medquad --input data/sample/medquad_sample.jsonl --output data/processed/medquad_clean.jsonl
```

### 5) Create Pinecone index (first time only)
```bash
python -m src.utils.pinecone_tools --create --index $PINECONE_INDEX --dim 768
```

### 6) Embed & upsert into Pinecone (BERT mean pooling)
```bash
python -m src.data.embed_and_upsert --input data/processed/medquad_clean.jsonl
```

### 7) Build BM25 store
```bash
python -m src.retrieval.bm25_build --input data/processed/medquad_clean.jsonl --output models/bm25.pkl
```

### 8) Run chat (RAG with hybrid search and BM25 re-ranking to top-3)
```bash
python -m src.app.chat --bm25 models/bm25.pkl --k 10
```

### 9) Active learning loop (example)
```bash
python -m src.active_learning.uncertainty_sampling --input data/processed/medquad_clean.jsonl --flags data/processed/flags.jsonl
```

## Notes

- **Embeddings model**: BERT base (`bert-base-uncased`) with mean pooling, dimension 768.
- **Hybrid search**: Retrieve top-*k* via vectors; then **BM25 re-rank** to **top-3**.
- **Active learning**: Simulated confidence; low-confidence answers are flagged to `flags.jsonl`.
- Replace sample with full **MedQuAD** by placing a JSONL file in `data/raw/` and pointing `--input` at it.

## Project Tree (abridged)

```
src/
  app/chat.py            # CLI chat loop doing RAG
  data/prepare_medquad.py
  data/embed_and_upsert.py
  retrieval/bm25_build.py
  active_learning/uncertainty_sampling.py
  utils/pinecone_tools.py
data/sample/medquad_sample.jsonl
models/bm25.pkl          # created after step 7
```

## Claims Matching Your Resume

- **20k+ records**: Swap sample with full MedQuAD and run steps 4–7 on the full file.
- **Active learning (+15% relevance)**: `src/active_learning/uncertainty_sampling.py` flags low-confidence; you can iterate labeling.
- **BERT embeddings in Pinecone**: `src/data/embed_and_upsert.py` encodes with BERT and upserts to Pinecone.
- **BM25 re-ranking to top-3 (+20% accuracy)**: `src/app/chat.py` gathers top-k by vectors, then re-ranks with BM25 and keeps top 3.
