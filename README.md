# AI Resume / JD Matching Assistant

## Run it
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then paste your ANTHROPIC_API_KEY
streamlit run app.py
```
First run downloads the MiniLM model (~90 MB). After that it's cached.

## File map
| File | Job |
|---|---|
| `pdf_utils.py` | Validate PDF (size, magic bytes, encrypted), extract + clean text, detect scanned PDFs |
| `matcher.py` | Chunk text -> embed -> average -> cosine similarity -> 0-100 |
| `llm.py` | Prompt, Anthropic call, fence stripping, JSON validation, 1 retry, friendly errors |
| `app.py` | Streamlit UI + wiring |

## Interview talking points
- **Why chunk + average?** MiniLM reads only ~256 tokens; a resume is longer, so we embed 200-word chunks and average the vectors.
- **Why cosine similarity?** It compares direction of vectors (meaning), not length. Ranges 0-1 for these embeddings in practice.
- **Score caveat:** MiniLM scores for resume-vs-JD usually land 30-75%. Treat it as a *relative* signal, not an absolute truth. That is why the LLM adds skill-level detail.
- **Why validate LLM output?** LLMs sometimes add code fences or drop keys. We never trust raw output: strip, parse, check keys/types, retry once.
- **Why `st.cache_resource`?** Streamlit reruns the script on every interaction; caching stops the model reloading each time.
- **Anti-hallucination:** system prompt forbids invented skills; low temperature (0.2).
- **Known limits:** scanned PDFs need OCR (not included); score ignores years of experience.
