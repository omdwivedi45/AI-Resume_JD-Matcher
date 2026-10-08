"""Embedding-based match score. No Streamlit code here either."""
from sklearn.metrics.pairwise import cosine_similarity

WORDS_PER_CHUNK = 200  # MiniLM only reads ~256 tokens, so we split long text into chunks
MAX_CHUNKS = 30


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = [
        " ".join(words[i : i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    return chunks[:MAX_CHUNKS] or [""]


def _embed_document(model, text: str):
    """Embed each chunk, then average them into ONE vector (shape 1 x 384)."""
    embeddings = model.encode(_chunk_text(text))
    return embeddings.mean(axis=0, keepdims=True)


def compute_match_score(model, resume_text: str, jd_text: str) -> float:
    """Cosine similarity of resume vs JD embeddings, returned as 0-100."""
    similarity = cosine_similarity(
        _embed_document(model, resume_text),
        _embed_document(model, jd_text),
    )[0][0]
    similarity = max(0.0, min(1.0, float(similarity)))  # clamp to [0, 1]
    return round(similarity * 100, 1)
