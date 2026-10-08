import os
import json

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from llm import LLMError, analyze_resume
from matcher import compute_match_score
from pdf_utils import PDFError, extract_text_from_pdf

load_dotenv()  # reads ANTHROPIC_API_KEY / ANTHROPIC_MODEL from .env

# Support Streamlit Cloud Secrets fallback
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    if "ANTHROPIC_MODEL" in st.secrets:
        os.environ["ANTHROPIC_MODEL"] = st.secrets["ANTHROPIC_MODEL"]
except Exception:
    pass

st.set_page_config(page_title="AI Resume / JD Matcher", page_icon="📄", layout="wide")



@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    """Cached: the model loads once per server, not on every button click."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def render_list(items: list[str], empty_msg: str) -> None:
    if not items:
        st.caption(empty_msg)
    for item in items:
        st.markdown(f"- {item}")


st.title("📄 AI Resume / JD Matching Assistant")
st.caption("Upload your resume, add a job description, and get a match score with AI feedback.")

left, right = st.columns(2)

with left:
    st.subheader("1. Resume")
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"], key="resume")

with right:
    st.subheader("2. Job Description")
    jd_mode = st.radio("JD input", ["Paste text", "Upload PDF"], horizontal=True)
    jd_pasted, jd_file = "", None
    if jd_mode == "Paste text":
        jd_pasted = st.text_area("Paste the JD here", height=200)
    else:
        jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd")

if st.button("Analyze match", type="primary"):
    # ---- 1. Validate inputs ----
    if resume_file is None:
        st.error("Please upload your resume PDF.")
        st.stop()
    if jd_mode == "Paste text" and not jd_pasted.strip():
        st.error("Please paste the job description.")
        st.stop()
    if jd_mode == "Upload PDF" and jd_file is None:
        st.error("Please upload the job description PDF.")
        st.stop()

    # ---- 2. Extract text ----
    try:
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file) if jd_mode == "Upload PDF" else jd_pasted.strip()
    except PDFError as exc:
        st.error(str(exc))
        st.stop()

    # ---- 3. Score ----
    with st.spinner("Computing match score..."):
        score = compute_match_score(load_model(), resume_text, jd_text)

    st.divider()
    st.subheader("Overall Match Score")
    st.progress(int(score))
    st.metric("Semantic similarity", f"{score}%")

    # ---- 4. LLM analysis ----
    try:
        with st.spinner("Analyzing with AI..."):
            result = analyze_resume(resume_text, jd_text, score)
    except LLMError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:  # last safety net so the app never crashes
        st.error("Something unexpected went wrong. Please try again.")
        st.stop()

    # ---- 5. Show results ----
    st.info(result["summary"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Matched skills")
        render_list(result["matched_skills"], "No matched skills found.")
    with col_b:
        st.subheader("❌ Missing skills")
        render_list(result["missing_skills"], "No missing skills found.")

    st.subheader("💪 Strengths")
    render_list(result["strengths"], "No strengths listed.")

    st.subheader("🛠 Improvement suggestions")
    render_list(result["improvement_suggestions"], "No suggestions.")

    with st.expander("Raw JSON"):
        st.code(json.dumps(result, indent=2), language="json")
