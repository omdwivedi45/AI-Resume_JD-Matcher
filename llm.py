"""Google Gemini API call + JSON validation. No Streamlit code here."""
import json
import os
import re

from google import genai
from google.genai import types
from google.genai.errors import APIError

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_CHARS_PER_DOC = 12_000  # ~3k tokens each; keeps cost and latency low

SYSTEM_PROMPT = """You are an expert technical recruiter reviewing a resume against a job description.

RULES:
- Only mention skills, tools and experience that literally appear in the resume or the job description. Do NOT invent skills or experience.
- matched_skills: skills present in BOTH the resume and the JD.
- missing_skills: skills required by the JD that are NOT in the resume.
- strengths: the candidate's strongest points relevant to this JD, based only on the resume.
- improvement_suggestions: concrete, actionable edits to the resume for this JD (without adding fake experience).
- summary: at most 3 sentences. Mention the match score given to you.
- If a list has nothing to report, return an empty list.

Return ONLY a valid JSON object with exactly these keys and nothing else (no markdown, no code fences, no commentary):
{
  "matched_skills": ["..."],
  "missing_skills": ["..."],
  "strengths": ["..."],
  "improvement_suggestions": ["..."],
  "summary": "..."
}"""

REQUIRED_KEYS = {
    "matched_skills": list,
    "missing_skills": list,
    "strengths": list,
    "improvement_suggestions": list,
    "summary": str,
}


class LLMError(Exception):
    """Raised with a user-friendly message when the analysis fails."""


def truncate_text(text: str, limit: int = MAX_CHARS_PER_DOC) -> str:
    """Cut at `limit` characters, backing up to the last space so no word is split."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ...[truncated]"


def parse_and_validate(raw: str) -> dict:
    """Strip code fences, parse JSON, check keys and types. Raises ValueError."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Response is not valid JSON.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Response is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON must be an object.")

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in data:
            raise ValueError(f"Missing key: {key}")
        if not isinstance(data[key], expected_type):
            raise ValueError(f"'{key}' must be a {expected_type.__name__}.")
        if expected_type is list and not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"'{key}' must contain only strings.")

    sentences = re.split(r"(?<=[.!?])\s+", data["summary"].strip())
    data["summary"] = " ".join(sentences[:3])

    return {key: data[key] for key in REQUIRED_KEYS}


def analyze_resume(resume_text: str, jd_text: str, score: float) -> dict:
    """Ask Google Gemini for structured analysis. Retries once on bad JSON."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY not found. Add it to your .env file or Streamlit secrets.")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise LLMError(f"Failed to initialize Gemini client: {exc}")

    user_prompt = (
        f"MATCH SCORE (embedding cosine similarity): {score}%\n\n"
        f"=== RESUME ===\n{truncate_text(resume_text)}\n\n"
        f"=== JOB DESCRIPTION ===\n{truncate_text(jd_text)}"
    )

    last_error = "unknown error"

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text or ""
            return parse_and_validate(raw)
        except APIError as exc:
            raise LLMError(f"Gemini API Error: {exc.message}")
        except ValueError as exc:
            last_error = str(exc)
            user_prompt += f"\n\nPrevious response was invalid JSON ({last_error}). Please strictly return the JSON object."
        except Exception as exc:
            raise LLMError(f"Could not reach Gemini API: {exc}")

    raise LLMError(f"The AI returned an unreadable response twice ({last_error}). Please try again.")
