"""Anthropic API call + JSON validation. No Streamlit code here."""
import json
import os
import re

import anthropic

DEFAULT_MODEL = "claude-3-5-sonnet-latest"
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
        # Fallback: grab everything between the first { and the last }
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

    # Enforce "max 3 sentences" ourselves instead of trusting the model
    sentences = re.split(r"(?<=[.!?])\s+", data["summary"].strip())
    data["summary"] = " ".join(sentences[:3])

    return {key: data[key] for key in REQUIRED_KEYS}  # drop any extra keys


def analyze_resume(resume_text: str, jd_text: str, score: float) -> dict:
    """Ask the LLM for structured analysis. Retries once on bad JSON."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not found. Add it to your .env file and restart the app.")

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = (
        f"MATCH SCORE (embedding cosine similarity): {score}%\n\n"
        f"=== RESUME ===\n{truncate_text(resume_text)}\n\n"
        f"=== JOB DESCRIPTION ===\n{truncate_text(jd_text)}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    last_error = "unknown error"

    for attempt in range(2):  # first try + one retry
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except anthropic.AuthenticationError:
            raise LLMError("Invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
        except anthropic.RateLimitError:
            raise LLMError("Rate limit reached. Please wait a minute and try again.")
        except anthropic.APIConnectionError:
            raise LLMError("Could not reach the Anthropic API. Check your internet connection.")
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error (status {exc.status_code}). Please try again.")

        raw = "".join(block.text for block in response.content if block.type == "text")
        try:
            return parse_and_validate(raw)
        except ValueError as exc:
            last_error = str(exc)
            # Retry: show the model its bad answer and tell it what was wrong
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw or "(empty)"},
                {
                    "role": "user",
                    "content": f"That response was invalid ({last_error}). "
                    "Return ONLY the JSON object with the 5 required keys.",
                },
            ]

    raise LLMError(f"The AI returned an unreadable response twice ({last_error}). Please try again.")
