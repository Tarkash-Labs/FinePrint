import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed: %s", exc)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("JSON extraction failed. Raw response: %s", cleaned[:2000])
            raise
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as inner_exc:
            logger.warning("JSON extraction decode failed: %s", inner_exc)
            logger.warning("JSON extraction raw response: %s", cleaned[:2000])
            raise


def extract_json_any(text: str) -> list | dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            s, e = cleaned.find(sc), cleaned.rfind(ec)
            if s != -1 and e > s:
                try:
                    return json.loads(cleaned[s : e + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning("extract_json_any failed: %s", cleaned[:300])
        raise


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
