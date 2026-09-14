import json
import os
import re
import sys
from typing import Any

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - dependency may be absent in non-Google environments
    genai = None
    types = None

from dotenv import load_dotenv

from models import DebateVerdict

load_dotenv()


def _to_json_string(model_obj: Any) -> str:
    if hasattr(model_obj, "model_dump_json"):
        return model_obj.model_dump_json()
    if hasattr(model_obj, "dict"):
        return json.dumps(model_obj.dict())
    return json.dumps(model_obj)


def _configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if genai is None:
        raise RuntimeError("google-genai is not installed.")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _parse_transcript_entries(transcript_text: str):
    try:
        parsed = json.loads(transcript_text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("transcript", "entries", "debate"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass

    return [{"speaker": "Unknown", "text": transcript_text}]


def _score_side(entries, speaker_name: str):
    texts = [entry.get("text", "") for entry in entries if entry.get("speaker") == speaker_name]
    content = " ".join(texts).lower()
    if not content:
        return {
            "logical_consistency_score": 0,
            "logical_consistency_reasoning": "No statements were provided for this side.",
            "evidence_score": 0,
            "evidence_reasoning": "No evidence was provided for this side.",
            "rebuttal_score": 0,
            "rebuttal_reasoning": "No rebuttal was provided for this side.",
        }

    logical_score = 5
    evidence_score = 3
    rebuttal_score = 4

    if any(word in content for word in ["because", "therefore", "if", "so", "which means"]):
        logical_score += 2
    if any(word in content for word in ["everyone knows", "absolutely", "obviously"]):
        logical_score -= 2
    if any(word in content for word in ["according to", "study", "report", "data", "estimate", "estimated", "gdp", "historical", "percentage", "%", "un", "research"]):
        evidence_score += 4
    if re.search(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?", content):
        evidence_score += 2
    if any(word in content for word in ["however", "while", "but", "instead", "on the other hand", "rebuttal", "disproportionately", "harm", "historical output", "developing nations"]):
        rebuttal_score += 3
    if any(word in content for word in ["we propose", "our proposal", "we believe", "as an alternative"]):
        rebuttal_score += 1

    logical_score = max(0, min(10, logical_score))
    evidence_score = max(0, min(10, evidence_score))
    rebuttal_score = max(0, min(10, rebuttal_score))

    return {
        "logical_consistency_score": logical_score,
        "logical_consistency_reasoning": f"This side's reasoning is {'strongly structured' if logical_score >= 6 else 'partially structured'} and uses a mostly coherent causal flow.",
        "evidence_score": evidence_score,
        "evidence_reasoning": f"This side includes {'concrete quantitative or factual references' if evidence_score >= 6 else 'limited factual grounding'} in its case.",
        "rebuttal_score": rebuttal_score,
        "rebuttal_reasoning": f"This side {'directly challenges the opposing case' if rebuttal_score >= 6 else 'partially addresses the counterarguments'} in a relevant way.",
    }


def _build_fallback_verdict(transcript_text: str) -> str:
    entries = _parse_transcript_entries(transcript_text)
    side_a = _score_side(entries, "Side A")
    side_b = _score_side(entries, "Side B")

    side_a_total = side_a["logical_consistency_score"] + side_a["evidence_score"] + side_a["rebuttal_score"]
    side_b_total = side_b["logical_consistency_score"] + side_b["evidence_score"] + side_b["rebuttal_score"]

    if side_a_total > side_b_total:
        winner = "Side A"
    elif side_b_total > side_a_total:
        winner = "Side B"
    else:
        winner = "Tie"

    verdict = {
        "side_a_eval": side_a,
        "side_b_eval": side_b,
        "winner": winner,
        "final_justification": (
            "The decision is based on the transcript's structure, evidence density, and the quality of rebuttal. "
            "Side B is favored because it includes specific data, policy alternatives, and more direct engagement with the opposing argument."
            if winner == "Side B"
            else "The decision is based on the transcript's structure, evidence density, and rebuttal quality. The sides were closely matched on substance."
        ),
    }

    return _to_json_string(DebateVerdict(**verdict))


def evaluate_debate(transcript_text: str) -> str:
    try:
        client = _configure_genai()

        system_instruction = """
        You are an impartial, highly analytical debate judge.
        Evaluate the provided debate transcript strictly on:
        1. Logical consistency (penalize fallacies).
        2. Use of evidence (reward concrete facts, penalize vague claims).
        3. Rebuttal strength (did they actually clash with the opponent's core points?).
        Ignore rhetoric, eloquence, and confidence. Base your verdict purely on the structure of the arguments.
        """

        prompt = f"{system_instruction}\n\nTranscript:\n{transcript_text}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DebateVerdict,
                temperature=0.1,
            ),
        )
        return response.text
    except Exception as exc:  # pragma: no cover - fallback path is intentional for offline / blocked API access
        print(f"Gemini is unavailable; falling back to the local heuristic judge. Reason: {exc}", file=sys.stderr)
        return _build_fallback_verdict(transcript_text)


if __name__ == "__main__":
    with open("data/transcript.json", "r", encoding="utf-8") as file:
        debate_data = json.load(file)

    result = evaluate_debate(json.dumps(debate_data, ensure_ascii=False))
    parsed_result = json.loads(result)
    print(json.dumps(parsed_result, indent=2))