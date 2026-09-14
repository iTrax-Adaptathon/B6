import os
import re
from pathlib import Path
from typing import List, Dict, Union, Optional

# Non-speaker words preceding colons to ignore as speaker identifiers
EXCLUDED_LABELS = {
    "note", "notice", "warning", "caution", "tip", 
    "topic", "motion", "resolution", "transcript", 
    "context", "summary", "time", "date"
}

# Regex pattern matching speaker tags like:
# "Side A:", "Speaker 1:", "[Side B]:", "**Speaker 2**:", "Debater 1:", "Side A (Opening):"
SPEAKER_PATTERN = re.compile(
    r'^\s*(?:\[|\*{1,2})?\s*'
    r'('
    r'Side\s+[A-Za-z0-9]+'
    r'|Speaker\s+[A-Za-z0-9]+'
    r'|Debater\s+[A-Za-z0-9]+'
    r'|Affirmative|Negative|Proponent|Opponent|Proposition|Opposition|Government'
    r'|[A-Za-z][A-Za-z0-9\s_\.\'-]{1,30}?'
    r')'
    r'(?:\s*\([^)]*\))?'
    r'\s*(?:\]|\*{1,2})?\s*[:\-]\s*(.*)$',
    re.IGNORECASE
)


def format_transcript_for_prompt(turns: List[Dict[str, str]]) -> str:
    """
    Formats a list of speaker dialogue turns into a clean, structured string
    that can be passed directly into Gemini's debate judge prompt.
    """
    blocks = []
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"].strip()
        blocks.append(f"[{speaker}]:\n{text}")
    return "\n\n".join(blocks)


def parse_transcript_text(
    raw_text: str,
    as_string: bool = False,
    strict: bool = True
) -> Union[List[Dict[str, str]], str]:
    """
    Parses raw unstructured text containing debate dialogue.

    Args:
        raw_text: Unstructured debate text.
        as_string: If True, returns a clean formatted string for LLM prompts.
                   If False, returns a list of dictionaries [{'speaker': ..., 'text': ...}].
        strict: If True, raises ValueError if dialogue occurs before any speaker label.
                If False, labels unassigned initial text as 'Context'.

    Returns:
        List of dictionaries with 'speaker' and 'text' keys, or a formatted string.

    Raises:
        ValueError: If input is empty, has missing speaker labels, or invalid content.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Transcript content is empty.")

    lines = raw_text.splitlines()
    turns: List[Dict[str, str]] = []
    current_speaker: Optional[str] = None
    current_dialogue_parts: List[str] = []

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            # Skip empty lines cleanly
            continue

        match = SPEAKER_PATTERN.match(line)
        if match:
            candidate_speaker = match.group(1).strip()
            # Verify candidate speaker is not an excluded metadata keyword
            if candidate_speaker.lower() not in EXCLUDED_LABELS:
                # Save previous speaker turn if present
                if current_speaker:
                    dialogue = " ".join(current_dialogue_parts).strip()
                    if dialogue:
                        turns.append({"speaker": current_speaker, "text": dialogue})
                    current_dialogue_parts = []

                current_speaker = candidate_speaker
                initial_text = match.group(2).strip()
                if initial_text:
                    current_dialogue_parts.append(initial_text)
                continue

        # Line does not define a new speaker
        if current_speaker is not None:
            current_dialogue_parts.append(line)
        else:
            # Content before any speaker label is identified
            if strict:
                raise ValueError(
                    f"Missing speaker label on line {line_num}: '{line}'. "
                    "Each section of dialogue must begin with a speaker identifier (e.g., 'Side A:', 'Speaker 1:')."
                )
            else:
                current_speaker = "Context"
                current_dialogue_parts.append(line)

    # Flush the final speaker turn
    if current_speaker:
        dialogue = " ".join(current_dialogue_parts).strip()
        if dialogue:
            turns.append({"speaker": current_speaker, "text": dialogue})

    if not turns:
        raise ValueError("No speaker labels or dialogue found in transcript.")

    if as_string:
        return format_transcript_for_prompt(turns)

    return turns


def parse_transcript(
    file_path: Union[str, Path],
    as_string: bool = False,
    strict: bool = True
) -> Union[List[Dict[str, str]], str]:
    """
    Reads an unstructured debate transcript .txt file, identifies speaker names,
    and extracts dialogue.

    Args:
        file_path: Path to the .txt debate transcript file.
        as_string: If True, returns a clean formatted string for the judge prompt.
                   If False, returns a list of dictionaries.
        strict: If True, enforces speaker labels for all text sections.

    Returns:
        Structured dialogue turns as list of dicts or formatted prompt string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty, missing speaker labels, or unparseable.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    return parse_transcript_text(content, as_string=as_string, strict=strict)


if __name__ == "__main__":
    import sys
    import json

    # Demonstration / CLI usage
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default sample path if available
        target_path = os.path.join(os.path.dirname(__file__), "data", "sample_debate.txt")

    if os.path.exists(target_path):
        print(f"--- Parsing: {target_path} ---")
        structured_list = parse_transcript(target_path, as_string=False)
        print("\n[Structured List of Dictionaries]:")
        print(json.dumps(structured_list, indent=2))

        print("\n[Formatted Prompt String for Gemini]:")
        prompt_string = parse_transcript(target_path, as_string=True)
        print(prompt_string)
    else:
        print(f"Usage: python transcript_parser.py <path_to_transcript.txt>")
