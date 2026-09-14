"""
Convenience alias exposing transcript parsing functionality from transcript_parser.
"""
from transcript_parser import (
    parse_transcript,
    parse_transcript_text,
    format_transcript_for_prompt,
    SPEAKER_PATTERN
)

__all__ = [
    "parse_transcript",
    "parse_transcript_text",
    "format_transcript_for_prompt",
    "SPEAKER_PATTERN"
]

if __name__ == "__main__":
    import sys
    import subprocess
    subprocess.run([sys.executable, "transcript_parser.py"] + sys.argv[1:])
