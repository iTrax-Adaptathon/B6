import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from models import DebateVerdict

load_dotenv()


def calculate_deterministic_winner(result_data: dict) -> dict:
    """
    Calculate the final winner using only the numerical scores.

    This prevents the AI from arbitrarily choosing a winner.
    """

    side_a = result_data["side_a_eval"]
    side_b = result_data["side_b_eval"]

    side_a_total = (
        side_a["logical_consistency_score"]
        + side_a["evidence_score"]
        + side_a["rebuttal_score"]
    )

    side_b_total = (
        side_b["logical_consistency_score"]
        + side_b["evidence_score"]
        + side_b["rebuttal_score"]
    )

    if side_a_total > side_b_total:
        winner = "Side A"

    elif side_b_total > side_a_total:
        winner = "Side B"

    else:
        winner = "Tie"


    score_difference = abs(
        side_a_total - side_b_total
    )


    # Simple verdict confidence based on score gap

    if score_difference >= 6:
        confidence = "High"

    elif score_difference >= 3:
        confidence = "Medium"

    else:
        confidence = "Low"


    result_data["winner"] = winner

    result_data["side_a_total"] = side_a_total
    result_data["side_b_total"] = side_b_total

    result_data["verdict_confidence"] = confidence

    result_data["score_difference"] = score_difference


    result_data["comparative_analysis"] = (
        f"Side A scored {side_a_total}/30 while "
        f"Side B scored {side_b_total}/30. "
        f"The score difference is {score_difference} points. "
        f"Based on the deterministic scoring rule, "
        f"the final result is {winner}."
    )


    if winner == "Side A":

        result_data["final_justification"] = (
            f"Side A wins with {side_a_total}/30 "
            f"against Side B's {side_b_total}/30. "
            f"The verdict is determined directly from the "
            f"combined Logic, Evidence, and Rebuttal scores."
        )

    elif winner == "Side B":

        result_data["final_justification"] = (
            f"Side B wins with {side_b_total}/30 "
            f"against Side A's {side_a_total}/30. "
            f"The verdict is determined directly from the "
            f"combined Logic, Evidence, and Rebuttal scores."
        )

    else:

        result_data["final_justification"] = (
            f"Both sides scored {side_a_total}/30. "
            f"The deterministic scoring system therefore "
            f"declares the debate a tie."
        )


    return result_data



def evaluate_debate(transcript_text: str) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set in the .env file."
        )


    client = genai.Client(
        api_key=api_key
    )


    system_instruction = """
You are an impartial and highly analytical debate evaluator.

Your task is ONLY to evaluate the quality of the arguments.

Evaluate both sides using these three categories:

1. Logical Consistency
   - Score from 0 to 10.
   - Reward coherent reasoning.
   - Penalize contradictions, unsupported assumptions,
     and logical fallacies.

2. Use of Evidence
   - Score from 0 to 10.
   - Reward concrete facts, statistics, examples,
     data, and relevant evidence.
   - Penalize vague or unsupported claims.

3. Strength of Rebuttal
   - Score from 0 to 10.
   - Evaluate how directly each side responds
     to the opponent's core arguments.
   - Reward effective counterarguments.

Important:

Do NOT judge based on:
- confidence
- emotional language
- rhetoric
- writing style
- verbosity

The final winner will be calculated separately
by the backend from the numerical scores.

Still return all fields required by the response schema.

For the winner field, you may return "Tie".
For final_justification and comparative_analysis,
briefly explain the score differences without
overriding the mathematical score totals.
"""


    prompt = f"""
{system_instruction}

DEBATE TRANSCRIPT:

{transcript_text}

Return a structured evaluation for both sides.
"""


    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=prompt,

        config=types.GenerateContentConfig(

            temperature=0.1,

            response_mime_type="application/json",

            response_schema=DebateVerdict,

        ),

    )


    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )


    # Convert Gemini JSON response to Python dictionary

    result_data = json.loads(
        response.text
    )


    # IMPORTANT:
    # Gemini does not control the final winner anymore.

    result_data = calculate_deterministic_winner(
        result_data
    )


    # Return the updated result to FastAPI

    return json.dumps(
        result_data,
        ensure_ascii=False
    )



if __name__ == "__main__":

    with open(
        "data/transcript.json",
        "r",
        encoding="utf-8"
    ) as file:

        debate_data = json.load(file)


    result = evaluate_debate(

        json.dumps(
            debate_data,
            ensure_ascii=False
        )

    )


    parsed_result = json.loads(
        result
    )


    print(

        json.dumps(
            parsed_result,
            indent=2,
            ensure_ascii=False
        )

    )
    