import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from models import DebateVerdict

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def evaluate_debate(transcript_text: str) -> str:
    # We use gemini-1.5-pro or gemini-1.5-flash as they support structured outputs well
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    # The strict judging rubric
    system_instruction = """
    You are an extremely strict, impartial, and highly analytical debate judge.
    Evaluate the provided debate transcript with absolute rigor based on the following three criteria:
    1. Logical Consistency: Penalize fallacies, contradictions, and unsubstantiated leaps in logic.
    2. Use of Evidence: Reward concrete facts, empirical data, and verified sources; heavily penalize vague or unsupported assertions.
    3. Rebuttal Strength: Assess how directly, specifically, and effectively the speaker countered the opponent's core arguments.

    STRICT JUDGING RULES:
    - Evaluate Logical Consistency, Use of Evidence, and Rebuttal Strength completely independently.
    - You are explicitly forbidden from giving identical scores across metrics (e.g., scoring 7 across all categories) unless completely warranted by separate, independent justification for each metric.
    - The winner MUST be determined strictly by the highest total numerical tally (the sum of Logical Consistency, Evidence, and Rebuttal scores for Side A versus Side B). In case of an exact numeric tie, declare 'Tie'.
    - Disregard rhetoric, eloquence, and confidence. Base your verdict and scores purely on the substantive structure of the arguments.
    """
    
    prompt = f"{system_instruction}\n\nTranscript:\n{transcript_text}"
    
    # Force the LLM to return data matching our DebateVerdict schema
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=DebateVerdict,
            temperature=0.1, # Low temperature for more analytical, less creative responses
        ),
        request_options={"timeout": 30},
    )
    
    return response.text

if __name__ == "__main__":
    # Load your test transcript
    with open("data/transcript.json", "r") as file:
        debate_data = json.load(file)
    
    # Run the judge
    result = evaluate_debate(str(debate_data))
    
    # Print the beautifully structured JSON result
    parsed_result = json.loads(result)
    print(json.dumps(parsed_result, indent=2))