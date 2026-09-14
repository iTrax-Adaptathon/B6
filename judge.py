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
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # The strict judging rubric
    system_instruction = """
    You are an impartial, highly analytical debate judge. 
    Evaluate the provided debate transcript strictly on:
    1. Logical consistency (penalize fallacies).
    2. Use of evidence (reward concrete facts, penalize vague claims).
    3. Rebuttal strength (did they actually clash with the opponent's core points?).
    Ignore rhetoric, eloquence, and confidence. Base your verdict purely on the structure of the arguments.
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