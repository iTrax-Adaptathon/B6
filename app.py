from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os

# Import the evaluate_debate function you already built!
from judge import evaluate_debate 

app = FastAPI()

# Data model for what the frontend will send us
class DebateInput(BaseModel):
    side_a: str
    side_b: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Serve the index.html file you just created
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/judge")
async def judge_api(debate: DebateInput):
    try:
        # Format the frontend input to match the transcript style your AI expects
        transcript = [
            {"speaker": "Side A", "text": debate.side_a},
            {"speaker": "Side B", "text": debate.side_b}
        ]
        
        # Run your existing Gemini logic!
        result_json_string = evaluate_debate(str(transcript))
        
        # Return the parsed JSON back to the frontend
        return json.loads(result_json_string)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Runs the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)