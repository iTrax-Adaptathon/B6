from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import json

from judge import evaluate_debate


app = FastAPI()


# ============================================================
# DATA MODELS
# ============================================================

class DebateStatement(BaseModel):
    speaker: str
    text: str


class DebateInput(BaseModel):
    topic: str
    statements: List[DebateStatement]


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():

    with open(
        "index.html",
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# DEBATE JUDGE API
# ============================================================

@app.post("/api/judge")
async def judge_api(debate: DebateInput):

    try:

        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if not debate.topic.strip():

            raise ValueError(
                "A debate topic is required."
            )


        if len(debate.statements) < 2:

            raise ValueError(
                "At least two statements are required."
            )


        side_a_count = 0
        side_b_count = 0


        transcript_statements = []


        # ----------------------------------------------------
        # BUILD CHRONOLOGICAL TRANSCRIPT
        # ----------------------------------------------------

        for number, statement in enumerate(
            debate.statements,
            start=1
        ):

            speaker = statement.speaker.strip()
            text = statement.text.strip()


            if speaker not in ["Side A", "Side B"]:

                raise ValueError(
                    f"Invalid speaker in statement {number}."
                )


            if not text:

                raise ValueError(
                    f"Statement {number} is empty."
                )


            if speaker == "Side A":

                side_a_count += 1

            else:

                side_b_count += 1


            transcript_statements.append({

                "statement_number": number,

                "speaker": speaker,

                "text": text

            })


        # ----------------------------------------------------
        # MAKE SURE BOTH SIDES PARTICIPATED
        # ----------------------------------------------------

        if side_a_count == 0 or side_b_count == 0:

            raise ValueError(
                "Both Side A and Side B must participate "
                "before the debate can be judged."
            )


        # ----------------------------------------------------
        # FINAL TRANSCRIPT
        # ----------------------------------------------------

        transcript = {

            "topic": debate.topic.strip(),

            "format": "dynamic_chronological_debate",

            "statement_count": len(
                transcript_statements
            ),

            "side_a_statement_count":
                side_a_count,

            "side_b_statement_count":
                side_b_count,

            "statements":
                transcript_statements

        }


        # ----------------------------------------------------
        # SEND COMPLETE DEBATE TO JUDGE
        # ----------------------------------------------------

        result_json_string = evaluate_debate(

            json.dumps(
                transcript,
                ensure_ascii=False,
                indent=2
            )

        )


        return json.loads(
            result_json_string
        )


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# RUN SERVER DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )