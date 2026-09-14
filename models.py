from pydantic import BaseModel, Field

class ArgumentEvaluation(BaseModel):
    logical_consistency_score: int = Field(description="Score 0-10 for logical soundness")
    logical_consistency_reasoning: str = Field(description="1 sentence explaining the logic score")
    evidence_score: int = Field(description="Score 0-10 for use of concrete data")
    evidence_reasoning: str = Field(description="1 sentence explaining the evidence score")
    rebuttal_score: int = Field(description="Score 0-10 for addressing the opponent's claims")
    rebuttal_reasoning: str = Field(description="1 sentence explaining the rebuttal score")

class DebateVerdict(BaseModel):
    side_a_eval: ArgumentEvaluation
    side_b_eval: ArgumentEvaluation
    winner: str = Field(description="Must be exactly 'Side A', 'Side B', or 'Tie'")
    final_justification: str = Field(description="A brief summary of why the winner was chosen.")