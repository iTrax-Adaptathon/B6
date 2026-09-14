from pydantic import BaseModel, Field

class ArgumentEvaluation(BaseModel):
    logical_consistency_reasoning: str = Field(description="Step-by-step reasoning evaluating the logical soundness and consistency of the argument")
    logical_consistency_score: int = Field(description="Score 0-10 for logical soundness")
    evidence_reasoning: str = Field(description="Step-by-step reasoning evaluating the use of concrete facts and data versus unsupported claims")
    evidence_score: int = Field(description="Score 0-10 for use of concrete data")
    rebuttal_reasoning: str = Field(description="Step-by-step reasoning evaluating how directly and effectively the opponent's points were addressed")
    rebuttal_score: int = Field(description="Score 0-10 for addressing the opponent's claims")

class DebateVerdict(BaseModel):
    side_a_eval: ArgumentEvaluation
    side_b_eval: ArgumentEvaluation
    comparative_analysis: str = Field(
        description="Step-by-step comparative analysis mathematically comparing Side A and Side B's total scores and metric differences before declaring a winner"
    )
    winner: str = Field(description="Must be exactly 'Side A', 'Side B', or 'Tie'")
    final_justification: str = Field(description="A brief summary of why the winner was chosen.")