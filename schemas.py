from pydantic import BaseModel, ConfigDict
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class RecommendationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reply: str
    recommendations: list[RecommendationOut]
    end_of_conversation: bool


class TurnAnalysis(BaseModel):
    intent: Literal["clarify", "recommend", "refine", "compare", "refuse", "confirm"]
    role_or_context: str | None
    seniority: str | None
    must_have_skills: list[str]
    preferred_test_types: list[str]
    max_duration_minutes: int | None
    compare_target_names: list[str]
    conversation_complete: bool
    # Names of assessments the LLM deems relevant from the provided catalog list.
    # Populated only for recommend/refine intents — used to sync the structured
    # recommendations array with what the reply text actually discusses.
    selected_assessment_names: list[str]
    reasoning: str


class ReplyOut(BaseModel):
    reply: str
