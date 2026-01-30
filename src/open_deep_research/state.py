"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from typing import List, Optional, Dict, Any, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from enum import Enum

MessageLikeRepresentation = Union[BaseMessage, Dict[str, Any], str, List[Union[str, Dict[str, Any]]]]

class AttackType(str, Enum):
    """Типы обнаруженных атак."""
    FALSE_INFORMATION = "false_information"
    BIASED_DATA = "biased_data"
    CONFIRMATION_BIAS = "confirmation_bias"
    COMMAND_INJECTION = "command_injection"
    LINK_INJECTION = "link_injection"
    EMAIL_INJECTION = "email_injection"
    UNBALANCED_VIEW = "unbalanced_view"


class DetectedAttack(BaseModel):
    """Обнаруженная атака."""
    type: AttackType
    description: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    mitigation: str


class SecurityState(BaseModel):
    """Состояние безопасности."""
    detected_attacks: List[DetectedAttack] = []
    sanitized_content: Optional[str] = None
    needs_human_review: bool = False
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0)

###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(TypedDict, total=False):
    messages: List[MessageLikeRepresentation]  # <-- ВСЕГДА одинаковый тип!
    research_brief: Optional[str]
    supervisor_messages: List[MessageLikeRepresentation]  # <-- Тот же тип
    researcher_messages: List[MessageLikeRepresentation]  # <-- Тот же тип
    notes: List[str]
    raw_notes: List[str]
    research_iterations: int
    tool_call_iterations: int
    security_state: Optional[Any]  # Пока Any, потом заменим
    original_research_content: Optional[str]

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []