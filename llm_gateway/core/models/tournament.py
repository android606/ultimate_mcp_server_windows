import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class TournamentBase(BaseModel):
    """Base class for tournament-related models."""
    tournament_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: str = "PENDING"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TournamentPlayerBase(BaseModel):
    """Base class for tournament players/competitors."""
    player_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    model_id: str


class TournamentMatch(BaseModel):
    """Represents a match between players in a tournament."""
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tournament_id: str
    players: List[str]  # List of player_ids
    winner: Optional[str] = None  # player_id of winner
    match_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class TournamentStatus(str, Enum):
    PENDING = "PENDING"
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class ModelConfig(BaseModel):
    model_id: str
    # Add any specific parameters per model for the tournament if needed
    # e.g., temperature_override: Optional[float] = None

class TournamentConfig(BaseModel):
    name: str
    prompt: str
    models: List[ModelConfig]
    rounds: int = 5
    tournament_type: Literal["code", "text"] = "code"
    # Add other config like temperature defaults, code extraction strategy etc.
    extraction_model_id: Optional[str] = None

class ModelResponseData(BaseModel):
    model_id: str
    round_num: int
    response_text: Optional[str] = None
    thinking_process: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    response_file_path: Optional[str] = None
    
    extracted_code: Optional[str] = None
    extracted_code_path: Optional[str] = None

class TournamentRoundResult(BaseModel):
    round_num: int
    status: TournamentStatus = TournamentStatus.PENDING
    responses: Dict[str, ModelResponseData] = Field(default_factory=dict)
    comparison_file_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    storage_path: Optional[str] = None

class TournamentData(BaseModel):
    tournament_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: TournamentConfig
    status: TournamentStatus = TournamentStatus.PENDING
    current_round: int = 0
    rounds_results: List[TournamentRoundResult] = Field(default_factory=list)
    storage_path: Optional[str] = None
    name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# --- Tool Input/Output Models (used in server.py) ---

class CreateTournamentInput(BaseModel):
    name: str = Field(..., description="A descriptive name for the tournament.")
    prompt: str = Field(..., description="The initial challenge prompt or question.")
    model_ids: List[str] = Field(..., description="List of provider:model_id strings to include.")
    rounds: int = Field(default=3, ge=1, le=10, description="Number of refinement rounds (e.g., 3 rounds means initial + 3 refinements).")
    tournament_type: Literal["code", "text"] = Field(default="code", description="Type of tournament ('code' or 'text'). Controls processing.")
    extraction_model_id: Optional[str] = Field(default=None, description="Optional model ID to use for code extraction (defaults to Claude Haiku).")
    
    @validator('model_ids')
    def check_model_ids_non_empty(cls, v):
        if not v:
            raise ValueError('model_ids list cannot be empty.')
        # Optional: Add validation for format 'provider:model_id' here?
        # for model_id in v:
        #     if ':' not in model_id:
        #         raise ValueError(f'Invalid model_id format: {model_id}. Expected format: provider:model_id')
        return v
    
    @validator('name')
    def check_name_non_empty(cls, v):
        if not v or not v.strip():
             raise ValueError('Tournament name cannot be empty.')
        return v.strip() # Trim whitespace

class CreateTournamentOutput(BaseModel):
    tournament_id: str = Field(..., description="The unique ID for the initiated tournament.")
    status: TournamentStatus

class GetTournamentStatusInput(BaseModel):
    tournament_id: str = Field(..., description="The unique ID of the tournament to check.")

class GetTournamentStatusOutput(BaseModel):
    tournament_id: str
    name: str
    tournament_type: Literal["code", "text"]
    status: TournamentStatus
    current_round: int
    total_rounds: int
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None

# Consider adding input/output models for getting full details, listing tournaments etc. 

# --- Added Models for Tools --- 
class TournamentBasicInfo(BaseModel):
    """Basic information about a tournament for list views."""
    tournament_id: str
    name: str
    tournament_type: Literal["code", "text"]
    status: TournamentStatus
    current_round: int
    total_rounds: int
    created_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

# We might need a model to wrap the list for the tool output if FastMCP requires it,
# but List[TournamentBasicInfo] should work based on the decorator usage.
# class ListTournamentsOutput(BaseModel):
#     tournaments: List[TournamentBasicInfo]

class GetTournamentResultsInput(BaseModel):
    """Input model for retrieving full tournament results."""
    tournament_id: str = Field(..., description="The unique ID of the tournament to retrieve.")

# The output is simply the full TournamentData model itself.
# No separate output model needed unless we want to selectively filter fields. 

class CancelTournamentInput(BaseModel):
    """Input model for cancelling a tournament."""
    tournament_id: str = Field(..., description="The unique ID of the tournament to cancel.")

class CancelTournamentOutput(BaseModel):
    """Output model after attempting cancellation."""
    tournament_id: str
    status: TournamentStatus # Return the status after attempting cancellation
    message: str 