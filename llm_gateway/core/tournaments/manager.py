"""
Tournament manager for tournament lifecycle operations.
"""
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from llm_gateway.core.models.tournament import (
    CreateTournamentInput,
    ModelConfig,
    TournamentConfig,
    TournamentData,
    TournamentStatus,
)
from llm_gateway.utils import get_logger

logger = get_logger("llm_gateway.tournaments.manager")

# Define storage path relative to project structure or use config
STORAGE_DIR = Path(__file__).parent.parent.parent.parent / "storage" 
TOURNAMENT_STORAGE_BASE = STORAGE_DIR / "tournaments"

class TournamentManager:
    def __init__(self):
        # In-memory cache; real app might use DB + cache
        self.tournaments: Dict[str, TournamentData] = {}
        TOURNAMENT_STORAGE_BASE.mkdir(parents=True, exist_ok=True)
        logger.info(f"Tournament storage initialized at: {TOURNAMENT_STORAGE_BASE}")
        # Load existing tournaments on startup
        self._load_all_tournaments()

    def create_tournament(self, input_data: CreateTournamentInput) -> Optional[TournamentData]:
        """Create a new tournament with validation."""
        try:
            logger.debug(f"Creating tournament with name: {input_data.name}, {len(input_data.model_ids)} models")
            # Create config first
            tournament_config = TournamentConfig(
                name=input_data.name,  # Pass name to config
                prompt=input_data.prompt,
                models=[ModelConfig(model_id=mid) for mid in input_data.model_ids],
                rounds=input_data.rounds,
                tournament_type=input_data.tournament_type,
                extraction_model_id=input_data.extraction_model_id
            )
            
            # Create full TournamentData with required fields
            tournament = TournamentData(
                name=input_data.name,  # Required field from input
                config=tournament_config,
                # Don't set storage_path here, will be set after ID is generated
                start_time=datetime.now(timezone.utc),  # Add timezone-aware datetime
                end_time=None  # Explicitly set as None initially
            )
            
            # Now that tournament ID is generated, set the storage path using the ID
            tournament.storage_path = str(self._get_storage_path(tournament.tournament_id))
            
            logger.debug(f"Generated tournament ID: {tournament.tournament_id}")
            self.tournaments[tournament.tournament_id] = tournament
            logger.debug("Added tournament to in-memory cache, now saving state")
            self._save_tournament_state(tournament)
            logger.debug("Tournament state saved to disk")
            return tournament
        except ValidationError as ve:
            logger.error(f"Tournament validation failed: {ve}")
            return None

    def get_tournament(self, tournament_id: str, force_reload: bool = False) -> Optional[TournamentData]:
        """Retrieves tournament data, checking cache first unless forced."""
        logger.debug(f"Getting tournament {tournament_id} (force_reload={force_reload})")
        if not force_reload and tournament_id in self.tournaments:
            logger.debug(f"Tournament {tournament_id} found in cache")
            return self.tournaments[tournament_id]
        # Attempt to load from storage if not in cache or reload forced
        logger.debug(f"Tournament {tournament_id} not in cache or force reload requested, loading from disk")
        tournament = self._load_tournament_state(tournament_id)
        if tournament:
            logger.debug(f"Tournament {tournament_id} loaded from disk successfully")
        else:
            logger.debug(f"Tournament {tournament_id} not found on disk")
        return tournament

    def update_tournament_status(self, tournament_id: str, status: TournamentStatus, message: str = None):
        tournament = self.get_tournament(tournament_id, force_reload=True) # Ensure we have latest
        if tournament:
            tournament.status = status
            tournament.error_message = message
            tournament.updated_at = datetime.now()
            self._save_tournament_state(tournament)
            logger.info(f"Tournament {tournament_id} status updated to {status}.")
        else:
             logger.warning(f"Attempted to update status for non-existent tournament {tournament_id}")

    def _save_tournament_state(self, tournament: TournamentData):
        """Saves the tournament state to a JSON file in its storage directory."""
        if not tournament.storage_path:
            logger.error(f"Cannot save state for tournament {tournament.tournament_id}: storage_path not set.")
            return
            
        state_file = Path(tournament.storage_path) / "tournament_state.json"
        try:
            # Ensure parent directory exists
            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, 'w', encoding='utf-8') as f:
                json_data = tournament.model_dump_json(indent=2)
                f.write(json_data)
            logger.debug(f"Saved state for tournament {tournament.tournament_id} to {state_file}")
        except IOError as e:
            logger.error(f"Failed to save state for tournament {tournament.tournament_id}: {e}")

    def _load_tournament_state(self, tournament_id: str) -> Optional[TournamentData]:
        """Loads tournament state from its JSON file."""
        # First try the direct path by tournament ID (legacy/expected path)
        state_file = TOURNAMENT_STORAGE_BASE / tournament_id / "tournament_state.json"
        logger.debug(f"Attempting to load tournament state from {state_file}")
        
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    tournament = TournamentData(**data)
                    self.tournaments[tournament_id] = tournament # Update cache
                    logger.debug(f"Loaded state for tournament {tournament_id} from {state_file}")
                    return tournament
            except (IOError, json.JSONDecodeError, ValidationError) as e:
                 logger.error(f"Failed to load state for tournament {tournament_id} from {state_file}: {e}")
                 # Continue to the fallback below
        else:
            logger.debug(f"State file not found at direct path: {state_file}")
            
        # Fallback: Scan all subdirectories for matching tournament ID
        logger.debug(f"Searching subdirectories for tournament {tournament_id}...")
        if TOURNAMENT_STORAGE_BASE.exists():
            for subdir in TOURNAMENT_STORAGE_BASE.iterdir():
                if subdir.is_dir():
                    alt_state_file = subdir / "tournament_state.json"
                    if alt_state_file.exists():
                        try:
                            with open(alt_state_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Check if this file contains our tournament
                                if data.get("tournament_id") == tournament_id:
                                    logger.debug(f"Found tournament {tournament_id} in directory: {subdir}")
                                    tournament = TournamentData(**data)
                                    self.tournaments[tournament_id] = tournament # Update cache
                                    return tournament
                        except (IOError, json.JSONDecodeError) as e:
                            logger.debug(f"Error reading {alt_state_file}: {e}")
                            continue  # Try next directory
        
        logger.debug(f"Tournament {tournament_id} not found in any storage location")
        return None

    def _load_all_tournaments(self):
        """Scans storage and loads all existing tournament states into memory."""
        logger.info(f"Scanning {TOURNAMENT_STORAGE_BASE} for existing tournaments...")
        count = 0
        if not TOURNAMENT_STORAGE_BASE.exists():
            logger.warning("Tournament storage directory does not exist. No tournaments loaded.")
            return
            
        for item in TOURNAMENT_STORAGE_BASE.iterdir():
            if item.is_dir():
                tournament_id = item.name
                if tournament_id not in self.tournaments: # Avoid reloading if already cached
                    logger.debug(f"Found potential tournament directory: {tournament_id}")
                    loaded = self._load_tournament_state(tournament_id)
                    if loaded:
                        count += 1
        logger.info(f"Finished scan. Loaded {count} existing tournaments into cache.")

    def start_tournament_execution(self, tournament_id: str):
        """
        Initiates the asynchronous execution of the tournament using asyncio.
        
        Args:
            tournament_id: The ID of the tournament to run.
        """
        logger.debug(f"Attempting to start tournament execution for {tournament_id}")
        tournament = self.get_tournament(tournament_id)
        if not tournament:
            logger.error(f"Cannot start execution: Tournament {tournament_id} not found.")
            return False
        
        if tournament.status not in [TournamentStatus.PENDING, TournamentStatus.CREATED]:
            logger.warning(f"Tournament {tournament_id} is not in a runnable state ({tournament.status}). Cannot start.")
            return False

        # Update status to RUNNING before starting the task
        logger.debug(f"Setting tournament {tournament_id} status to RUNNING")
        tournament.status = TournamentStatus.RUNNING
        tournament.start_time = datetime.now(timezone.utc)
        self._save_tournament_state(tournament)
        logger.debug("Tournament state saved with RUNNING status")

        # --- Use asyncio.create_task for background execution --- 
        logger.info(f"Scheduling tournament {tournament_id} using asyncio.create_task.")
        try:
            from llm_gateway.core.tournaments.tasks import run_tournament_async
            # Use asyncio.create_task to run in background within the existing loop
            logger.debug(f"Creating asyncio task for tournament {tournament_id}")
            asyncio.create_task(run_tournament_async(tournament_id)) 
            logger.info(f"Asyncio task created for tournament {tournament_id}.")
            return True # Indicate it was triggered asynchronously
        except Exception as e:
             logger.error(f"Error creating asyncio task for tournament {tournament_id}: {e}", exc_info=True)
             # Attempt to mark as failed if task creation crashes
             tournament.status = TournamentStatus.FAILED
             tournament.error_message = f"Failed during asyncio task creation: {e}" # Updated field name
             self._save_tournament_state(tournament)
             return False

    def cancel_tournament(self, tournament_id: str) -> Tuple[bool, str]:
        """Attempts to cancel a running tournament by updating its status."""
        tournament = self.get_tournament(tournament_id, force_reload=True)
        if not tournament:
            logger.warning(f"Cannot cancel non-existent tournament {tournament_id}")
            return False, "Tournament not found."
        
        if tournament.status == TournamentStatus.RUNNING:
            logger.info(f"Attempting to cancel tournament {tournament_id}...")
            self.update_tournament_status(tournament_id, TournamentStatus.CANCELLED, "Tournament cancelled by user request.")
            # Add end time when cancelling
            tournament.end_time = datetime.now(timezone.utc)
            self._save_tournament_state(tournament)
            logger.info(f"Tournament {tournament_id} status set to CANCELLED.")
            # Note: The background task needs to check this status to actually stop.
            return True, "Cancellation requested. Tournament status set to CANCELLED."
        elif tournament.status in [TournamentStatus.COMPLETED, TournamentStatus.FAILED, TournamentStatus.CANCELLED]:
             msg = f"Tournament {tournament_id} is already finished or cancelled (Status: {tournament.status})."
             logger.warning(msg)
             return False, msg
        elif tournament.status == TournamentStatus.PENDING:
            # If pending, we can just cancel it directly
            self.update_tournament_status(tournament_id, TournamentStatus.CANCELLED, "Tournament cancelled before starting.")
            tournament.end_time = datetime.now(timezone.utc)
            self._save_tournament_state(tournament)
            logger.info(f"Pending tournament {tournament_id} cancelled.")
            return True, "Pending tournament cancelled successfully."
        else:
            # Should not happen, but handle unknown state
            msg = f"Tournament {tournament_id} is in an unexpected state ({tournament.status}). Cannot cancel."
            logger.error(msg)
            return False, msg

    def list_tournaments(self) -> List[Dict[str, Any]]:
        """Lists basic info for all tournaments currently loaded in the manager."""
        # Now relies on the cache being populated by __init__ or subsequent loads
        logger.debug(f"Listing {len(self.tournaments)} tournaments from cache.")
        
        # Now, compile the list from the potentially updated cache
        basic_list = []
        for _, t_data in self.tournaments.items(): # Use _ for unused t_id
            basic_list.append({
                "tournament_id": t_data.tournament_id,
                "name": t_data.name, # Add name
                "status": t_data.status,
                "current_round": t_data.current_round,
                "total_rounds": t_data.config.rounds,
                "created_at": t_data.created_at,
                "start_time": t_data.start_time, # Add start time
                "end_time": t_data.end_time,   # Add end time
            })
            
        # Sort by creation time, newest first (optional)
        basic_list.sort(key=lambda x: x['created_at'], reverse=True)
            
        logger.info(f"Found {len(basic_list)} tournaments to list.")
        return basic_list
        
    def _get_storage_path(self, tournament_id: str) -> Path:
        """Generate storage path for a tournament using ISO datetime and meaningful name
        
        Format: YYYY-MM-DD_HH-MM-SS_tournament-name_UUID-suffix
        Creates semantically meaningful directory names for easier browsing.
        """
        # Get the tournament from cache if available
        tournament = self.tournaments.get(tournament_id)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create a meaningful name from the tournament name
        if tournament and tournament.name:
            # Clean the tournament name - replace spaces with underscores, remove special chars
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', tournament.name.lower().replace(' ', '_'))
            
            # Limit length to avoid overly long paths
            safe_name = safe_name[:30]
            
            # Get just the first 8 chars of UUID for uniqueness
            uuid_short = tournament_id.split('-')[0] if '-' in tournament_id else tournament_id[:8]
            
            # Format: date_time_name_uuid-short
            folder_name = f"{timestamp}_{safe_name}_{uuid_short}"
        else:
            # Fallback if no name available
            folder_name = f"{timestamp}_tournament_{tournament_id[:8]}"
        
        # Create path
        return TOURNAMENT_STORAGE_BASE / folder_name

# Instantiate the manager as a singleton (or use dependency injection)
tournament_manager = TournamentManager() 