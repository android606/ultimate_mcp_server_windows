"""Demonstration script for the advanced audio transcription tools."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console, Rule
from rich.markup import escape
from rich.panel import Panel

# Add the project root to the Python path
# This allows finding the llm_gateway package when running the script directly
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

EXAMPLE_DIR = Path(__file__).parent
DATA_DIR = EXAMPLE_DIR / "data"
# IMPORTANT: Place a sample audio file (e.g., sample_audio.mp3, sample_audio.wav) in the examples/data/ directory for this demo to run.
SAMPLE_AUDIO_PATH = str(DATA_DIR / "Steve_Jobs_Introducing_The_iPhone_compressed.mp3") # Replace with your actual file name

from mcp.types import AudioContent, TextContent  # noqa: E402

from llm_gateway.core import Gateway  # noqa: E402
from llm_gateway.utils import configure_logging, get_logger  # noqa: E402
from llm_gateway.utils.display import (  # noqa: E402
    CostTracker,
    display_text_content_result,
)

# --- Configuration ---
configure_logging()
logger = get_logger("audio_demo")

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"

# Define allowed audio extensions
ALLOWED_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]

# --- Helper Functions ---
def find_audio_files(directory: Path) -> List[Path]:
    """Finds audio files with allowed extensions in the given directory."""
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]

def get_mime_type(file_path: Path) -> Optional[str]:
    """Attempts to determine the MIME type based on file extension."""
    extension_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4", # Common mapping for m4a
    }
    return extension_map.get(file_path.suffix.lower())

async def main():
    """Runs the audio transcription demonstrations."""

    logger.info("Starting Audio Transcription Demo", emoji_key="audio")

    console = Console()
    console.print(Rule("[bold green]Audio Transcription Demo[/bold green]"))

    # --- Initialize Gateway ---
    # Uses default environment variables for API keys (e.g., OPENAI_API_KEY)
    llm = Gateway()
    tracker = CostTracker() # Instantiate the tracker

    # --- Find Audio Files ---
    audio_files = find_audio_files(DATA_DIR)
    if not audio_files:
        console.print(f"[bold red]Error:[/bold red] No audio files found in {DATA_DIR}. Please place audio files (e.g., .mp3, .wav) there.")
        return

    console.print(f"Found {len(audio_files)} audio file(s) in {DATA_DIR}:")
    for f in audio_files:
        console.print(f"- [cyan]{f.name}[/cyan]")
    console.print()

    # --- Process Each File ---
    for file_path in audio_files:
        try:
            console.print(Panel(
                f"Processing file: [cyan]{escape(str(file_path))}[/cyan]",
                title="Audio Transcription",
                border_style="blue"
            ))

            # --- Read Audio File ---
            try:
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
                console.print(f"[green]Successfully read audio file ({len(audio_bytes)} bytes).[/green]")
            except Exception as e:
                console.print(f"[bold red]Error reading audio file {escape(str(file_path))}:[/bold red] {escape(str(e))}")
                continue # Skip to next file

            # --- Prepare Audio Content ---
            # Determine content type based on file extension
            content_type = get_mime_type(file_path)
            if not content_type:
                console.print(f"[yellow]Warning:[/yellow] Could not determine MIME type for {escape(str(file_path))}. Assuming 'audio/mpeg'.")
                content_type = "audio/mpeg" # Default or make more robust

            audio_content = AudioContent(audio=audio_bytes, content_type=content_type)

            # --- Perform Transcription ---
            try:
                # Use the Gateway's transcribe method
                # Specify a model that supports transcription (e.g., openai/whisper-1)
                result = await llm.transcribe(
                    messages=[audio_content], # Pass AudioContent directly
                    model="openai/whisper-1"
                )
                console.print(f"[green]Transcription successful for {escape(str(file_path))}.[/green]")
                
                # --- Track Cost --- 
                tracker.add_call(result) # Add the result to the tracker

                # --- Display Result ---
                # Extract and display the text content using the utility function
                # Make sure the result has the text content expected by the display function
                # (The transcribe method should return a result object compatible with TextContent or similar)
                if isinstance(result, list):
                     # Handle potential list results (take the first TextContent)
                     text_result = next((item for item in result if isinstance(item, TextContent)), None)
                elif isinstance(result, TextContent):
                    text_result = result
                else:
                    # Attempt to adapt if result has a 'text' attribute (like CompletionResult)
                    if hasattr(result, 'text'):
                         text_result = result # display_text_content_result can handle this
                    else:
                         text_result = None # Cannot display if no text found
                         console.print("[yellow]Warning:[/yellow] Could not extract text from transcription result for display.")

                if text_result:                    
                    display_text_content_result(
                        f"Transcription Result for {escape(file_path.name)}",
                        text_result,
                        console_instance=console
                    )
                
            except Exception as e:
                console.print(f"[bold red]Transcription failed for {escape(str(file_path))}:[/bold red] {escape(str(e))}")
                # Optionally, add error details to tracker if needed, though cost might be 0
                # tracker.add_call({}, provider="Unknown", model="Unknown") # Example if you want to track failures

            console.print() # Add a blank line between files
        except Exception as outer_e:
             # Catch errors in the outer loop for a specific file
             console.print(f"[bold red]Unexpected error processing file {escape(str(file_path))}:[/bold red] {escape(str(outer_e))}")
             continue # Move to the next file

    # --- Display Final Cost Summary ---
    tracker.display_summary(console)

    logger.info("Audio Transcription Demo Finished", emoji_key="audio")

if __name__ == "__main__":
    # Ensure Whisper models are downloaded and dependencies (ffmpeg) are installed
    # Example: ~/whisper.cpp/models/download-ggml-model.sh large-v3-turbo
    # Example: sudo apt update && sudo apt install ffmpeg
    logger.info("Ensure prerequisites are met: ffmpeg installed, whisper.cpp built, models downloaded.", emoji_key="info")
    # Basic error handling for the async execution itself
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred running the demo: {e}")
        # Optionally, add more robust logging here if needed
        # import traceback
        # traceback.print_exc() 