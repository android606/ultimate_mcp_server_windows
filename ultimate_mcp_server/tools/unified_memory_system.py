"""Unified Memory System

This module provides a comprehensive memory, reasoning, and workflow tracking system
designed for LLM agents, merging sophisticated cognitive modeling with structured
process tracking.

Key Features:
- Multi-level memory hierarchy (working, episodic, semantic, procedural) with rich metadata.
- Structured workflow, action, artifact, and thought chain tracking.
- Associative memory graph with automatic linking capabilities.
- Vector embeddings for semantic similarity and clustering.
- Foundational tools for recording agent activity and knowledge.
- Integrated episodic memory creation linked to actions and artifacts.
- Basic cognitive state saving (structure defined, loading/saving tools ported).
- SQLite backend using aiosqlite with performance optimizations.
"""

import asyncio
import contextlib
import json
import os
import re
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiosqlite
import markdown
import numpy as np
from pygments.formatters import HtmlFormatter
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from ultimate_mcp_server.config import get_config

from ultimate_mcp_server.constants import (
    Provider as LLMGatewayProvider,  # To use provider constants
)
from ultimate_mcp_server.core.providers.base import (
    get_provider,  # For consolidation/reflection LLM calls
)

# Import error handling and decorators from agent_memory concepts
from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.services.vector.embeddings import get_embedding_service
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.unified_memory")

# ======================================================
# Configuration Settings
# ======================================================

# Load config once at module level for efficiency
try:
    config = get_config()
    # Extract agent memory config for easier access
    agent_memory_config = config.agent_memory
except Exception as config_e:
    logger.critical(f"CRITICAL: Failed to load configuration for unified_memory_system: {config_e}", exc_info=True)
    # Provide fallback defaults if config fails, allowing *some* functionality maybe?
    # Or raise the error immediately. Raising is probably safer.
    raise RuntimeError(f"Failed to initialize configuration for unified_memory_system: {config_e}") from config_e

# ======================================================
# Enums (Combined & Standardized)
# ======================================================

# --- Workflow & Action Status ---
class WorkflowStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ActionStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# --- Content Types ---
class ActionType(str, Enum):
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    PLANNING = "planning"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DECISION = "decision"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SUMMARY = "summary"
    CONSOLIDATION = "consolidation"
    MEMORY_OPERATION = "memory_operation"


class ArtifactType(str, Enum):
    FILE = "file"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    CODE = "code"
    DATA = "data"
    JSON = "json"
    URL = "url"


class ThoughtType(str, Enum):
    GOAL = "goal"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    EVIDENCE = "evidence"
    CONSTRAINT = "constraint"
    PLAN = "plan"
    DECISION = "decision"
    REFLECTION = "reflection"
    CRITIQUE = "critique"
    SUMMARY = "summary"
    INSIGHT = "insight"


# --- Memory System Types ---
class MemoryLevel(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryType(str, Enum):
    """Content type classifications for memories. Combines concepts."""

    OBSERVATION = "observation"  # Raw data or sensory input (like text)
    ACTION_LOG = "action_log"  # Record of an agent action
    TOOL_OUTPUT = "tool_output"  # Result from a tool
    ARTIFACT_CREATION = "artifact_creation"  # Record of artifact generation
    REASONING_STEP = "reasoning_step"  # Corresponds to a thought
    FACT = "fact"  # Verifiable piece of information
    INSIGHT = "insight"  # Derived understanding or pattern
    PLAN = "plan"  # Future intention or strategy
    QUESTION = "question"  # Posed question or uncertainty
    SUMMARY = "summary"  # Condensed information
    REFLECTION = "reflection"  # Meta-cognitive analysis (distinct from thought type)
    SKILL = "skill"  # Learned capability (like procedural)
    PROCEDURE = "procedure"  # Step-by-step method
    PATTERN = "pattern"  # Recognized recurring structure
    CODE = "code"  # Code snippet
    JSON = "json"  # Structured JSON data
    URL = "url"  # A web URL
    TEXT = "text"  # Generic text block (fallback)
    # Retain IMAGE? Needs blob storage/linking capability. Deferred.


class LinkType(str, Enum):
    """Types of associations between memories (from cognitive_memory)."""

    RELATED = "related"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TASK = "task"
    REFERENCES = "references"  # Added for linking thoughts/actions to memories


# ======================================================
# Database Schema (Defined as Individual Statements)
# ======================================================

# List of SQL statements for base schema setup (excluding ALTERs for deferred FKs)
SCHEMA_STATEMENTS = [
    # Pragmas first
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA cache_size=-32000;",
    "PRAGMA mmap_size=2147483647;",
    "PRAGMA busy_timeout=30000;",
    # Tables (in an order that respects simple inline FKs)
    """CREATE TABLE IF NOT EXISTS workflows (
        workflow_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT, goal TEXT, status TEXT NOT NULL,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
        parent_workflow_id TEXT, metadata TEXT, last_active INTEGER
    );""",
    """CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, parent_action_id TEXT, action_type TEXT NOT NULL,
        title TEXT, reasoning TEXT, tool_name TEXT, tool_args TEXT, tool_result TEXT, status TEXT NOT NULL,
        started_at INTEGER NOT NULL, completed_at INTEGER, sequence_number INTEGER,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (parent_action_id) REFERENCES actions(action_id) ON DELETE SET NULL
    );""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, artifact_type TEXT NOT NULL,
        name TEXT NOT NULL, description TEXT, path TEXT, content TEXT, metadata TEXT,
        created_at INTEGER NOT NULL, is_output BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL
    );""",
    """CREATE TABLE IF NOT EXISTS thought_chains (
        thought_chain_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, title TEXT NOT NULL, created_at INTEGER NOT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL
    );""",
    # ───────────────── embeddings ─────────────────
    """CREATE TABLE IF NOT EXISTS embeddings (
        id          TEXT PRIMARY KEY,
        memory_id   TEXT UNIQUE
                    REFERENCES memories(memory_id)
                    ON DELETE CASCADE
                    DEFERRABLE INITIALLY DEFERRED,
        model       TEXT    NOT NULL,
        embedding   BLOB    NOT NULL,
        dimension   INTEGER NOT NULL,
        created_at  INTEGER NOT NULL
    );""",
    # ───────────────── thoughts ───────────────────
    """CREATE TABLE IF NOT EXISTS thoughts (
        thought_id      TEXT PRIMARY KEY,
        thought_chain_id TEXT NOT NULL
                        REFERENCES thought_chains(thought_chain_id)
                        ON DELETE CASCADE,
        parent_thought_id TEXT
                        REFERENCES thoughts(thought_id)
                        ON DELETE SET NULL,
        thought_type    TEXT NOT NULL,
        content         TEXT NOT NULL,
        sequence_number INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        relevant_action_id   TEXT
                        REFERENCES actions(action_id)
                        ON DELETE SET NULL,
        relevant_artifact_id TEXT
                        REFERENCES artifacts(artifact_id)
                        ON DELETE SET NULL,
        relevant_memory_id   TEXT
                        REFERENCES memories(memory_id)
                        ON DELETE SET NULL
                        DEFERRABLE INITIALLY DEFERRED
    );""",
    # ───────────────── memories ───────────────────
    """CREATE TABLE IF NOT EXISTS memories (
        memory_id   TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL
                    REFERENCES workflows(workflow_id)
                    ON DELETE CASCADE,
        content     TEXT    NOT NULL,
        memory_level TEXT   NOT NULL,
        memory_type TEXT    NOT NULL,
        importance  REAL    DEFAULT 5.0,
        confidence  REAL    DEFAULT 1.0,
        description TEXT,
        reasoning   TEXT,
        source      TEXT,
        context     TEXT,
        tags        TEXT,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL,
        last_accessed INTEGER,
        access_count INTEGER DEFAULT 0,
        ttl         INTEGER DEFAULT 0,
        embedding_id TEXT
                    REFERENCES embeddings(id)
                    ON DELETE SET NULL,
        action_id   TEXT
                    REFERENCES actions(action_id)
                    ON DELETE SET NULL,
        thought_id  TEXT
                    REFERENCES thoughts(thought_id)
                    ON DELETE SET NULL
                    DEFERRABLE INITIALLY DEFERRED,
        artifact_id TEXT
                    REFERENCES artifacts(artifact_id)
                    ON DELETE SET NULL
    );""",
    # Memory Links (Inline FKs are fine)
    """CREATE TABLE IF NOT EXISTS memory_links (
        link_id TEXT PRIMARY KEY, source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL,
        link_type TEXT NOT NULL, strength REAL DEFAULT 1.0, description TEXT, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        FOREIGN KEY (target_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        UNIQUE(source_memory_id, target_memory_id, link_type)
    );""",
    # Tags and Junction Tables
    """CREATE TABLE IF NOT EXISTS tags (
        tag_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, description TEXT, category TEXT, created_at INTEGER NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS workflow_tags (
        workflow_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (workflow_id, tag_id),
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS action_tags (
        action_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (action_id, tag_id),
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS artifact_tags (
        artifact_id TEXT NOT NULL, tag_id INTEGER NOT NULL, PRIMARY KEY (artifact_id, tag_id),
        FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
    );""",
    # Dependencies Table
    """CREATE TABLE IF NOT EXISTS dependencies (
        dependency_id INTEGER PRIMARY KEY AUTOINCREMENT, source_action_id TEXT NOT NULL, target_action_id TEXT NOT NULL,
        dependency_type TEXT NOT NULL, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        FOREIGN KEY (target_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        UNIQUE(source_action_id, target_action_id, dependency_type)
    );""",
    # Cognitive States, Reflections, Memory Operations Log
    """CREATE TABLE IF NOT EXISTS cognitive_states (
        state_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        title TEXT NOT NULL,
        working_memory TEXT, -- Stores list of memory IDs as JSON
        focus_areas TEXT,    -- Stores list of memory IDs or topics as JSON
        context_actions TEXT,-- Stores list of action IDs as JSON
        current_goals TEXT,  -- Stores list of thought IDs or goal strings as JSON
        created_at INTEGER NOT NULL,
        is_latest BOOLEAN NOT NULL,
        focal_memory_id TEXT 
                        REFERENCES memories(memory_id)
                        ON DELETE SET NULL,
        last_active INTEGER,                        
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS reflections (
        reflection_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
        reflection_type TEXT NOT NULL, created_at INTEGER NOT NULL, referenced_memories TEXT,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
    );""",
    """CREATE TABLE IF NOT EXISTS memory_operations (
        operation_log_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, memory_id TEXT, action_id TEXT,
        operation TEXT NOT NULL, operation_data TEXT, timestamp INTEGER NOT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE SET NULL,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL
    );""",
    # Indices
    "CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_parent ON workflows(parent_workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_last_active ON workflows(last_active DESC);",
    "CREATE INDEX IF NOT EXISTS idx_actions_workflow_id ON actions(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_parent ON actions(parent_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_sequence ON actions(workflow_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_workflow_id ON artifacts(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_action_id ON artifacts(action_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);",
    "CREATE INDEX IF NOT EXISTS idx_thought_chains_workflow ON thought_chains(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_chain ON thoughts(thought_chain_id);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_sequence ON thoughts(thought_chain_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_type ON thoughts(thought_type);",
    "CREATE INDEX IF NOT EXISTS idx_thoughts_relevant_memory ON thoughts(relevant_memory_id);",  # Index still useful even if FK removed initially
    "CREATE INDEX IF NOT EXISTS idx_memories_workflow ON memories(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(memory_level);",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_action_id ON memories(action_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_thought_id ON memories(thought_id);",  # Index still useful
    "CREATE INDEX IF NOT EXISTS idx_memories_artifact_id ON memories(artifact_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_type ON memory_links(link_type);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_memory_id ON embeddings(memory_id);",  # Index FK column
    "CREATE INDEX IF NOT EXISTS idx_embeddings_dimension ON embeddings(dimension);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_workflow ON cognitive_states(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_latest ON cognitive_states(workflow_id, is_latest);",
    "CREATE INDEX IF NOT EXISTS idx_reflections_workflow ON reflections(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_workflow ON memory_operations(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_memory ON memory_operations(memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON memory_operations(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);",
    "CREATE INDEX IF NOT EXISTS idx_workflow_tags ON workflow_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_action_tags ON action_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_artifact_tags ON artifact_tags(tag_id);",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_action_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_states_last_active ON cognitive_states(last_active DESC);",

    # Virtual Table
    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        content, description, reasoning, tags,
        workflow_id UNINDEXED, memory_id UNINDEXED,
        content='memories', content_rowid='rowid', tokenize='porter unicode61'
    );""",
    # Triggers
    """CREATE TRIGGER IF NOT EXISTS memories_after_insert AFTER INSERT ON memories BEGIN
        INSERT INTO memory_fts(rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES (new.rowid, new.content, new.description, new.reasoning, new.tags, new.workflow_id, new.memory_id);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_delete AFTER DELETE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES ('delete', old.rowid, old.content, old.description, old.reasoning, old.tags, old.workflow_id, old.memory_id);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_update AFTER UPDATE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES ('delete', old.rowid, old.content, old.description, old.reasoning, old.tags, old.workflow_id, old.memory_id);
        INSERT INTO memory_fts(rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES (new.rowid, new.content, new.description, new.reasoning, new.tags, new.workflow_id, new.memory_id);
    END;""",
]

def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    # Ensure slicing doesn't go out of bounds if string is shorter than length
    return s[: min(length, len(s))]

class DBConnection:
    """Context manager for database connections using aiosqlite."""

    global SCHEMA_STATEMENTS  # Reference the list of statements
    _instance: Optional[Any] = None
    _lock = asyncio.Lock()
    _db_path_used: Optional[str] = None
    _init_lock_timeout = 15.0

    def __init__(self, db_path: str = agent_memory_config.db_path):
        self.db_path = db_path
        self.conn: Optional[Any] = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _initialize_instance(self) -> Any:
        """Handles the actual creation and setup of the database connection."""
        logger.info(f"Connecting to database: {self.db_path}", emoji_key="database")
        conn = await aiosqlite.connect(
            self.db_path,
            timeout=agent_memory_config.connection_timeout,
            # isolation_level=None # Let aiosqlite manage transactions by default
        )
        conn.row_factory = aiosqlite.Row

        # --- Apply critical PRAGMAs *before* any transaction starts ---
        # These often need exclusive access or cannot be run mid-transaction
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA foreign_keys = ON;")
        # Apply potentially less critical PRAGMAs (can often be set anytime)
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA temp_store=MEMORY;")
        await conn.execute("PRAGMA cache_size=-32000;")
        await conn.execute("PRAGMA mmap_size=2147483647;")
        await conn.execute("PRAGMA busy_timeout=30000;")
        logger.debug("Applied database PRAGMAs.")

        # --- Enable custom functions ---
        await conn.create_function("json_contains", 2, _json_contains, deterministic=True)
        await conn.create_function("json_contains_any", 2, _json_contains_any, deterministic=True)
        await conn.create_function("json_contains_all", 2, _json_contains_all, deterministic=True)
        await conn.create_function(
            "compute_memory_relevance", 5, _compute_memory_relevance, deterministic=True
        )

        # Check if DB needs initialization
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflows'"
        )
        table_exists = await cursor.fetchone()
        await cursor.close()

        if not table_exists:
            logger.info("Database schema not found. Initializing step-by-step...", emoji_key="gear")
            # --- Execute schema statements one by one within a transaction ---
            try:
                await conn.execute("BEGIN TRANSACTION;")  # Start transaction for CREATEs
                # Iterate through SCHEMA_STATEMENTS, *skipping* the PRAGMAs we already executed
                for i, stmt in enumerate(SCHEMA_STATEMENTS):
                    # Skip pragmas already set
                    if stmt.strip().upper().startswith("PRAGMA"):
                        continue
                    logger.debug(
                        f"Executing schema statement {i + 1}/{len(SCHEMA_STATEMENTS)}: {stmt[:100]}..."
                    )
                    await conn.execute(stmt)
                await conn.commit()  # Commit base schema creation
                logger.info("Base schema created successfully.")
            except aiosqlite.Error as e:
                logger.error(f"FAILED during base schema creation: {e}", exc_info=True)
                await conn.rollback()
                raise ToolError(f"Failed to create base database schema: {e}") from e

            logger.success(
                "Full database schema initialized successfully.", emoji_key="white_check_mark"
            )
        else:
            logger.info("Database schema already exists.", emoji_key="database")
            # Ensure FKs are on for existing connections too (redundant but safe)
            await conn.execute("PRAGMA foreign_keys = ON;")

        DBConnection._db_path_used = self.db_path
        return conn

    async def __aenter__(self) -> aiosqlite.Connection:
        """Acquires the singleton database connection instance."""
        # 1. Quick check without lock
        instance = DBConnection._instance
        if instance is not None:
            # Path consistency check for singleton reuse
            if self.db_path != DBConnection._db_path_used:
                logger.error(
                    f"DBConnection singleton mismatch: Already initialized with path '{DBConnection._db_path_used}', but requested '{self.db_path}'."
                )
                raise RuntimeError(
                    f"DBConnection singleton initialized with path '{DBConnection._db_path_used}', requested '{self.db_path}'"
                )
            # Ensure foreign keys are enabled for this specific use of the connection
            # Doing this on every enter ensures it's set for the current operation context
            await instance.execute("PRAGMA foreign_keys = ON;")
            return instance

        # 2. Acquire lock with timeout only if instance might need initialization
        try:
            # Use asyncio.timeout for the lock acquisition itself
            async with asyncio.timeout(DBConnection._init_lock_timeout):
                async with DBConnection._lock:
                    # 3. Double-check instance after acquiring lock
                    if DBConnection._instance is None:
                        # Call the separate initialization method
                        DBConnection._instance = await self._initialize_instance()
                    # Re-check path consistency inside lock to handle race condition if multiple threads tried init
                    elif self.db_path != DBConnection._db_path_used:
                        logger.error(
                            f"DBConnection singleton mismatch detected inside lock: Already initialized with path '{DBConnection._db_path_used}', but requested '{self.db_path}'."
                        )
                        raise RuntimeError(
                            f"DBConnection singleton initialized with path '{DBConnection._db_path_used}', requested '{self.db_path}'"
                        )

        except asyncio.TimeoutError:
            # Log timeout error and raise a ToolError
            logger.error(
                f"Timeout acquiring DB initialization lock after {DBConnection._init_lock_timeout}s. Possible deadlock or hang.",
                emoji_key="alarm_clock",
            )
            raise ToolError("Database initialization timed out.") from None
        except Exception as init_err:
            # Catch potential errors during _initialize_instance
            logger.error(
                f"Error during database initialization: {init_err}", exc_info=True, emoji_key="x"
            )
            # Ensure instance is None if initialization failed
            DBConnection._instance = None
            DBConnection._db_path_used = None
            raise ToolError(f"Database initialization failed: {init_err}") from init_err

        # Ensure FKs enabled for the first use after initialization and return the instance
        # Note: _initialize_instance also sets PRAGMA foreign_keys=ON, but setting it again here
        # ensures it's applied for the context manager's immediate use.
        await DBConnection._instance.execute("PRAGMA foreign_keys = ON;")
        return DBConnection._instance

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Releases the connection context (but doesn't close singleton).

        Propagates exceptions that occurred within the context.
        """
        # Note: This context manager manages access, not the lifecycle of the singleton connection.
        # Closing is handled by the explicit close_connection method.
        # The transaction manager context handles commit/rollback.
        if exc_type is not None:
            # Log the error that occurred *within* the 'async with DBConnection(...)' block
            logger.error(
                f"Database error occurred within DBConnection context: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )
            # Re-raise the exception to notify the caller
            raise exc_val
        pass  # If no exception, just pass (commit/rollback handled by transaction manager)

    @classmethod
    async def close_connection(cls):
        """Closes the singleton database connection if it exists.

        This method should be called explicitly by the application during shutdown
        to ensure resources are released cleanly.
        """
        if cls._instance:
            async with cls._lock:  # Ensure exclusive access for closing
                if cls._instance:  # Double check after lock
                    logger.info("Attempting to close database connection.", emoji_key="lock")
                    try:
                        await cls._instance.close()
                        logger.success(
                            "Database connection closed successfully.", emoji_key="white_check_mark"
                        )
                    except Exception as e:
                        logger.error(f"Error closing database connection: {e}", exc_info=True)
                    finally:
                        # Ensure the instance reference is cleared even if close fails
                        cls._instance = None
                        cls._db_path_used = None
                else:
                    logger.info(
                        "Database connection was closed by another task while waiting for lock."
                    )
        else:
            logger.info("No active database connection instance to close.")

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Provides an atomic transaction block using the singleton connection."""
        conn = await self.__aenter__()  # Acquire the connection instance
        try:
            # Explicitly BEGIN transaction. aiosqlite defaults might differ.
            # Using DEFERRED is generally fine unless immediate locking is needed.
            await conn.execute("BEGIN DEFERRED TRANSACTION")
            logger.debug("DB Transaction Started.")
            yield conn  # Provide the connection to the 'async with' block
        except Exception as e:
            logger.error(f"Exception during transaction, rolling back: {e}", exc_info=True)
            await conn.rollback()
            logger.warning("DB Transaction Rolled Back.", emoji_key="rewind")
            raise  # Re-raise the exception after rollback
        else:
            await conn.commit()
            logger.debug("DB Transaction Committed.")
        finally:
            # __aexit__ for the base DBConnection doesn't close the connection,
            # so we don't need to call it explicitly here. The transaction is finished.
            pass


# Custom SQLite helper functions
def _json_contains(json_text, search_value):
    if not json_text:
        return False
    try:
        return (
            search_value in json.loads(json_text)
            if isinstance(json.loads(json_text), list)
            else False
        )
    except Exception:
        return False


def _json_contains_any(json_text, search_values_json):
    if not json_text or not search_values_json:
        return False
    try:
        data = json.loads(json_text)
        search_values = json.loads(search_values_json)
        if not isinstance(data, list) or not isinstance(search_values, list):
            return False
        return any(value in data for value in search_values)
    except Exception:
        return False


def _json_contains_all(json_text, search_values_json):
    if not json_text or not search_values_json:
        return False
    try:
        data = json.loads(json_text)
        search_values = json.loads(search_values_json)
        if not isinstance(data, list) or not isinstance(search_values, list):
            return False
        return all(value in data for value in search_values)
    except Exception:
        return False


def _compute_memory_relevance(importance, confidence, created_at, access_count, last_accessed):
    """Computes a relevance score based on multiple factors. Uses Unix Timestamps."""
    now = time.time()
    age_hours = (now - created_at) / 3600 if created_at else 0
    recency_factor = 1.0 / (
        1.0 + (now - (last_accessed or created_at)) / 86400
    )  # Use created_at if never accessed

    decayed_importance = max(0, importance * (1.0 - agent_memory_config.memory_decay_rate * age_hours))
    usage_boost = min(1.0 + (access_count / 10.0), 2.0) if access_count else 1.0

    relevance = decayed_importance * usage_boost * confidence * recency_factor
    return min(max(relevance, 0.0), 10.0)


# ======================================================
# Utilities
# ======================================================

def to_iso_z(ts: float) -> str:  # helper ⇒  ISO‑8601 with trailing “Z”
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

def safe_format_timestamp(ts_value):
    """Safely formats a timestamp value (int, float, or ISO string) to ISO Z format."""
    if isinstance(ts_value, (int, float)):
        try:
            # Ensure it's not an extremely large number that might not be a valid timestamp
            if abs(ts_value) > 2**40: # Arbitrary large number check
                 logger.warning(f"Numeric timestamp {ts_value} seems out of range, returning as string.")
                 return str(ts_value)
            return to_iso_z(ts_value)
        except (OverflowError, OSError, ValueError, TypeError) as e:
            logger.warning(f"Failed to convert numeric timestamp {ts_value} to ISO: {e}")
            return str(ts_value) # Fallback to string representation of number
    elif isinstance(ts_value, str):
        # Try to parse and reformat to ensure consistency, but return original if parsing fails
        try:
            # Attempt parsing, assuming it might already be close to ISO
            dt_obj = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            # Reformat to our standard Z format
            return to_iso_z(dt_obj.timestamp())
        except ValueError:
            # If parsing fails, return the original string but log a warning
            logger.debug(f"Timestamp value '{ts_value}' is a string but not valid ISO format. Returning as is.")
            return ts_value
    elif ts_value is None:
        return None
    else:
        logger.warning(f"Unexpected timestamp type {type(ts_value)}, value: {ts_value}. Returning string representation.")
        return str(ts_value)

class MemoryUtils:
    """Utility methods for memory operations."""

    @staticmethod
    def generate_id() -> str:
        """Generate a unique UUID V4 string for database records."""
        return str(uuid.uuid4())

    @staticmethod
    async def serialize(obj: Any) -> Optional[str]:
        """Safely serialize an arbitrary Python object to a JSON string.

        Handles potential serialization errors and very large objects.
        Attempts to represent complex objects that fail direct serialization.
        If the final JSON string exceeds MAX_TEXT_LENGTH, it returns a
        JSON object indicating truncation.

        Args:
            obj: The Python object to serialize.

        Returns:
            A JSON string representation, or None if the input is None.
            Returns a specific error JSON structure if serialization fails or
            if the resulting JSON string exceeds MAX_TEXT_LENGTH.
        """
        if obj is None:
            return None
 
        # Define max_len *before* the try block using the loaded config
        try:
            max_len = agent_memory_config.max_text_length
        except NameError:
            # Fallback if agent_memory_config isn't loaded somehow (shouldn't happen)
            print("CRITICAL WARNING: agent_memory_config not loaded in serialize, using default max_len") # Use print as logger might not be ready
            max_len = 64000

        json_str = None  # Initialize variable

        try:
            # Attempt direct JSON serialization with reasonable defaults
            # Use default=str as a basic fallback for common non-serializable types like datetime
            json_str = json.dumps(obj, ensure_ascii=False, default=str)

        except TypeError as e:
            # Handle objects that are not directly serializable (like sets, custom classes)
            logger.debug(
                f"Direct JSON serialization failed for type {type(obj)}: {e}. Trying fallback."
            )
            try:
                # Attempt a fallback using string representation
                fallback_repr = str(obj)
                # Use MAX_TEXT_LENGTH from config
                max_len = agent_memory_config.max_text_length
                # Ensure fallback doesn't exceed limits either, using robust UTF-8 handling
                fallback_bytes = fallback_repr.encode("utf-8")
                if len(fallback_bytes) > max_len:
                    # Truncate the byte representation
                    truncated_bytes = fallback_bytes[:max_len]
                    # Decode back to string, replacing invalid byte sequences caused by truncation
                    truncated_repr = truncated_bytes.decode("utf-8", errors="replace")

                    # Optional refinement: Check if the last character is the replacement char (U+FFFD)
                    # If so, try truncating one byte less to avoid splitting a multi-byte char right at the end.
                    # This is a heuristic and might not always be perfect but can improve readability.
                    if truncated_repr.endswith("\ufffd") and max_len > 1:
                        # Try decoding one byte less
                        shorter_repr = fallback_bytes[: max_len - 1].decode(
                            "utf-8", errors="replace"
                        )
                        # If the shorter version *doesn't* end with the replacement character, use it.
                        if not shorter_repr.endswith("\ufffd"):
                            truncated_repr = shorter_repr

                    truncated_repr += "[TRUNCATED]"  # Add ellipsis to indicate truncation
                    logger.warning(
                        f"Fallback string representation truncated for type {type(obj)}."
                    )
                else:
                    # No truncation needed for the fallback string itself
                    truncated_repr = fallback_repr

                # Create the JSON string containing the error and the (potentially truncated) fallback
                json_str = json.dumps(
                    {
                        "error": f"Serialization failed for type {type(obj)}.",
                        "fallback_repr": truncated_repr,  # Store the safely truncated string representation
                    },
                    ensure_ascii=False,
                )

            except Exception as fallback_e:
                # Final fallback if even string conversion fails
                logger.error(
                    f"Could not serialize object of type {type(obj)} even with fallback: {fallback_e}",
                    exc_info=True,
                )
                json_str = json.dumps(
                    {
                        "error": f"Unserializable object type {type(obj)}. Fallback failed.",
                        "critical_error": str(fallback_e),
                    },
                    ensure_ascii=False,
                )

        # --- Check final length AFTER serialization attempt (success or fallback) ---
        # Ensure json_str is assigned before checking length
        if json_str is None:
            # This case should theoretically not be reached if the logic above is sound,
            # but added as a safeguard. It implies an unexpected path where serialization
            # didn't succeed but also didn't fall into the error handlers properly.
            logger.error(
                f"Internal error: json_str is None after serialization attempt for object of type {type(obj)}"
            )
            return json.dumps(
                {
                    "error": "Internal serialization error occurred.",
                    "original_type": str(type(obj)),
                },
                ensure_ascii=False,
            )

        # Check final length against MAX_TEXT_LENGTH (bytes)
        final_bytes = json_str.encode("utf-8")
        if len(final_bytes) > max_len:
            # If the generated JSON (even if it's an error JSON from fallback) is too long,
            # return a standard "too long" error marker with a preview.
            logger.warning(
                f"Serialized JSON string exceeds max length ({max_len} bytes). Returning truncated indicator."
            )
            # Provide a preview of the oversized JSON string
            preview_str = json_str[:200] + ("..." if len(json_str) > 200 else "")
            return json.dumps(
                {
                    "error": "Serialized content exceeded maximum length.",
                    "original_type": str(type(obj)),
                    "preview": preview_str,  # Provide a small preview of the oversized content
                },
                ensure_ascii=False,
            )
        else:
            # Return the valid JSON string if within limits
            return json_str

    @staticmethod
    async def deserialize(json_str: Optional[str]) -> Any:
        """Safely deserialize a JSON string back into a Python object.

        Handles None input and potential JSON decoding errors. If decoding fails,
        it returns the original string, assuming it might not have been JSON
        in the first place (e.g., a truncated representation).
        """
        if json_str is None:
            return None
        if not json_str.strip():  # Handle empty strings
            return None
        try:
            # Attempt to load the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If it fails, log the issue and return the original string
            # This might happen if the string stored was an error message or truncated data
            logger.debug(
                f"Failed to deserialize JSON: {e}. Content was: '{json_str[:100]}...'. Returning raw string."
            )
            return json_str
        except Exception as e:
            # Catch other potential errors during deserialization
            logger.error(
                f"Unexpected error deserializing JSON: {e}. Content: '{json_str[:100]}...'",
                exc_info=True,
            )
            return json_str  # Return original string as fallback

    @staticmethod
    def _validate_sql_identifier(identifier: str, identifier_type: str = "column/table") -> str:
        """Validates a string intended for use as an SQL table or column name.

        Prevents SQL injection by ensuring the identifier only contains
        alphanumeric characters and underscores. Raises ToolInputError if invalid.

        Args:
            identifier: The string to validate.
            identifier_type: A description of what the identifier represents (for error messages).

        Returns:
            The validated identifier if it's safe.

        Raises:
            ToolInputError: If the identifier is invalid.
        """
        # Simple regex: Allows letters, numbers, and underscores. Must start with a letter or underscore.
        # Adjust regex if more complex identifiers (e.g., quoted) are needed, but keep it strict.
        if not identifier or not re.fullmatch(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            logger.error(f"Invalid SQL identifier provided: '{identifier}'")
            raise ToolInputError(
                f"Invalid {identifier_type} name provided. Must be alphanumeric/underscore.",
                param_name=identifier_type,
            )
        # Optional: Check against a known allowlist of tables/columns if possible
        # known_tables = {"actions", "thoughts", "memories", ...}
        # if identifier_type == "table" and identifier not in known_tables:
        #     raise ToolInputError(f"Unknown table name provided: {identifier}", param_name=identifier_type)
        return identifier

    @staticmethod
    async def get_next_sequence_number(
        conn: aiosqlite.Connection, parent_id: str, table: str, parent_col: str
    ) -> int:
        """Get the next sequence number for ordering items within a parent scope.

        Args:
            conn: The database connection.
            parent_id: The ID of the parent entity (e.g., workflow_id, thought_chain_id).
            table: The name of the table containing the sequence number (e.g., 'actions', 'thoughts').
            parent_col: The name of the column linking to the parent entity.

        Returns:
            The next available integer sequence number (starting from 1).
        """
        # --- Validate dynamic identifiers to prevent SQL injection ---
        validated_table = MemoryUtils._validate_sql_identifier(table, "table")
        validated_parent_col = MemoryUtils._validate_sql_identifier(parent_col, "parent_col")
        # --- End Validation ---

        # Use validated identifiers in the f-string
        sql = f"SELECT MAX(sequence_number) FROM {validated_table} WHERE {validated_parent_col} = ?"
        # Use execute directly on the connection for context management
        async with conn.execute(sql, (parent_id,)) as cursor:
            row = await cursor.fetchone()
            # If no rows exist (row is None) or MAX is NULL, start at 1. Otherwise, increment max.
            # Access by index as row might be None or a tuple/row object
            max_sequence = row[0] if row and row[0] is not None else 0
            return max_sequence + 1

    @staticmethod
    async def process_tags(
        conn: aiosqlite.Connection, entity_id: str, tags: List[str], entity_type: str
    ) -> None:
        """Ensures tags exist in the 'tags' table and associates them with a given entity
           in the appropriate junction table (e.g., 'workflow_tags').

        Args:
            conn: The database connection.
            entity_id: The ID of the entity (workflow, action, artifact).
            tags: A list of tag names (strings) to associate. Duplicates are handled.
            entity_type: The type of the entity ('workflow', 'action', 'artifact'). Must form valid SQL identifiers when combined with '_tags' or '_id'.
        """
        if not tags:
            return  # Nothing to do if no tags are provided

        # Validate entity_type first as it forms part of identifiers
        # Allow only specific expected entity types
        allowed_entity_types = {"workflow", "action", "artifact"}
        if entity_type not in allowed_entity_types:
            raise ToolInputError(
                f"Invalid entity_type for tagging: {entity_type}", param_name="entity_type"
            )

        # Define and validate dynamic identifiers
        junction_table_name = f"{entity_type}_tags"
        id_column_name = f"{entity_type}_id"
        validated_junction_table = MemoryUtils._validate_sql_identifier(
            junction_table_name, "junction_table"
        )
        validated_id_column = MemoryUtils._validate_sql_identifier(id_column_name, "id_column")
        # --- End Validation ---

        tag_ids_to_link = []
        unique_tags = list(
            set(str(tag).strip().lower() for tag in tags if str(tag).strip())
        )  # Clean, lowercase, unique tags
        now_unix = int(time.time())

        if not unique_tags:
            return  # Nothing to do if tags are empty after cleaning

        # Ensure all unique tags exist in the 'tags' table and get their IDs
        for tag_name in unique_tags:
            # Attempt to insert the tag, ignoring if it already exists
            await conn.execute(
                """
                INSERT INTO tags (name, created_at) VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING;
                """,
                (tag_name, now_unix),
            )
            # Retrieve the tag_id (whether newly inserted or existing)
            cursor = await conn.execute("SELECT tag_id FROM tags WHERE name = ?", (tag_name,))
            row = await cursor.fetchone()
            await cursor.close()  # Close cursor

            if row:
                tag_ids_to_link.append(row["tag_id"])
            else:
                # This should ideally not happen due to the upsert logic, but log if it does
                logger.warning(f"Could not find or create tag_id for tag: {tag_name}")

        # Link the retrieved tag IDs to the entity in the junction table
        if tag_ids_to_link:
            link_values = [(entity_id, tag_id) for tag_id in tag_ids_to_link]
            # Use INSERT OR IGNORE to handle potential race conditions or duplicate calls gracefully
            # Use validated identifiers in the f-string
            await conn.executemany(
                f"INSERT OR IGNORE INTO {validated_junction_table} ({validated_id_column}, tag_id) VALUES (?, ?)",
                link_values,
            )
            logger.debug(f"Associated {len(link_values)} tags with {entity_type} {entity_id}")

    @staticmethod
    async def _log_memory_operation(
        conn: aiosqlite.Connection,
        workflow_id: str,
        operation: str,
        memory_id: Optional[str] = None,
        action_id: Optional[str] = None,
        operation_data: Optional[Dict] = None,
    ):
        """Logs an operation related to memory management or agent activity. Internal helper."""
        try:
            op_id = MemoryUtils.generate_id()
            timestamp_unix = int(time.time())
            # Serialize operation_data carefully using the updated serialize method
            op_data_json = (
                await MemoryUtils.serialize(operation_data) if operation_data is not None else None
            )

            await conn.execute(
                """
                INSERT INTO memory_operations
                (operation_log_id, workflow_id, memory_id, action_id, operation, operation_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (op_id, workflow_id, memory_id, action_id, operation, op_data_json, timestamp_unix),
            )
        except Exception as e:
            # Log failures robustly, don't let logging break main logic
            logger.error(
                f"CRITICAL: Failed to log memory operation '{operation}': {e}", exc_info=True
            )

    @staticmethod
    async def _update_memory_access(conn: aiosqlite.Connection, memory_id: str):
        """Updates the last_accessed timestamp and increments access_count for a memory. Internal helper."""
        now_unix = int(time.time())
        try:
            # Use COALESCE to handle the first access correctly
            await conn.execute(
                """
                UPDATE memories
                SET last_accessed = ?,
                    access_count = COALESCE(access_count, 0) + 1
                WHERE memory_id = ?
                """,
                (now_unix, memory_id),
            )
        except Exception as e:
            logger.warning(
                f"Failed to update memory access stats for {memory_id}: {e}", exc_info=True
            )


# ======================================================
# Embedding Service Integration & Semantic Search Logic
# ======================================================


async def _store_embedding(conn: aiosqlite.Connection, memory_id: str, text: str) -> Optional[str]:
    """Generates and stores an embedding for a memory using the EmbeddingService.

    Args:
        conn: Database connection.
        memory_id: ID of the memory.
        text: Text content to generate embedding for (often content + description).

    Returns:
        ID of the stored embedding record in the embeddings table, or None if failed.
    """
    try:
        embedding_service = get_embedding_service()  # Get singleton instance
        if not embedding_service.client:  # Check if service was initialized correctly (has client)
            logger.warning(
                "EmbeddingService client not available. Cannot generate embedding.",
                emoji_key="warning",
            )
            return None

        # Generate embedding using the service (handles caching internally)
        embedding_list = await embedding_service.create_embeddings(texts=[text])
        if not embedding_list or not embedding_list[0]:  # Extra check for empty embedding
            logger.warning(f"Failed to generate embedding for memory {memory_id}")
            return None
        embedding_array = np.array(embedding_list[0], dtype=np.float32)  # Ensure consistent dtype
        if embedding_array.size == 0:
            logger.warning(f"Generated embedding is empty for memory {memory_id}")
            return None

        # Get the embedding dimension
        embedding_dimension = embedding_array.shape[0]

        # Generate a unique ID for this embedding entry in our DB table
        embedding_db_id = MemoryUtils.generate_id()
        embedding_bytes = embedding_array.tobytes()
        model_used = embedding_service.model_name

        # Store embedding in our DB
        await conn.execute(
            """
            INSERT INTO embeddings (id, memory_id, model, embedding, dimension, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                id = excluded.id,
                model = excluded.model,
                embedding = excluded.embedding,
                dimension = excluded.dimension,
                created_at = excluded.created_at
            """,
            (
                embedding_db_id,
                memory_id,
                model_used,
                embedding_bytes,
                embedding_dimension,
                int(time.time()),
            ),
        )
        # Update the memory record to link to this *embedding table entry ID*
        # Note: The cognitive_memory schema had embedding_id as FK to embeddings.id
        # We will store embedding_db_id here.
        await conn.execute(
            "UPDATE memories SET embedding_id = ? WHERE memory_id = ?", (embedding_db_id, memory_id)
        )

        logger.debug(
            f"Stored embedding {embedding_db_id} (Dim: {embedding_dimension}) for memory {memory_id}"
        )
        return embedding_db_id  # Return the ID of the row in the embeddings table

    except Exception as e:
        logger.error(f"Failed to store embedding for memory {memory_id}: {e}", exc_info=True)
        return None


async def _find_similar_memories(
    conn: aiosqlite.Connection,
    query_text: str,
    workflow_id: Optional[str] = None,
    limit: int = 5,
    threshold: float = agent_memory_config.similarity_threshold,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Finds memories with similar semantic meaning using embeddings stored in SQLite.
       Filters by workflow, level, type, dimension, and TTL.

    Args:
        conn: Database connection.
        query_text: Query text to find similar memories.
        workflow_id: Optional workflow ID to limit search.
        limit: Maximum number of results to return *after similarity calculation*.
        threshold: Minimum similarity score (0-1).
        memory_level: Optional memory level to filter by.
        memory_type: Optional memory type to filter by.

    Returns:
        List of tuples (memory_id, similarity_score) sorted by similarity descending.
    """
    try:
        embedding_service = get_embedding_service()
        if not embedding_service.client:
            logger.warning(
                "EmbeddingService client not available. Cannot perform semantic search.",
                emoji_key="warning",
            )
            return []

        # 1. Generate query embedding
        query_embedding_list = await embedding_service.create_embeddings(texts=[query_text])
        if not query_embedding_list or not query_embedding_list[0]:  # Extra check
            logger.warning(f"Failed to generate query embedding for: '{query_text[:50]}...'")
            return []
        query_embedding = np.array(
            query_embedding_list[0], dtype=np.float32
        )  # Ensure consistent dtype
        if query_embedding.size == 0:
            logger.warning(f"Generated query embedding is empty for: '{query_text[:50]}...'")
            return []

        query_dimension = query_embedding.shape[0]
        query_embedding_2d = query_embedding.reshape(1, -1)  # Reshape for scikit-learn

        # 2. Build query to fetch candidate embeddings from DB, including filters
        sql = """
        SELECT m.memory_id, e.embedding
        FROM memories m
        JOIN embeddings e ON m.embedding_id = e.id
        WHERE e.dimension = ?
        """
        params: List[Any] = [query_dimension]

        if workflow_id:
            sql += " AND m.workflow_id = ?"
            params.append(workflow_id)
        if memory_level:
            sql += " AND m.memory_level = ?"
            params.append(memory_level.lower())  # Ensure lowercase for comparison
        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type.lower())  # Ensure lowercase

        # Add TTL check
        now_unix = int(time.time())
        sql += " AND (m.ttl = 0 OR m.created_at + m.ttl > ?)"
        params.append(now_unix)

        # Optimization: Potentially limit candidates fetched *before* calculating all similarities
        # Fetching more candidates than `limit` allows for better ranking after similarity calculation
        candidate_limit = max(limit * 5, 50)  # Fetch more candidates than needed
        sql += " ORDER BY m.last_accessed DESC NULLS LAST LIMIT ?"  # Prioritize recently accessed
        params.append(candidate_limit)

        # 3. Fetch candidate embeddings (only those with matching dimension)
        candidates: List[Tuple[str, bytes]] = []
        async with conn.execute(sql, params) as cursor:
            candidates = await cursor.fetchall()  # Fetchall is ok for limited candidates

        if not candidates:
            logger.debug(
                f"No candidate memories found matching filters (including dimension {query_dimension}) for semantic search."
            )
            return []

        # 4. Calculate similarities for candidates
        similarities: List[Tuple[str, float]] = []
        for memory_id, embedding_bytes in candidates:
            try:
                # Deserialize embedding from bytes
                memory_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                if memory_embedding.size == 0:
                    logger.warning(f"Skipping empty embedding blob for memory {memory_id}")
                    continue

                # Reshape for scikit-learn compatibility
                memory_embedding_2d = memory_embedding.reshape(1, -1)

                # --- Safety Check: Verify dimensions again (should match due to SQL filter) ---
                # This primarily guards against database corruption or schema inconsistencies.
                if query_embedding_2d.shape[1] != memory_embedding_2d.shape[1]:
                    logger.warning(
                        f"Dimension mismatch detected for memory {memory_id} (Query: {query_embedding_2d.shape[1]}, DB: {memory_embedding_2d.shape[1]}) despite DB filter. Skipping."
                    )
                    continue
                # --- End Safety Check ---

                # Calculate cosine similarity
                similarity = sk_cosine_similarity(query_embedding_2d, memory_embedding_2d)[0][0]

                # 5. Filter by threshold
                if similarity >= threshold:
                    similarities.append((memory_id, float(similarity)))

            except Exception as e:
                logger.warning(f"Error processing embedding for memory {memory_id}: {e}")
                continue

        # 6. Sort by similarity and limit to the final requested count
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Calculated similarities for {len(candidates)} candidates (Dim: {query_dimension}). Found {len(similarities)} memories above threshold {threshold} before limiting to {limit}."
        )
        return similarities[:limit]

    except Exception as e:
        logger.error(f"Failed to find similar memories: {e}", exc_info=True)
        return []


# ======================================================
# Public Tool Functions
# ======================================================


# --- 1. Initialization ---
@with_tool_metrics
@with_error_handling
async def initialize_memory_system(db_path: str = agent_memory_config.db_path) -> Dict[str, Any]:
    """Initializes the Unified Agent Memory system and checks embedding service status.

    Creates or verifies the database schema using aiosqlite, applies optimizations,
    and attempts to initialize the singleton EmbeddingService. **Raises ToolError if
    the embedding service fails to initialize or is non-functional.**

    Args:
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Initialization status dictionary (only if successful).
        {
            "success": true,
            "message": "Unified Memory System initialized successfully.",
            "db_path": "/path/to/unified_agent_memory.db",
            "embedding_service_functional": true, # Will always be true if function returns successfully
            "embedding_service_warning": null,
            "processing_time": 0.123
        }

    Raises:
        ToolError: If database initialization fails OR if the EmbeddingService
                   cannot be initialized or lacks a functional client (e.g., missing API key).
    """
    start_time = time.time()
    logger.info("Initializing Unified Memory System...", emoji_key="rocket")
    embedding_service_warning = None  # This will now likely be part of the error message

    try:
        # Initialize/Verify Database Schema via DBConnection context manager
        async with DBConnection(db_path) as conn:
            # Perform a simple check to ensure DB connection is working
            cursor = await conn.execute("SELECT count(*) FROM workflows")
            _ = await cursor.fetchone()
            await cursor.close()  # Close cursor
            # No explicit commit needed here if using default aiosqlite behavior or autocommit
        logger.success("Unified Memory System database connection verified.", emoji_key="database")

        # Attempt to initialize/get the EmbeddingService singleton and VERIFY functionality
        try:
            # This call triggers the service's __init__ if it's the first time
            embedding_service = get_embedding_service()
            # Check if the service has its client (e.g., requires API key)
            if embedding_service.client is not None:
                logger.info("EmbeddingService initialized and functional.", emoji_key="brain")
            else:
                embedding_service_warning = (
                    "EmbeddingService client not available (check API key?). Embeddings disabled."
                )
                logger.error(embedding_service_warning, emoji_key="warning")  # Log as error
                # Raise explicit error instead of just returning False status
                raise ToolError(embedding_service_warning)
        except Exception as embed_init_err:
            # This includes the explicit ToolError raised above for missing client
            if not isinstance(
                embed_init_err, ToolError
            ):  # Avoid double wrapping if it was the specific client error
                embedding_service_warning = f"Failed to initialize EmbeddingService: {str(embed_init_err)}. Embeddings disabled."
                logger.error(embedding_service_warning, emoji_key="error", exc_info=True)
                raise ToolError(embedding_service_warning) from embed_init_err
            else:
                # Re-raise the ToolError directly if it was the specific missing client error
                raise embed_init_err

        # If we reach here, both DB and Embedding Service are functional
        processing_time = time.time() - start_time
        logger.success(
            "Unified Memory System initialized successfully (DB and Embeddings OK).",
            emoji_key="white_check_mark",
            time=processing_time,
        )

        return {
            "success": True,
            "message": "Unified Memory System initialized successfully.",
            "db_path": os.path.abspath(db_path),
            "embedding_service_functional": True,  # Will always be true if this return is reached
            "embedding_service_warning": None,  # No warning if successful
            "processing_time": processing_time,
        }
    except Exception as e:
        # This catches errors during DB initialization OR the ToolError raised from embedding failure
        processing_time = time.time() - start_time
        # Ensure it's logged as a critical failure
        logger.error(
            f"Failed to initialize memory system: {str(e)}",
            emoji_key="x",
            exc_info=True,
            time=processing_time,
        )
        # Re-raise as ToolError if it wasn't already one
        if isinstance(e, ToolError):
            raise e
        else:
            # Wrap unexpected DB errors
            raise ToolError(f"Memory system initialization failed: {str(e)}") from e


# --- 2. Workflow Management Tools ---

@with_tool_metrics
@with_error_handling
async def create_workflow(
    title: str,
    description: Optional[str] = None,
    goal: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent_workflow_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Creates a new workflow, including a default thought chain and initial goal thought if specified.

    Args:
        title: A clear, descriptive title for the workflow.
        description: (Optional) A more detailed explanation of the workflow's purpose.
        goal: (Optional) The high-level goal or objective. If provided, an initial 'goal' thought is created.
        tags: (Optional) List of keyword tags to categorize this workflow.
        metadata: (Optional) Additional structured data about the workflow.
        parent_workflow_id: (Optional) ID of a parent workflow.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing information about the created workflow and its primary thought chain.
        Timestamps are returned as ISO 8601 strings.
        {
            "workflow_id": "uuid-string",
            "title": "Workflow Title",
            "description": "...",
            "goal": "...",
            "status": "active",
            "created_at": "iso-timestampZ",
            "updated_at": "iso-timestampZ",
            "tags": ["tag1"],
            "primary_thought_chain_id": "uuid-string",
            "success": true
        }

    Raises:
        ToolInputError: If title is empty or parent workflow doesn't exist.
        ToolError: If the database operation fails.
    """
    # Validate required input
    if not title or not isinstance(title, str):
        raise ToolInputError("Workflow title must be a non-empty string", param_name="title")

    # Generate IDs and timestamps
    workflow_id = MemoryUtils.generate_id()
    now_unix = int(time.time())

    logger.debug(
        f"Inside create_workflow (wf_id={workflow_id[:8]}): Received db_path = '{db_path}'"
    )

    try:
        async with DBConnection(db_path) as conn:
            # Check parent workflow existence if provided
            if parent_workflow_id:
                cursor = await conn.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (parent_workflow_id,)
                )
                parent_exists = await cursor.fetchone()
                await cursor.close()  # Close cursor
                if not parent_exists:
                    raise ToolInputError(
                        f"Parent workflow not found: {parent_workflow_id}",
                        param_name="parent_workflow_id",
                    )

            # Serialize metadata
            metadata_json = await MemoryUtils.serialize(metadata)

            # Insert the main workflow record
            await conn.execute(
                """
                INSERT INTO workflows
                (workflow_id, title, description, goal, status, created_at, updated_at, parent_workflow_id, metadata, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    title,
                    description,
                    goal,
                    WorkflowStatus.ACTIVE.value,
                    now_unix,
                    now_unix,
                    parent_workflow_id,
                    metadata_json,
                    now_unix,
                ),
            )

            # Process and associate tags with the workflow
            await MemoryUtils.process_tags(conn, workflow_id, tags or [], "workflow")

            # Create the default thought chain associated with this workflow
            thought_chain_id = MemoryUtils.generate_id()
            chain_title = f"Main reasoning for: {title}"  # Default title
            await conn.execute(
                "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, ?, ?)",
                (thought_chain_id, workflow_id, chain_title, now_unix),
            )

            # If a goal was provided, add it as the first thought in the default chain
            if goal:
                thought_id = MemoryUtils.generate_id()
                # Get sequence number (will be 1 for the first thought)
                seq_no = await MemoryUtils.get_next_sequence_number(
                    conn, thought_chain_id, "thoughts", "thought_chain_id"
                )
                await conn.execute(
                    """
                    INSERT INTO thoughts
                    (thought_id, thought_chain_id, thought_type, content, sequence_number, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thought_id, thought_chain_id, ThoughtType.GOAL.value, goal, seq_no, now_unix),
                )

            # Commit the transaction
            await conn.commit()

            # Prepare the result dictionary, formatting timestamps for output

            result = {
                "workflow_id": workflow_id,
                "title": title,
                "description": description,
                "goal": goal,
                "status": WorkflowStatus.ACTIVE.value,
                "created_at": to_iso_z(now_unix),
                "updated_at": to_iso_z(now_unix),
                "tags": tags or [],
                "primary_thought_chain_id": thought_chain_id,  # default chain ID for the agent
                "success": True,
            }
            logger.info(
                f"Created workflow '{title}' ({workflow_id}) with primary thought chain {thought_chain_id}",
                emoji_key="clipboard",
            )
            return result

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        # Log the error and raise a generic ToolError
        logger.error(f"Error creating workflow: {e}", exc_info=True)
        raise ToolError(f"Failed to create workflow: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def update_workflow_status(
    workflow_id: str,
    status: str,
    completion_message: Optional[str] = None,
    update_tags: Optional[List[str]] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Updates the status of a workflow. (Ported from agent_memory, adapted).
    Timestamps are returned as ISO 8601 strings.
    """
    try:
        status_enum = WorkflowStatus(status.lower())
    except ValueError as e:
        valid_statuses = [s.value for s in WorkflowStatus]
        raise ToolInputError(
            f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}",
            param_name="status",
        ) from e

    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # Check existence first
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            exists = await cursor.fetchone()
            await cursor.close()
            if not exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

            update_params = [
                status_enum.value,
                now_unix,
                now_unix,
            ] 
            set_clauses = "status = ?, updated_at = ?, last_active = ?"

            if status_enum in [
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.ABANDONED,
            ]:
                set_clauses += ", completed_at = ?"
                update_params.append(now_unix)

            # Add workflow_id to params for WHERE clause
            update_params.append(workflow_id)

            await conn.execute(
                f"UPDATE workflows SET {set_clauses} WHERE workflow_id = ?", update_params
            )

            # Add completion message as thought
            if completion_message:
                cursor = await conn.execute(
                    "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                    (workflow_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row:
                    thought_chain_id = row["thought_chain_id"]
                    seq_no = await MemoryUtils.get_next_sequence_number(
                        conn, thought_chain_id, "thoughts", "thought_chain_id"
                    )
                    thought_id = MemoryUtils.generate_id()
                    thought_type = (
                        ThoughtType.SUMMARY.value
                        if status_enum == WorkflowStatus.COMPLETED
                        else ThoughtType.REFLECTION.value
                    )
                    await conn.execute(
                        "INSERT INTO thoughts (thought_id, thought_chain_id, thought_type, content, sequence_number, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            thought_id,
                            thought_chain_id,
                            thought_type,
                            completion_message,
                            seq_no,
                            now_unix,
                        ),
                    )

            # Process additional tags
            await MemoryUtils.process_tags(conn, workflow_id, update_tags or [], "workflow")
            await conn.commit()

            # Prepare the result dictionary
            result = {
                "workflow_id": workflow_id,
                "status": status_enum.value,
                "updated_at": to_iso_z(now_unix),  # Use the timestamp when the update happened
                "success": True,
            }

            # Add completed_at only if applicable
            if status_enum in [
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.ABANDONED,
            ]:
                result["completed_at"] = to_iso_z(now_unix)  # Use the same timestamp

            # Log the update *after* commit
            logger.info(
                f"Updated workflow {workflow_id} status to '{status_enum.value}'",
                emoji_key="arrows_counterclockwise",
            )

            return result  # Return the result dictionary for all cases
    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow status: {e}", exc_info=True)
        raise ToolError(f"Failed to update workflow status: {str(e)}") from e


# --- 3. Action Tracking Tools ---

@with_tool_metrics
@with_error_handling
async def record_action_start(
    workflow_id: str,
    action_type: str,
    reasoning: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    parent_action_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    related_thought_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Records the start of an action within a workflow and creates a corresponding episodic memory.

    Use this tool whenever you begin a significant step in your workflow. It logs the action details
    and automatically creates a linked memory entry summarizing the action's initiation and reasoning.

    Args:
        workflow_id: The ID of the workflow this action belongs to.
        action_type: The type of action (e.g., 'tool_use', 'reasoning', 'planning'). See ActionType enum.
        reasoning: An explanation of why this action is being taken.
        tool_name: (Optional) The name of the tool being used (required if action_type is 'tool_use').
        tool_args: (Optional) Arguments passed to the tool (used if action_type is 'tool_use').
        title: (Optional) A brief, descriptive title for this action. Auto-generated if omitted.
        parent_action_id: (Optional) ID of parent action if this is a sub-action.
        tags: (Optional) List of tags to categorize this action.
        related_thought_id: (Optional) ID of a thought that led to this action.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        A dictionary containing information about the started action and the linked memory.

    Raises:
        ToolInputError: If required parameters are missing or invalid, or referenced entities don't exist.
        ToolError: If the database operation fails.
    """
    # --- Input Validation ---
    try:
        action_type_enum = ActionType(action_type.lower())
    except ValueError as e:
        valid_types = [t.value for t in ActionType]
        raise ToolInputError(
            f"Invalid action_type '{action_type}'. Must be one of: {', '.join(valid_types)}",
            param_name="action_type",
        ) from e

    if not reasoning or not isinstance(reasoning, str):
        raise ToolInputError("Reasoning must be a non-empty string", param_name="reasoning")
    if action_type_enum == ActionType.TOOL_USE and not tool_name:
        raise ToolInputError(
            "Tool name is required for 'tool_use' action type", param_name="tool_name"
        )

    # --- Initialization ---
    action_id = MemoryUtils.generate_id()
    memory_id = MemoryUtils.generate_id()  # Pre-generate ID for the linked memory
    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # --- Existence Checks (Workflow, Parent Action, Related Thought) ---
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()
            if not wf_exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

            if parent_action_id:
                cursor = await conn.execute(
                    "SELECT 1 FROM actions WHERE action_id = ? AND workflow_id = ?",
                    (parent_action_id, workflow_id),
                )
                parent_exists = await cursor.fetchone()
                await cursor.close()
                if not parent_exists:
                    raise ToolInputError(
                        f"Parent action '{parent_action_id}' not found or does not belong to workflow '{workflow_id}'.",
                        param_name="parent_action_id",
                    )

            if related_thought_id:
                cursor = await conn.execute(
                    "SELECT 1 FROM thoughts t JOIN thought_chains tc ON t.thought_chain_id = tc.thought_chain_id WHERE t.thought_id = ? AND tc.workflow_id = ?",
                    (related_thought_id, workflow_id),
                )
                thought_exists = await cursor.fetchone()
                await cursor.close()
                if not thought_exists:
                    raise ToolInputError(
                        f"Related thought '{related_thought_id}' not found or does not belong to workflow '{workflow_id}'.",
                        param_name="related_thought_id",
                    )

            # --- Determine Action Title ---
            sequence_number = await MemoryUtils.get_next_sequence_number(
                conn, workflow_id, "actions", "workflow_id"
            )
            auto_title = title
            if not auto_title:
                if action_type_enum == ActionType.TOOL_USE and tool_name:
                    auto_title = f"Using {tool_name}"
                else:
                    first_sentence = reasoning.split(".")[0].strip()
                    auto_title = first_sentence[:50] + ("..." if len(first_sentence) > 50 else "")
            if not auto_title:  # Fallback if reasoning was very short
                auto_title = f"{action_type_enum.value.capitalize()} Action #{sequence_number}"

            # --- Insert Action Record ---
            tool_args_json = await MemoryUtils.serialize(tool_args)
            await conn.execute(
                """
                INSERT INTO actions (action_id, workflow_id, parent_action_id, action_type, title,
                reasoning, tool_name, tool_args, status, started_at, sequence_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    workflow_id,
                    parent_action_id,
                    action_type_enum.value,
                    auto_title,
                    reasoning,
                    tool_name,
                    tool_args_json,
                    ActionStatus.IN_PROGRESS.value,
                    now_unix,
                    sequence_number,
                ),
            )

            # --- Process Tags for Action ---
            await MemoryUtils.process_tags(conn, action_id, tags or [], "action")

            # --- Link Action to Related Thought ---
            if related_thought_id:
                await conn.execute(
                    "UPDATE thoughts SET relevant_action_id = ? WHERE thought_id = ?",
                    (action_id, related_thought_id),
                )

            # --- Create Linked Episodic Memory ---
            memory_content = f"Started action [{sequence_number}] '{auto_title}' ({action_type_enum.value}). Reasoning: {reasoning}"
            if tool_name:
                memory_content += f" Tool: {tool_name}."
            mem_tags = ["action_start", action_type_enum.value] + (tags or [])
            mem_tags_json = json.dumps(list(set(mem_tags)))

            await conn.execute(
                """
                 INSERT INTO memories (memory_id, workflow_id, action_id, content, memory_level, memory_type,
                 importance, confidence, tags, created_at, updated_at, access_count)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """,
                (
                    memory_id,
                    workflow_id,
                    action_id,
                    memory_content,
                    MemoryLevel.EPISODIC.value,
                    MemoryType.ACTION_LOG.value,
                    5.0,
                    1.0,
                    mem_tags_json,
                    now_unix,
                    now_unix,
                    0,
                ),
            )
            await MemoryUtils._log_memory_operation(
                conn, workflow_id, "create_from_action_start", memory_id, action_id
            )

            # --- Update Workflow Timestamp ---
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            )

            # --- Commit Transaction ---
            await conn.commit()

            # --- Prepare Result (Format timestamp for output) ---
            result = {
                "action_id": action_id,
                "workflow_id": workflow_id,
                "action_type": action_type_enum.value,
                "title": auto_title,
                "tool_name": tool_name,
                "status": ActionStatus.IN_PROGRESS.value,
                "started_at": to_iso_z(now_unix),
                "sequence_number": sequence_number,
                "tags": tags or [],
                "linked_memory_id": memory_id,
                "success": True,
            }

            logger.info(
                f"Started action '{auto_title}' ({action_id}) in workflow {workflow_id}",
                emoji_key="fast_forward",
            )

            return result

    except ToolInputError:
        raise  # Re-raise for specific handling
    except Exception as e:
        logger.error(f"Error recording action start: {e}", exc_info=True)
        raise ToolError(f"Failed to record action start: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def record_action_completion(
    action_id: str,
    status: str = "completed",
    tool_result: Optional[Any] = None,
    summary: Optional[str] = None,
    conclusion_thought: Optional[str] = None,
    conclusion_thought_type: str = "inference",  # Default type for conclusion
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Records the completion or failure of an action and updates its linked memory.

    Marks an action (previously started with record_action_start) as finished,
    stores the tool result if applicable, optionally adds a summary or concluding thought,
    and updates the corresponding 'action_log' memory entry.

    Args:
        action_id: The ID of the action to complete.
        status: (Optional) Final status: 'completed', 'failed', or 'skipped'. Default 'completed'.
        tool_result: (Optional) The result returned by the tool for 'tool_use' actions.
        summary: (Optional) A brief summary of the action's outcome or findings.
        conclusion_thought: (Optional) A thought derived from this action's completion.
        conclusion_thought_type: (Optional) Type for the conclusion thought. Default 'inference'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary confirming the action completion.
        {
            "action_id": "action-uuid",
            "workflow_id": "workflow-uuid",
            "status": "completed" | "failed" | "skipped",
            "completed_at": "iso-timestamp",
            "conclusion_thought_id": "thought-uuid" | None,
            "success": true
        }

    Raises:
        ToolInputError: If action not found or status/thought type is invalid.
        ToolError: If database operation fails.
    """
    start_time = time.time()
    # --- Validate Status ---
    try:
        status_enum = ActionStatus(status.lower())
        if status_enum not in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.SKIPPED]:
            raise ValueError("Status must indicate completion, failure, or skipping.")
    except ValueError as e:
        valid_statuses = [
            s.value for s in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.SKIPPED]
        ]
        raise ToolInputError(
            f"Invalid completion status '{status}'. Must be one of: {', '.join(valid_statuses)}",
            param_name="status",
        ) from e

    # --- Validate Thought Type (if conclusion thought provided) ---
    thought_type_enum = None
    if conclusion_thought:
        try:
            thought_type_enum = ThoughtType(conclusion_thought_type.lower())
        except ValueError as e:
            valid_types = [t.value for t in ThoughtType]
            raise ToolInputError(
                f"Invalid thought type '{conclusion_thought_type}'. Must be one of: {', '.join(valid_types)}",
                param_name="conclusion_thought_type",
            ) from e

    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Verify Action and Get Workflow ID ---
            cursor = await conn.execute(
                "SELECT workflow_id, status FROM actions WHERE action_id = ?", (action_id,)
            )
            action_row = await cursor.fetchone()
            await cursor.close()
            if not action_row:
                raise ToolInputError(f"Action not found: {action_id}", param_name="action_id")
            workflow_id = action_row["workflow_id"]
            current_status = action_row["status"]
            if current_status not in [ActionStatus.IN_PROGRESS.value, ActionStatus.PLANNED.value]:
                logger.warning(
                    f"Action {action_id} already has terminal status '{current_status}'. Allowing update anyway."
                )

            # --- 2. Update Action Record ---
            tool_result_json = await MemoryUtils.serialize(tool_result)
            await conn.execute(
                """
                UPDATE actions
                SET status = ?,
                    completed_at = ?,
                    tool_result = ?
                WHERE action_id = ?
                """,
                (status_enum.value, now_unix, tool_result_json, action_id),  # *** Use now_unix ***
            )

            # --- 3. Update Workflow Timestamp ---
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),  # *** Use now_unix ***
            )

            # --- 4. Add Conclusion Thought (if provided) ---
            conclusion_thought_id = None
            if conclusion_thought and thought_type_enum:
                cursor = await conn.execute(
                    "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                    (workflow_id,),
                )
                chain_row = await cursor.fetchone()
                await cursor.close()
                if chain_row:
                    thought_chain_id = chain_row["thought_chain_id"]
                    seq_no = await MemoryUtils.get_next_sequence_number(
                        conn, thought_chain_id, "thoughts", "thought_chain_id"
                    )
                    conclusion_thought_id = MemoryUtils.generate_id()
                    await conn.execute(
                        """
                        INSERT INTO thoughts
                            (thought_id, thought_chain_id, thought_type, content, sequence_number, created_at, relevant_action_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conclusion_thought_id,
                            thought_chain_id,
                            thought_type_enum.value,
                            conclusion_thought,
                            seq_no,
                            now_unix,
                            action_id,
                        ),  # *** Use now_unix ***
                    )
                    logger.debug(
                        f"Recorded conclusion thought {conclusion_thought_id} for action {action_id}"
                    )
                else:
                    logger.warning(
                        f"Could not find primary thought chain for workflow {workflow_id} to add conclusion thought."
                    )

            # --- 5. Update Linked Episodic Memory ---
            cursor = await conn.execute(
                "SELECT memory_id, content FROM memories WHERE action_id = ? AND memory_type = ?",
                (action_id, MemoryType.ACTION_LOG.value),
            )
            memory_row = await cursor.fetchone()
            await cursor.close()
            if memory_row:
                memory_id = memory_row["memory_id"]
                original_content = memory_row["content"]
                update_parts = [f"Completed ({status_enum.value})."]
                if summary:
                    update_parts.append(f"Summary: {summary}")
                if tool_result is not None:
                    if isinstance(tool_result, dict):
                        update_parts.append(f"Result: [Dict with {len(tool_result)} keys]")
                    elif isinstance(tool_result, list):
                        update_parts.append(f"Result: [List with {len(tool_result)} items]")
                    elif tool_result:
                        update_parts.append("Result: Success")
                    elif tool_result is False:
                        update_parts.append("Result: Failure")
                    else:
                        update_parts.append("Result obtained.")
                update_text = " ".join(update_parts)
                new_content = original_content + " " + update_text
                importance_mult = 1.0
                if status_enum == ActionStatus.FAILED:
                    importance_mult = 1.2
                elif status_enum == ActionStatus.SKIPPED:
                    importance_mult = 0.8
                await conn.execute(
                    """
                    UPDATE memories
                    SET content = ?,
                        importance = importance * ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (new_content, importance_mult, now_unix, memory_id),
                )
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "update_from_action_completion",
                    memory_id,
                    action_id,
                    {"status": status_enum.value, "summary_added": bool(summary)},
                )
                logger.debug(f"Updated linked memory {memory_id} for completed action {action_id}")
            else:
                logger.warning(
                    f"Could not find corresponding action_log memory for completed action {action_id} to update."
                )

            # --- 6. Commit Transaction ---
            await conn.commit()

            # --- 7. Prepare Result (Format timestamp for output) ---

            result = {
                "action_id": action_id,
                "workflow_id": workflow_id,
                "status": status_enum.value,
                "completed_at": to_iso_z(now_unix),
                "conclusion_thought_id": conclusion_thought_id,
                "success": True,
                "processing_time": time.time() - start_time,
            }

            logger.info(
                f"Completed action {action_id} with status {status_enum.value}",
                emoji_key="white_check_mark",
                duration=result["processing_time"],
            )

            return result

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Error recording action completion for {action_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to record action completion: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_action_details(
    action_id: Optional[str] = None,
    action_ids: Optional[List[str]] = None,
    include_dependencies: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves detailed information about one or more actions.

    Fetch complete details about specific actions by their IDs, either individually
    or in batch. Optionally includes information about action dependencies.

    Args:
        action_id: ID of a single action to retrieve (ignored if action_ids is provided)
        action_ids: Optional list of action IDs to retrieve in batch
        include_dependencies: Whether to include dependency information for each action
        db_path: Path to the SQLite database file

    Returns:
        Dictionary containing action details:
        {
            "actions": [
                {
                    "action_id": "uuid-string",
                    "workflow_id": "workflow-uuid",
                    "action_type": "tool_use",
                    "status": "completed",
                    "title": "Load data",
                    ... other action fields ...
                    "dependencies": { # Only if include_dependencies=True
                        "depends_on": [{"action_id": "action-id-1", "type": "requires"}],
                        "dependent_actions": [{"action_id": "action-id-3", "type": "informs"}]
                    }
                },
                ... more actions if batch ...
            ],
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If neither action_id nor action_ids is provided, or if no matching actions found
        ToolError: If database operation fails
    """
    start_time = time.time()

    # Validate inputs
    if not action_id and not action_ids:
        raise ToolInputError(
            "Either action_id or action_ids must be provided", param_name="action_id"
        )

    # Ensure target_action_ids is a list
    target_action_ids = []
    if action_ids:
        if isinstance(action_ids, list):
            target_action_ids = action_ids

    elif action_id:
        target_action_ids = [action_id]

    if not target_action_ids:  # Should not happen due to initial check, but safeguard
        raise ToolInputError("No valid action IDs specified.", param_name="action_id")

    try:
        async with DBConnection(db_path) as conn:
            placeholders = ", ".join(["?"] * len(target_action_ids))
            # Ensure the query correctly joins tags and groups
            select_query = f"""
                SELECT a.*, GROUP_CONCAT(DISTINCT t.name) as tags_str
                FROM actions a
                LEFT JOIN action_tags at ON a.action_id = at.action_id
                LEFT JOIN tags t ON at.tag_id = t.tag_id
                WHERE a.action_id IN ({placeholders})
                GROUP BY a.action_id
            """

            actions_result = []
            cursor = await conn.execute(select_query, target_action_ids)
            # Iterate using async for
            async for row in cursor:
                # Convert row to dict for easier manipulation
                action_data = dict(row)

                # Format timestamps
                if action_data.get("started_at"):
                    action_data["started_at"] = to_iso_z(action_data["started_at"])
                if action_data.get("completed_at"):
                    action_data["completed_at"] = to_iso_z(action_data["completed_at"])

                # Process tags
                if action_data.get("tags_str"):
                    action_data["tags"] = action_data["tags_str"].split(",")
                else:
                    action_data["tags"] = []
                action_data.pop("tags_str", None)  # Remove the intermediate column

                if action_data.get("tool_args"):
                    action_data["tool_args"] = await MemoryUtils.deserialize(
                        action_data["tool_args"]
                    )
                if action_data.get("tool_result"):
                    action_data["tool_result"] = await MemoryUtils.deserialize(
                        action_data["tool_result"]
                    )

                # Include dependencies if requested
                if include_dependencies:
                    action_data["dependencies"] = {"depends_on": [], "dependent_actions": []}
                    # Fetch actions this one depends ON (target_action_id is the dependency)
                    dep_cursor_on = await conn.execute(
                        "SELECT target_action_id, dependency_type FROM dependencies WHERE source_action_id = ?",
                        (action_data["action_id"],),
                    )
                    depends_on_rows = await dep_cursor_on.fetchall()
                    await dep_cursor_on.close()
                    action_data["dependencies"]["depends_on"] = [
                        {"action_id": r["target_action_id"], "type": r["dependency_type"]}
                        for r in depends_on_rows
                    ]

                    # Fetch actions that depend ON this one (source_action_id depends on this)
                    dep_cursor_by = await conn.execute(
                        "SELECT source_action_id, dependency_type FROM dependencies WHERE target_action_id = ?",
                        (action_data["action_id"],),
                    )
                    dependent_rows = await dep_cursor_by.fetchall()
                    await dep_cursor_by.close()
                    action_data["dependencies"]["dependent_actions"] = [
                        {"action_id": r["source_action_id"], "type": r["dependency_type"]}
                        for r in dependent_rows
                    ]

                actions_result.append(action_data)
            await cursor.close()  # Close the main cursor

            if not actions_result:
                action_ids_str = ", ".join(target_action_ids[:5]) + (
                    "..." if len(target_action_ids) > 5 else ""
                )
                raise ToolInputError(
                    f"No actions found with IDs: {action_ids_str}",
                    param_name="action_id" if action_id else "action_ids",
                )

            processing_time = time.time() - start_time
            logger.info(
                f"Retrieved details for {len(actions_result)} actions",
                emoji_key="search",
                time=processing_time,
            )

            result = {
                "actions": actions_result,
                "success": True,
                "processing_time": processing_time,
            }
            return result

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Error retrieving action details: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve action details: {str(e)}") from e


# ======================================================
# Contextual Summarization
# ======================================================


@with_tool_metrics
@with_error_handling
async def summarize_context_block(
    text_to_summarize: str,
    target_tokens: int = 500,
    context_type: str = "actions",  # "actions", "memories", "thoughts", etc.
    workflow_id: Optional[str] = None,
    provider: str = None,  # Use enum/constant for default
    model: Optional[str] = None,  # Default model
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Summarizes a specific block of context for an agent, optimized for preserving key information.

    A specialized version of summarize_text designed specifically for compressing agent context
    blocks like action histories, memory sets, or thought chains. Uses optimized prompting
    based on context_type to preserve the most relevant information for agent decision-making.

    Args:
        text_to_summarize: Context block text to summarize
        target_tokens: Desired length of summary (default 500)
        context_type: Type of context being summarized (affects prompting)
        workflow_id: Optional workflow ID for logging
        provider: (Optional) LLM provider to use (e.g., 'openai', 'anthropic').
                  Default 'anthropic'.
        model: (Optional) Specific LLM model name (e.g., 'gpt-4.1-mini',
               'claude-3-5-haiku-20241022'). If None, uses provider's default.
               Default 'claude-3-5-haiku-20241022'.
        db_path: Path to the SQLite database file

    Returns:
        Dictionary containing the generated summary:
        {
            "summary": "Concise context summary...",
            "context_type": "actions",
            "compression_ratio": 0.25,  # ratio of summary length to original length
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If text_to_summarize is empty
        ToolError: If summarization fails or provider is invalid
    """
    start_time = time.time()

    if not text_to_summarize:
        raise ToolInputError("Text to summarize cannot be empty", param_name="text_to_summarize")

    # Select appropriate prompt template based on context type
    if context_type == "actions":
        prompt_template = """
You are an expert context summarizer for an AI agent. Your task is to summarize the following ACTION HISTORY logs
while preserving the most important information for the agent to maintain situational awareness.

For actions, focus on:
1. Key actions that changed state or produced important outputs
2. Failed actions and their error reasons
3. The most recent 2-3 actions regardless of importance
4. Any actions that created artifacts or memories
5. Sequential relationships between actions

Produce a VERY CONCISE summary that maintains the chronological flow and preserves action IDs
when referring to specific actions. Aim for approximately {target_tokens} tokens.

ACTION HISTORY TO SUMMARIZE:
{text_to_summarize}

CONCISE ACTION HISTORY SUMMARY:
"""
    elif context_type == "memories":
        prompt_template = """
You are an expert context summarizer for an AI agent. Your task is to summarize the following MEMORY ENTRIES
while preserving the most important information for the agent to maintain understanding.

For memories, focus on:
1. High importance memories (importance > 7)
2. High confidence memories (confidence > 0.8)
3. Insights and facts over observations
4. Memory IDs should be preserved when referring to specific memories
5. Connected memories that form knowledge networks

Produce a VERY CONCISE summary that preserves the key information, high-value insights, and
critical relationships. Aim for approximately {target_tokens} tokens.

MEMORY ENTRIES TO SUMMARIZE:
{text_to_summarize}

CONCISE MEMORY SUMMARY:
"""
    elif context_type == "thoughts":
        prompt_template = """
You are an expert context summarizer for an AI agent. Your task is to summarize the following THOUGHT CHAINS
while preserving the reasoning, decisions, and insights.

For thoughts, focus on:
1. Goals, decisions, and conclusions
2. Key hypotheses and critical reflections
3. The most recent thoughts that may affect current reasoning
4. Thought IDs should be preserved when referring to specific thoughts

Produce a VERY CONCISE summary that captures the agent's reasoning process and main insights.
Aim for approximately {target_tokens} tokens.

THOUGHT CHAINS TO SUMMARIZE:
{text_to_summarize}

CONCISE THOUGHT SUMMARY:
"""
    else:
        # Generic template for other context types
        prompt_template = """
You are an expert context summarizer for an AI agent. Your task is to create a concise summary of the following text
while preserving the most important information for the agent to maintain awareness and functionality.

Focus on information that is:
1. Recent and relevant to current goals
2. Critical for understanding the current state
3. Containing unique identifiers that need to be preserved
4. Representing significant events, insights, or patterns

Produce a VERY CONCISE summary that maximizes the agent's ability to operate with this reduced context.
Aim for approximately {target_tokens} tokens.

TEXT TO SUMMARIZE:
{text_to_summarize}

CONCISE SUMMARY:
"""
    try:
        # Determine provider/model to use
        config = get_config()
        provider_to_use = provider or config.default_provider or LLMGatewayProvider.ANTHROPIC.value # Fallback chain
        provider_instance = await get_provider(provider_to_use)
        if not provider_instance:
            raise ToolError(f"Failed to initialize provider '{provider_to_use}'.")

        # Use passed model, or provider's default, or hardcoded fallback
        model_to_use = model or provider_instance.get_default_model() # Use provider's default method

        # Generate summary
        generation_result = await provider_instance.generate_completion(
            prompt=prompt_template.format(
                text_to_summarize=text_to_summarize, target_tokens=target_tokens
            ),
            model=model_to_use,  # Use the variable holding the desired model
            max_tokens=target_tokens + 50,  # Add some buffer for prompt tokens
            temperature=0.2,  # Lower temperature for more deterministic summaries
        )

        summary_text = generation_result.text.strip()
        if not summary_text:
            raise ToolError("LLM returned empty context summary.")

        # Calculate compression ratio
        # Avoid division by zero if text_to_summarize is empty (although checked earlier)
        original_length = max(1, len(text_to_summarize))
        compression_ratio = len(summary_text) / original_length

        # Log the operation if workflow_id provided
        if workflow_id:
            async with DBConnection(db_path) as conn:
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "compress_context",
                    None,
                    None,
                    {
                        "context_type": context_type,
                        "original_length": len(text_to_summarize),
                        "summary_length": len(summary_text),
                        "compression_ratio": compression_ratio,
                        "provider": provider,  # Log the provider used
                        "model": model_to_use,  # Log the model used
                    },
                )
                await conn.commit()

        processing_time = time.time() - start_time
        logger.info(
            f"Compressed {context_type} context: {len(text_to_summarize)} -> {len(summary_text)} chars (Ratio: {compression_ratio:.2f}, LLM: {provider}/{model_to_use or 'default'})",
            emoji_key="compression",
            time=processing_time,
        )

        return {
            "summary": summary_text,
            "context_type": context_type,
            "compression_ratio": compression_ratio,
            "success": True,
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Error summarizing context block: {e}", exc_info=True)
        raise ToolError(f"Failed to summarize context block: {str(e)}") from e


# ======================================================
# 3.5 Action Dependency Tools
# ======================================================


@with_tool_metrics
@with_error_handling
async def add_action_dependency(
    source_action_id: str,
    target_action_id: str,
    dependency_type: str = "requires",  # e.g., requires, informs, blocks
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Records a dependency between two actions within the same workflow.

    Use this during planning or reflection to explicitly state relationships, like:
    - Action B 'requires' the output of Action A.
    - Action C 'informs' the decision made in Action D.
    - Action E 'blocks' Action F until E is complete.

    Args:
        source_action_id: The ID of the action that depends on the target action.
        target_action_id: The ID of the action that the source action depends upon.
        dependency_type: (Optional) Describes the nature of the dependency (e.g., 'requires', 'informs', 'blocks'). Default 'requires'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary confirming the dependency creation.
        {
            "source_action_id": "source-uuid",
            "target_action_id": "target-uuid",
            "dependency_type": "requires",
            "dependency_id": 123, # Auto-incremented ID
            "created_at": "iso-timestamp",
            "success": true,
            "processing_time": 0.04
        }

    Raises:
        ToolInputError: If IDs are missing, the same, actions not found, or actions belong to different workflows.
        ToolError: If the database operation fails.
    """
    if not source_action_id:
        raise ToolInputError("Source action ID required.", param_name="source_action_id")
    if not target_action_id:
        raise ToolInputError("Target action ID required.", param_name="target_action_id")
    if source_action_id == target_action_id:
        raise ToolInputError(
            "Source and target action IDs cannot be the same.", param_name="source_action_id"
        )
    if not dependency_type:
        raise ToolInputError("Dependency type cannot be empty.", param_name="dependency_type")

    start_time = time.time()
    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # --- Validate Actions & Workflow Consistency ---
            source_workflow_id = None
            target_workflow_id = None
            cursor = await conn.execute(
                "SELECT workflow_id FROM actions WHERE action_id = ?", (source_action_id,)
            )
            source_row = await cursor.fetchone()
            await cursor.close()
            if not source_row:
                raise ToolInputError(
                    f"Source action {source_action_id} not found.", param_name="source_action_id"
                )
            source_workflow_id = source_row["workflow_id"]

            cursor = await conn.execute(
                "SELECT workflow_id FROM actions WHERE action_id = ?", (target_action_id,)
            )
            target_row = await cursor.fetchone()
            await cursor.close()
            if not target_row:
                raise ToolInputError(
                    f"Target action {target_action_id} not found.", param_name="target_action_id"
                )
            target_workflow_id = target_row["workflow_id"]

            if source_workflow_id != target_workflow_id:
                raise ToolInputError(
                    f"Source action ({source_action_id}) and target action ({target_action_id}) belong to different workflows.",
                    param_name="target_action_id",
                )
            workflow_id = source_workflow_id  # Both actions are in this workflow

            # --- Insert Dependency (Ignoring duplicates) ---
            dependency_id = None
            cursor = await conn.execute(  # Use explicit cursor variable
                """
                INSERT OR IGNORE INTO dependencies
                (source_action_id, target_action_id, dependency_type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    source_action_id,
                    target_action_id,
                    dependency_type,
                    now_unix,
                ),
            )
            # Check if a row was actually inserted
            if cursor.rowcount > 0:
                dependency_id = cursor.lastrowid
                logger.debug(f"Inserted new dependency row with ID: {dependency_id}")
            else:
                # If IGNORE occurred, fetch the existing dependency_id
                existing_cursor = await conn.execute(  # Use different cursor variable
                    "SELECT dependency_id FROM dependencies WHERE source_action_id = ? AND target_action_id = ? AND dependency_type = ?",
                    (source_action_id, target_action_id, dependency_type),
                )
                existing_row = await existing_cursor.fetchone()
                await existing_cursor.close()
                if existing_row:
                    dependency_id = existing_row["dependency_id"]
                    logger.debug(
                        f"Dependency already existed. Retrieved existing ID: {dependency_id}"
                    )
                else:
                    logger.warning(
                        f"Dependency insert was ignored, but couldn't retrieve existing row for ({source_action_id}, {target_action_id}, {dependency_type})"
                    )

            await cursor.close()  # Close the main insertion cursor

            # --- Update Workflow Timestamp ---
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            )

            # Log operation (even if ignored, log the attempt)
            log_data = {
                "source_action_id": source_action_id,
                "target_action_id": target_action_id,
                "dependency_type": dependency_type,
                "db_dependency_id": dependency_id,
            }
            await MemoryUtils._log_memory_operation(
                conn, workflow_id, "add_dependency", None, source_action_id, log_data
            )

            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Added dependency ({dependency_type}) from {source_action_id} to {target_action_id}",
                emoji_key="link",
            )

            return {
                "source_action_id": source_action_id,
                "target_action_id": target_action_id,
                "dependency_type": dependency_type,
                "dependency_id": dependency_id,  # May be None if IGNORE failed lookup
                "created_at": to_iso_z(now_unix),
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error adding action dependency: {e}", exc_info=True)
        raise ToolError(f"Failed to add action dependency: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_action_dependencies(
    action_id: str,
    direction: str = "downstream",  # "downstream" (depends on this) or "upstream" (this depends on)
    dependency_type: Optional[str] = None,
    include_details: bool = False,  # Whether to fetch full action details
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves actions that depend on the given action (downstream) or actions the given action depends on (upstream).

    Use this to understand the relationship between actions in a workflow.
    - direction='downstream': Find actions that need this one to complete first ('get_dependent_actions').
    - direction='upstream': Find actions that this one needs to complete first ('get_action_prerequisites').

    Args:
        action_id: The ID of the action to query dependencies for.
        direction: (Optional) 'downstream' (actions depending on this one) or 'upstream' (actions this one depends on). Default 'downstream'.
        dependency_type: (Optional) Filter by the type of dependency (e.g., 'requires').
        include_details: (Optional) If True, returns full details of the related actions, otherwise just their IDs and titles. Default False.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing a list of dependent or prerequisite actions.
        {
            "action_id": "query-action-uuid",
            "direction": "downstream" | "upstream",
            "related_actions": [
                {
                    "action_id": "related-action-uuid",
                    "title": "Related Action Title",
                    "dependency_type": "requires",
                    # ... more details if include_details=True ...
                },
                ...
            ],
            "success": true,
            "processing_time": 0.06
        }

    Raises:
        ToolInputError: If action ID not found or direction is invalid.
        ToolError: If the database operation fails.
    """
    if not action_id:
        raise ToolInputError("Action ID required.", param_name="action_id")
    if direction not in ["downstream", "upstream"]:
        raise ToolInputError(
            "Direction must be 'downstream' or 'upstream'.", param_name="direction"
        )

    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            cursor = await conn.execute("SELECT 1 FROM actions WHERE action_id = ?", (action_id,))
            action_exists = await cursor.fetchone()
            await cursor.close()
            if not action_exists:
                raise ToolInputError(f"Action {action_id} not found.", param_name="action_id")

            select_cols = "a.action_id, a.title, dep.dependency_type"
            if include_details:
                # Fetch timestamps as integers
                select_cols += (
                    ", a.action_type, a.status, a.started_at, a.completed_at, a.sequence_number"
                )

            if direction == "downstream":
                query = f"SELECT {select_cols} FROM dependencies dep JOIN actions a ON dep.source_action_id = a.action_id WHERE dep.target_action_id = ?"
                params = [action_id]
            else:  # upstream
                query = f"SELECT {select_cols} FROM dependencies dep JOIN actions a ON dep.target_action_id = a.action_id WHERE dep.source_action_id = ?"
                params = [action_id]

            if dependency_type:
                query += " AND dep.dependency_type = ?"
                params.append(dependency_type)

            query += " ORDER BY a.sequence_number ASC"

            related_actions = []
            cursor = await conn.execute(query, params)
            async for row in cursor:
                action_data = dict(row)
                if include_details:
                    if action_data.get("started_at"):
                        action_data["started_at"] = to_iso_z(action_data["started_at"])
                    if action_data.get("completed_at"):
                        action_data["completed_at"] = to_iso_z(action_data["completed_at"])
                related_actions.append(action_data)
            await cursor.close()

            processing_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(related_actions)} {direction} dependencies for action {action_id}",
                emoji_key="left_right_arrow",
            )
            return {
                "action_id": action_id,
                "direction": direction,
                "related_actions": related_actions,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting action dependencies: {e}", exc_info=True)
        raise ToolError(f"Failed to get action dependencies: {str(e)}") from e


# --- 4. Artifact Tracking Tools ---

@with_tool_metrics
@with_error_handling
async def record_artifact(
    workflow_id: str,
    name: str,
    artifact_type: str,
    action_id: Optional[str] = None,
    description: Optional[str] = None,
    path: Optional[str] = None,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_output: bool = False,
    tags: Optional[List[str]] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Records information about an artifact created during a workflow
       and creates a corresponding linked episodic memory entry.

    Use this tool to keep track of files, code, data, or other outputs generated during your
    workflow. This creates a persistent record of these artifacts that you can reference later
    and include in reports. It also creates a memory entry about the artifact's creation.

    Args:
        workflow_id: The ID of the workflow this artifact belongs to.
        name: A descriptive name for the artifact.
        artifact_type: Type of artifact. Use a value from ArtifactType enum: 'file', 'text',
                      'image', 'table', 'chart', 'code', 'data', 'json', 'url'.
        action_id: (Optional) The ID of the action that created this artifact.
        description: (Optional) A detailed description of the artifact's purpose or contents.
        path: (Optional) Filesystem path to the artifact if it's a file.
        content: (Optional) The content of the artifact if it's text-based. Can be large.
        metadata: (Optional) Additional structured information about the artifact.
        is_output: (Optional) Whether this is a final output of the workflow. Default False.
        tags: (Optional) List of tags to categorize this artifact.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        A dictionary containing information about the recorded artifact and linked memory.
        {
            "artifact_id": "artifact-uuid",
            "workflow_id": "workflow-uuid",
            "name": "requirements.txt",
            "artifact_type": "file",
            "path": "/path/to/requirements.txt",
            "created_at": "iso-timestamp",
            "is_output": false,
            "tags": ["dependency"],
            "linked_memory_id": "memory-uuid", # ID of the memory entry about this artifact
            "success": true,
            "processing_time": 0.09
        }

    Raises:
        ToolInputError: If required parameters are missing or invalid.
        ToolError: If the database operation fails.
    """
    start_time = time.time()
    # --- Input Validation ---
    if not name:
        raise ToolInputError("Artifact name required", param_name="name")
    try:
        artifact_type_enum = ArtifactType(artifact_type.lower())
    except ValueError as e:
        valid_types = [t.value for t in ArtifactType]
        raise ToolInputError(
            f"Invalid artifact_type '{artifact_type}'. Must be one of: {', '.join(valid_types)}",
            param_name="artifact_type",
        ) from e

    artifact_id = MemoryUtils.generate_id()
    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # --- Existence Checks ---
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()
            if not wf_exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")
            if action_id:
                cursor = await conn.execute(
                    "SELECT 1 FROM actions WHERE action_id = ? AND workflow_id = ?",
                    (action_id, workflow_id),
                )
                action_exists = await cursor.fetchone()
                await cursor.close()
                if not action_exists:
                    raise ToolInputError(
                        f"Action {action_id} not found or does not belong to workflow {workflow_id}",
                        param_name="action_id",
                    )

            # --- Prepare Data ---
            metadata_json = await MemoryUtils.serialize(metadata)
            db_content = None
            # Use max_text_length from config
            max_len = agent_memory_config.max_text_length
            if content:
                content_bytes = content.encode("utf-8")
                if len(content_bytes) > max_len:
                    logger.warning(
                        f"Artifact content for '{name}' exceeds max length ({max_len} bytes). Storing truncated version in DB."
                    )
                    truncated_bytes = content_bytes[:max_len]
                    db_content = truncated_bytes.decode("utf-8", errors="replace")
                    if db_content.endswith("\ufffd") and max_len > 1:
                        db_content_shorter = content_bytes[: max_len - 1].decode(
                            "utf-8", errors="replace"
                        )
                        if not db_content_shorter.endswith("\ufffd"):
                            db_content = db_content_shorter
                    db_content += "..."
                else:
                    db_content = content

            # --- Insert Artifact Record ---
            await conn.execute(
                """
                INSERT INTO artifacts (artifact_id, workflow_id, action_id, artifact_type, name,
                description, path, content, metadata, created_at, is_output)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    workflow_id,
                    action_id,
                    artifact_type_enum.value,
                    name,
                    description,
                    path,
                    db_content,
                    metadata_json,
                    now_unix,
                    is_output,
                ),
            )
            logger.debug(f"Inserted artifact record {artifact_id}")

            # --- Process Tags ---
            artifact_tags = tags or []
            await MemoryUtils.process_tags(conn, artifact_id, artifact_tags, "artifact")
            logger.debug(f"Processed {len(artifact_tags)} tags for artifact {artifact_id}")

            # --- Update Workflow Timestamp ---
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            )

            # --- Create Linked Episodic Memory about the Artifact Creation ---
            memory_id = MemoryUtils.generate_id()
            memory_content = f"Artifact '{name}' (type: {artifact_type_enum.value}) was created"
            if action_id:
                memory_content += f" during action '{action_id[:8]}...'"
            if description:
                memory_content += f". Description: {description[:100]}..."
            if path:
                memory_content += f". Located at: {path}"
            elif db_content and db_content != content:
                memory_content += ". Content stored (truncated)."
            elif content:
                memory_content += ". Content stored directly."
            if is_output:
                memory_content += ". Marked as a final workflow output."
            mem_tags = list(set(["artifact_creation", artifact_type_enum.value] + artifact_tags))
            mem_importance = 6.0 if is_output else 5.0

            await conn.execute(
                """
                 INSERT INTO memories (memory_id, workflow_id, action_id, artifact_id, content, memory_level, memory_type,
                 importance, confidence, tags, created_at, updated_at, access_count)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """,
                (
                    memory_id,
                    workflow_id,
                    action_id,
                    artifact_id,
                    memory_content,
                    MemoryLevel.EPISODIC.value,
                    MemoryType.ARTIFACT_CREATION.value,
                    mem_importance,
                    1.0,
                    json.dumps(mem_tags),
                    now_unix,
                    now_unix,
                    0,
                ),  # Memories already use Unix timestamps
            )
            logger.debug(f"Inserted linked memory record {memory_id} for artifact {artifact_id}")

            # --- Log Operations ---
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create_artifact",
                None,
                action_id,
                {"artifact_id": artifact_id, "name": name, "type": artifact_type_enum.value},
            )
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create_from_artifact",
                memory_id,
                action_id,
                {"artifact_id": artifact_id},
            )

            # --- Commit Transaction ---
            await conn.commit()

            # --- Prepare Result (Format timestamp for output) ---

            result = {
                "artifact_id": artifact_id,
                "workflow_id": workflow_id,
                "name": name,
                "artifact_type": artifact_type_enum.value,
                "path": path,
                "content_stored_in_db": bool(db_content),
                "created_at": to_iso_z(now_unix),
                "is_output": is_output,
                "tags": artifact_tags,
                "linked_memory_id": memory_id,
                "success": True,
                "processing_time": time.time() - start_time,
            }

            logger.info(
                f"Recorded artifact '{name}' ({artifact_id}) and linked memory {memory_id} in workflow {workflow_id}",
                emoji_key="package",
            )

            return result
    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Error recording artifact: {e}", exc_info=True)
        raise ToolError(f"Failed to record artifact: {str(e)}") from e


# --- 5. Thought & Reasoning Tools ---
@with_tool_metrics
@with_error_handling
async def record_thought(
    workflow_id: str,
    content: str,
    thought_type: str = "inference",
    thought_chain_id: Optional[str] = None,
    parent_thought_id: Optional[str] = None,
    relevant_action_id: Optional[str] = None,
    relevant_artifact_id: Optional[str] = None,
    relevant_memory_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Records a thought in a reasoning chain, potentially linking to memory and creating an associated memory entry.

    If an existing database connection (`conn`) is provided, this function will use it
    and operate within the existing transaction context (no internal commit). Otherwise,
    it will acquire a new connection and manage its own transaction.

    Args:
        workflow_id: The ID of the workflow this thought belongs to.
        content: The textual content of the thought.
        thought_type: (Optional) Type of thought (e.g., 'goal', 'plan', 'inference'). Default 'inference'.
        thought_chain_id: (Optional) ID of the chain to add to. If None, adds to the primary chain.
        parent_thought_id: (Optional) ID of the parent thought in the chain.
        relevant_action_id: (Optional) ID of an action this thought relates to.
        relevant_artifact_id: (Optional) ID of an artifact this thought relates to.
        relevant_memory_id: (Optional) ID of a memory this thought relates to.
        db_path: (Optional) Path to the SQLite database file. Used if `conn` is not provided.
        conn: (Optional) An existing aiosqlite database connection to use for database operations.
              If provided, commit/rollback is handled externally.

    Returns:
        Dictionary containing information about the recorded thought.
        Timestamps are returned as ISO 8601 strings.
        {
            "thought_id": "uuid-string",
            "thought_chain_id": "uuid-string",
            "thought_type": "inference",
            "content": "Thought content...",
            "sequence_number": 5,
            "created_at": "iso-timestampZ",
            "linked_memory_id": "uuid-string" | None,
            "success": true
        }

    Raises:
        ToolInputError: If required parameters are missing or invalid, or referenced entities don't exist.
        ToolError: If the database operation fails.
    """
    # --- Input Validation ---
    if not content or not isinstance(content, str):
        raise ToolInputError("Thought content must be a non-empty string", param_name="content")

    try:
        thought_type_enum = ThoughtType(thought_type.lower())
    except ValueError as e:
        valid_types = [t.value for t in ThoughtType]
        raise ToolInputError(
            f"Invalid thought_type '{thought_type}'. Must be one of: {', '.join(valid_types)}",
            param_name="thought_type",
        ) from e

    thought_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    linked_memory_id = None  # Initialize

    # ======================================================
    # == Inner Helper Function for Database Operations ==
    # ======================================================
    async def _perform_db_operations(db_conn: aiosqlite.Connection):
        """Inner function to perform DB ops using the provided connection."""
        nonlocal linked_memory_id  # Allow modification of outer scope variable

        # --- Existence Checks for Foreign Keys ---
        async with db_conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            wf_exists = await cursor.fetchone()
            if not wf_exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

        if parent_thought_id:
            async with db_conn.execute(
                "SELECT 1 FROM thoughts WHERE thought_id = ?", (parent_thought_id,)
            ) as cursor:
                pthought_exists = await cursor.fetchone()
                if not pthought_exists:
                    raise ToolInputError(
                        f"Parent thought not found: {parent_thought_id}",
                        param_name="parent_thought_id",
                    )

        if relevant_action_id:
            async with db_conn.execute(
                "SELECT 1 FROM actions WHERE action_id = ?", (relevant_action_id,)
            ) as cursor:
                raction_exists = await cursor.fetchone()
                if not raction_exists:
                    raise ToolInputError(
                        f"Relevant action not found: {relevant_action_id}",
                        param_name="relevant_action_id",
                    )

        if relevant_artifact_id:
            async with db_conn.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (relevant_artifact_id,)
            ) as cursor:
                rartifact_exists = await cursor.fetchone()
                if not rartifact_exists:
                    raise ToolInputError(
                        f"Relevant artifact not found: {relevant_artifact_id}",
                        param_name="relevant_artifact_id",
                    )

        if relevant_memory_id:
            async with db_conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (relevant_memory_id,)
            ) as cursor:
                rmemory_exists = await cursor.fetchone()
                if not rmemory_exists:
                    raise ToolInputError(
                        f"Relevant memory not found: {relevant_memory_id}",
                        param_name="relevant_memory_id",
                    )

        # --- Determine Target Thought Chain ---
        target_thought_chain_id = thought_chain_id
        if not target_thought_chain_id:
            async with db_conn.execute(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                (workflow_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    if conn:  # Check if running in external transaction
                        raise ToolError(
                            f"Primary thought chain for workflow {workflow_id} not found. Cannot auto-create within existing transaction."
                        )
                    else:
                        # If running standalone, we can create it.
                        target_thought_chain_id = MemoryUtils.generate_id()
                        logger.info(
                            f"No existing thought chain found for workflow {workflow_id}, creating default."
                        )
                        await db_conn.execute(
                            "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, ?, ?)",
                            (target_thought_chain_id, workflow_id, "Main reasoning", now_unix),
                        )
                else:
                    target_thought_chain_id = row["thought_chain_id"]
        else:
            # Verify the provided thought_chain_id exists and belongs to the workflow
            async with db_conn.execute(
                "SELECT 1 FROM thought_chains WHERE thought_chain_id = ? AND workflow_id = ?",
                (target_thought_chain_id, workflow_id),
            ) as cursor:
                chain_exists = await cursor.fetchone()
                if not chain_exists:
                    raise ToolInputError(
                        f"Provided thought chain {target_thought_chain_id} not found or does not belong to workflow {workflow_id}",
                        param_name="thought_chain_id",
                    )

        # --- Get Sequence Number ---
        sequence_number = await MemoryUtils.get_next_sequence_number(
            db_conn, target_thought_chain_id, "thoughts", "thought_chain_id"
        )

        # --- Insert Thought Record ---
        await db_conn.execute(
            """
            INSERT INTO thoughts (
                thought_id, thought_chain_id, parent_thought_id, thought_type, content,
                sequence_number, created_at, relevant_action_id, relevant_artifact_id, relevant_memory_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thought_id,
                target_thought_chain_id,
                parent_thought_id,
                thought_type_enum.value,
                content,
                sequence_number,
                now_unix,
                relevant_action_id,
                relevant_artifact_id,
                relevant_memory_id,
            ),
        )

        # --- Update Workflow Timestamp ---
        await db_conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # --- Create Linked Memory for Important Thoughts ---
        # Define which thought types trigger memory creation
        important_thought_types = [
            ThoughtType.GOAL.value,
            ThoughtType.DECISION.value,
            ThoughtType.SUMMARY.value,
            ThoughtType.REFLECTION.value,
            ThoughtType.HYPOTHESIS.value,
            ThoughtType.INSIGHT.value,
        ]

        if thought_type_enum.value in important_thought_types:
            linked_memory_id = MemoryUtils.generate_id()
            mem_content = (
                f"Thought [{sequence_number}] ({thought_type_enum.value.capitalize()}): {content}"
            )
            mem_tags = ["reasoning", thought_type_enum.value]
            # Give Decision/Goal slightly higher importance
            mem_importance = (
                7.5
                if thought_type_enum.value in [ThoughtType.GOAL.value, ThoughtType.DECISION.value]
                else 6.5
            )

            await db_conn.execute(
                """
                 INSERT INTO memories (
                     memory_id, workflow_id, thought_id, content, memory_level, memory_type,
                     importance, confidence, tags, created_at, updated_at, access_count
                 )
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """,
                (
                    linked_memory_id,
                    workflow_id,
                    thought_id,
                    mem_content,
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.REASONING_STEP.value,  # Semantic level makes sense
                    mem_importance,
                    1.0,
                    json.dumps(mem_tags),
                    now_unix,
                    now_unix,
                    0,
                ),
            )
            await MemoryUtils._log_memory_operation(
                db_conn,
                workflow_id,
                "create_from_thought",
                linked_memory_id,
                None,
                {"thought_id": thought_id},
            )

        # Return values needed for the final result dictionary
        return target_thought_chain_id, sequence_number

    # ======================================================
    # == End Inner Helper Function ==
    # ======================================================

    try:
        target_thought_chain_id_res = None
        sequence_number_res = None

        if conn:
            # Use the provided connection within the existing transaction
            target_thought_chain_id_res, sequence_number_res = await _perform_db_operations(conn)
            # --- NO COMMIT HERE - Handled by the outer transaction manager ---
            logger.debug(
                f"Executed record_thought within provided transaction for chain {target_thought_chain_id_res}"
            )
        else:
            # Manage connection and transaction locally
            db_manager = DBConnection(db_path)
            async with db_manager.transaction() as local_conn:
                target_thought_chain_id_res, sequence_number_res = await _perform_db_operations(
                    local_conn
                )
            # --- COMMIT/ROLLBACK handled by the transaction manager ---
            logger.debug(
                f"Executed record_thought with internal transaction for chain {target_thought_chain_id_res}"
            )

        # --- Prepare Result ---
        result = {
            "thought_id": thought_id,
            "thought_chain_id": target_thought_chain_id_res,
            "thought_type": thought_type_enum.value,
            "content": content,  # Return original content for confirmation
            "sequence_number": sequence_number_res,
            "created_at": to_iso_z(now_unix),  # Convert timestamp for output
            "linked_memory_id": linked_memory_id,  # Will be None if thought wasn't 'important'
            "success": True,
        }

        logger.info(
            f"Recorded thought ({thought_type_enum.value}) in workflow {workflow_id}",
            emoji_key="brain",
        )

        return result

    except ToolInputError:
        # Log ToolInputError specifically for clarity
        logger.warning(
            f"Input error recording thought: {traceback.format_exc(limit=0)}", exc_info=False
        )
        raise  # Re-raise specific input errors
    except ToolError as te:
        # Log ToolError with slightly more context
        logger.error(f"Tool error recording thought: {te}", exc_info=True)
        raise te
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Unexpected error recording thought: {e}", exc_info=True)
        raise ToolError(f"Failed to record thought: {str(e)}") from e


# --- 6. Core Memory Tools ---
@with_tool_metrics
@with_error_handling
async def store_memory(
    workflow_id: str,
    content: str,
    memory_type: str,
    memory_level: str = MemoryLevel.EPISODIC.value,
    importance: float = 5.0,
    confidence: float = 1.0,
    description: Optional[str] = None,
    reasoning: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
    ttl: Optional[int] = None,
    context_data: Optional[Dict[str, Any]] = None,
    generate_embedding: bool = True,  # Flag to control embedding generation
    suggest_links: bool = True,  # Flag to control link suggestion
    link_suggestion_threshold: float = agent_memory_config.similarity_threshold,
    max_suggested_links: int = 3,  # Limit suggestions
    action_id: Optional[str] = None,
    thought_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Stores a new memory entry, generates embeddings, and suggests semantic links.

    Stores a new piece of knowledge or observation, potentially linking it to actions,
    thoughts, or artifacts. Optionally generates a vector embedding and uses it to
    find and suggest links to existing semantically similar memories within the same workflow.

    Args:
        workflow_id: The ID of the workflow this memory belongs to.
        content: The main content of the memory.
        memory_type: The type classification (e.g., 'observation', 'fact', 'insight'). See MemoryType enum.
        memory_level: (Optional) The memory hierarchy level. Default 'episodic'. See MemoryLevel enum.
        importance: (Optional) Importance score (1.0-10.0). Default 5.0.
        confidence: (Optional) Confidence score (0.0-1.0). Default 1.0.
        description: (Optional) A brief description or title for the memory.
        reasoning: (Optional) Explanation of why this memory is relevant or how it was derived.
        source: (Optional) Origin of the memory (e.g., tool name, user input, filename).
        tags: (Optional) List of keywords for categorization. Type and level are added automatically.
        ttl: (Optional) Time-to-live in seconds (0 for permanent, None for level default).
        context_data: (Optional) Additional JSON-serializable context about the memory's creation.
        generate_embedding: (Optional) Whether to generate a vector embedding. Default True.
        suggest_links: (Optional) Whether to find and suggest links to similar memories. Default True.
        link_suggestion_threshold: (Optional) Min similarity score for suggested links. Default SIMILARITY_THRESHOLD.
        max_suggested_links: (Optional) Max number of links to suggest. Default 3.
        action_id: (Optional) ID of an associated action.
        thought_id: (Optional) ID of an associated thought.
        artifact_id: (Optional) ID of an associated artifact.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing details of the stored memory and any suggested links.
        {
            "memory_id": "uuid",
            "workflow_id": "uuid",
            "memory_level": "episodic",
            "memory_type": "observation",
            "content_preview": "Column A seems...",
            "importance": 6.0,
            "confidence": 1.0,
            "created_at_unix": 1678886400,
            "tags": ["observation", "episodic", ...],
            "embedding_id": "uuid" | None,
            "linked_action_id": "uuid" | None,
            "linked_thought_id": "uuid" | None,
            "linked_artifact_id": "uuid" | None,
            "suggested_links": [ # Included if suggest_links=True and matches found
                {
                    "target_memory_id": "uuid",
                    "target_description": "Similar memory desc...",
                    "target_type": "observation",
                    "similarity": 0.85,
                    "suggested_link_type": "related"
                }, ...
            ],
            "success": true,
            "processing_time": 0.25
        }

    Raises:
        ToolInputError: If required parameters are missing or invalid.
        ToolError: If the database operation fails.
    """
    # Parameter validation
    if not content:
        raise ToolInputError("Content cannot be empty.", param_name="content")
    try:
        mem_type = MemoryType(memory_type.lower())
    except ValueError as e:
        valid_types_str = ", ".join([mt.value for mt in MemoryType])
        raise ToolInputError(
            f"Invalid memory_type. Use one of: {valid_types_str}", param_name="memory_type"
        ) from e
    try:
        mem_level = MemoryLevel(memory_level.lower())
    except ValueError as e:
        valid_levels_str = ", ".join([ml.value for ml in MemoryLevel])
        raise ToolInputError(
            f"Invalid memory_level. Use one of: {valid_levels_str}", param_name="memory_level"
        ) from e
    if not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0-10.0.", param_name="importance")
    if not 0.0 <= confidence <= 1.0:
        raise ToolInputError("Confidence must be 0.0-1.0.", param_name="confidence")
    if not 0.0 <= link_suggestion_threshold <= 1.0:
        raise ToolInputError(
            "Link suggestion threshold must be 0.0-1.0.", param_name="link_suggestion_threshold"
        )
    if not isinstance(max_suggested_links, int) or max_suggested_links < 0:
        raise ToolInputError(
            "Max suggested links must be a non-negative integer.", param_name="max_suggested_links"
        )

    memory_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    now_iso = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    # Prepare tags and TTL
    final_tags = list(
        set([str(t).lower() for t in (tags or [])] + [mem_type.value, mem_level.value])
    )  # Also add level as tag
    effective_ttl = ttl if ttl is not None else agent_memory_config.ttl_working

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Existence checks for foreign keys ---
            async with conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Workflow not found: {workflow_id}", param_name="workflow_id"
                    )
            if action_id:
                async with conn.execute(
                    "SELECT 1 FROM actions WHERE action_id = ?", (action_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Action {action_id} not found", param_name="action_id"
                        )
            if thought_id:
                async with conn.execute(
                    "SELECT 1 FROM thoughts WHERE thought_id = ?", (thought_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Thought {thought_id} not found", param_name="thought_id"
                        )
            if artifact_id:
                async with conn.execute(
                    "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Artifact {artifact_id} not found", param_name="artifact_id"
                        )

            # --- 2. Insert the main memory record ---
            await conn.execute(
                """
                INSERT INTO memories (memory_id, workflow_id, content, memory_level, memory_type, importance, confidence,
                description, reasoning, source, context, tags, created_at, updated_at, last_accessed, access_count, ttl,
                action_id, thought_id, artifact_id, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    memory_id,
                    workflow_id,
                    content,
                    mem_level.value,
                    mem_type.value,
                    importance,
                    confidence,
                    description or "",
                    reasoning or "",
                    source or "",
                    await MemoryUtils.serialize(context_data) if context_data else "{}",
                    json.dumps(final_tags),
                    now_unix,
                    now_unix,
                    None,
                    0,
                    effective_ttl,
                    action_id,
                    thought_id,
                    artifact_id,
                ),
            )

            # --- 3. Generate and store embedding (if requested) ---
            embedding_db_id = None
            embedding_generated_successfully = False
            if generate_embedding:
                text_for_embedding = f"{description}: {content}" if description else content
                try:
                    embedding_db_id = await _store_embedding(conn, memory_id, text_for_embedding)
                    if embedding_db_id:
                        embedding_generated_successfully = True
                        logger.debug(
                            f"Successfully generated embedding {embedding_db_id} for memory {memory_id}"
                        )
                    else:
                        logger.warning(
                            f"Embedding generation skipped or failed for memory {memory_id}"
                        )
                except Exception as embed_err:
                    logger.error(
                        f"Error during embedding generation/storage for memory {memory_id}: {embed_err}",
                        exc_info=True,
                    )

            # --- 4. Suggest Semantic Links (if requested and embedding succeeded) ---
            suggested_links_list = []
            if suggest_links and embedding_generated_successfully and max_suggested_links > 0:
                logger.debug(
                    f"Attempting to find similar memories for link suggestion (threshold={link_suggestion_threshold})..."
                )
                try:
                    # Use the text used for embedding generation for the search
                    text_for_search = f"{description}: {content}" if description else content
                    similar_memories = await _find_similar_memories(
                        conn=conn,
                        query_text=text_for_search,
                        workflow_id=workflow_id,  # Limit suggestions to the same workflow
                        limit=max_suggested_links + 1,  # Fetch slightly more to filter self
                        threshold=link_suggestion_threshold,
                        memory_level=None,  # Search across all levels for links initially
                    )

                    if similar_memories:
                        # Fetch details for potential link targets
                        similar_ids = [
                            sim_id for sim_id, _ in similar_memories if sim_id != memory_id
                        ]  # Exclude self
                        if similar_ids:
                            placeholders = ",".join("?" * len(similar_ids))
                            async with conn.execute(
                                f"SELECT memory_id, description, memory_type FROM memories WHERE memory_id IN ({placeholders})",
                                similar_ids,
                            ) as cursor:
                                target_details = {
                                    row["memory_id"]: dict(row) for row in await cursor.fetchall()
                                }

                            # Format suggestions
                            score_map = dict(similar_memories)
                            for sim_id in similar_ids:
                                if sim_id in target_details:
                                    details = target_details[sim_id]
                                    similarity = score_map.get(sim_id, 0.0)
                                    # Basic suggested link type logic (can be expanded)
                                    suggested_type = LinkType.RELATED.value
                                    if mem_type.value == details.get("memory_type"):
                                        suggested_type = (
                                            LinkType.SEQUENTIAL.value
                                            if mem_level.value == MemoryLevel.EPISODIC.value
                                            else LinkType.SUPPORTS.value
                                        )
                                    elif (
                                        mem_type.value == MemoryType.INSIGHT.value
                                        and details.get("memory_type") == MemoryType.FACT.value
                                    ):
                                        suggested_type = LinkType.GENERALIZES.value

                                    suggested_links_list.append(
                                        {
                                            "target_memory_id": sim_id,
                                            "target_description": details.get("description", ""),
                                            "target_type": details.get("memory_type", ""),
                                            "similarity": round(similarity, 4),
                                            "suggested_link_type": suggested_type,
                                        }
                                    )
                                    if len(suggested_links_list) >= max_suggested_links:
                                        break  # Stop if we reached the limit
                        logger.info(
                            f"Generated {len(suggested_links_list)} link suggestions for memory {memory_id}"
                        )

                except Exception as link_err:
                    logger.error(
                        f"Error suggesting links for memory {memory_id}: {link_err}", exc_info=True
                    )

            # --- 5. Update Workflow Timestamp ---
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_iso, now_unix, workflow_id),
            )

            # --- 6. Log Operation ---
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create",
                memory_id,
                action_id,
                {
                    "memory_level": mem_level.value,
                    "memory_type": mem_type.value,
                    "importance": importance,
                    "embedding_generated": embedding_generated_successfully,
                    "links_suggested": len(suggested_links_list),
                    "tags": final_tags,
                },
            )

            # --- 7. Commit Transaction ---
            await conn.commit()

            # --- 8. Prepare Result ---
            processing_time = time.time() - start_time
            result = {
                "memory_id": memory_id,
                "workflow_id": workflow_id,
                "memory_level": mem_level.value,
                "memory_type": mem_type.value,
                "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
                "importance": importance,
                "confidence": confidence,
                "created_at_unix": now_unix,
                "tags": final_tags,
                "embedding_id": embedding_db_id,
                "linked_action_id": action_id,
                "linked_thought_id": thought_id,
                "linked_artifact_id": artifact_id,
                "suggested_links": suggested_links_list,  # Include suggestions
                "success": True,
                "processing_time": processing_time,
            }
            logger.info(
                f"Stored memory {memory_id} ({mem_type.value}) in workflow {workflow_id}. Links suggested: {len(suggested_links_list)}.",
                emoji_key="floppy_disk",
                time=processing_time,
            )
            return result

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Failed to store memory: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Failed to store memory: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_memory_by_id(
    memory_id: str,
    include_links: bool = True,  # Default True for richer context
    include_context: bool = True,  # Default True for semantic context
    context_limit: int = 5,  # Limit for semantic context results
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves a specific memory by its ID, optionally including links and semantic context.

    Fetches the core memory details, updates access time, and can optionally retrieve:
    - Both incoming and outgoing links to other memories.
    - Semantically similar memories based on embedding comparison.

    Args:
        memory_id: ID of the memory to retrieve.
        include_links: (Optional) Whether to include incoming/outgoing links. Default True.
        include_context: (Optional) Whether to include semantically similar memories. Default True.
        context_limit: (Optional) Max number of semantic context memories to return. Default 5.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the memory details, links, and context.
        {
            "memory_id": "uuid-string",
            "workflow_id": "uuid-string",
            "content": "Memory content...",
            # ... other core memory fields ...
            "tags": ["tag1", "tag2"],
            "created_at_unix": 1649712000,
            "updated_at_unix": 1649712000,
            "last_accessed_unix": 1649712000,
            "outgoing_links": [ { ...link_details... } ], # Only if include_links=True
            "incoming_links": [ { ...link_details... } ], # Only if include_links=True
            "semantic_context": [ { ...context_memory_details... } ], # Only if include_context=True
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If the memory ID is not provided or the memory is not found.
        ToolError: If the memory has expired or a database operation fails.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    start_time = time.time()
    result_memory = {}  # Initialize result dictionary

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Fetch Core Memory Data ---
            async with conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

                # Convert row to dict
                result_memory = dict(row)  # aiosqlite.Row is dict-like

            # --- 2. Check TTL ---
            if result_memory.get("ttl", 0) > 0:
                expiry_time = result_memory["created_at"] + result_memory["ttl"]
                if expiry_time <= int(time.time()):
                    await conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                    await conn.commit()
                    logger.warning(f"Attempted to access expired memory {memory_id}.")
                    raise ToolError(f"Memory {memory_id} has expired.")

            # --- 3. Parse JSON Fields ---
            result_memory["tags"] = await MemoryUtils.deserialize(result_memory.get("tags"))
            result_memory["context"] = await MemoryUtils.deserialize(
                result_memory.get("context")
            )  # Original creation context

            # --- 4. Update Access Statistics ---
            await MemoryUtils._update_memory_access(conn, memory_id)
            await MemoryUtils._log_memory_operation(
                conn, result_memory["workflow_id"], "access_by_id", memory_id
            )

            # --- 5. Fetch Links (Incoming & Outgoing) ---
            result_memory["outgoing_links"] = []
            result_memory["incoming_links"] = []
            if include_links:
                # Fetch Outgoing Links (Source = current memory)
                outgoing_query = """
                SELECT ml.link_id, ml.target_memory_id, ml.link_type, ml.strength, ml.description,
                       m.description AS target_description, m.memory_type AS target_type
                FROM memory_links ml
                JOIN memories m ON ml.target_memory_id = m.memory_id
                WHERE ml.source_memory_id = ?
                """
                async with conn.execute(outgoing_query, (memory_id,)) as cursor:
                    async for link_row in cursor:
                        result_memory["outgoing_links"].append(
                            {
                                "link_id": link_row["link_id"],
                                "target_memory_id": link_row["target_memory_id"],
                                "target_description": link_row["target_description"],
                                "target_type": link_row["target_type"],
                                "link_type": link_row["link_type"],
                                "strength": link_row["strength"],
                                "description": link_row["description"],  # Link description
                            }
                        )

                # Fetch Incoming Links (Target = current memory)
                incoming_query = """
                SELECT ml.link_id, ml.source_memory_id, ml.link_type, ml.strength, ml.description,
                       m.description AS source_description, m.memory_type AS source_type
                FROM memory_links ml
                JOIN memories m ON ml.source_memory_id = m.memory_id
                WHERE ml.target_memory_id = ?
                """
                async with conn.execute(incoming_query, (memory_id,)) as cursor:
                    async for link_row in cursor:
                        result_memory["incoming_links"].append(
                            {
                                "link_id": link_row["link_id"],
                                "source_memory_id": link_row["source_memory_id"],
                                "source_description": link_row["source_description"],
                                "source_type": link_row["source_type"],
                                "link_type": link_row["link_type"],
                                "strength": link_row["strength"],
                                "description": link_row["description"],  # Link description
                            }
                        )

            # --- 6. Fetch Semantic Context ---
            result_memory["semantic_context"] = []
            if include_context and result_memory.get("embedding_id"):
                # Formulate text for similarity search (use description + content)
                search_text = result_memory.get("content", "")
                if result_memory.get("description"):
                    search_text = f"{result_memory['description']}: {search_text}"

                if search_text:
                    try:
                        # Find similar memories (excluding self)
                        similar_results = await _find_similar_memories(
                            conn=conn,
                            query_text=search_text,
                            workflow_id=result_memory["workflow_id"],  # Search within same workflow
                            limit=context_limit + 1,  # Fetch one extra in case self is included
                            threshold=agent_memory_config.similarity_threshold
                            * 0.9,  # Slightly lower threshold for context
                        )

                        if similar_results:
                            # Get IDs, excluding the current memory ID
                            similar_ids = [
                                mem_id for mem_id, score in similar_results if mem_id != memory_id
                            ][:context_limit]
                            score_map = dict(similar_results)  # Keep scores

                            if similar_ids:
                                # Fetch details for context memories
                                placeholders = ", ".join(["?"] * len(similar_ids))
                                context_query = "SELECT memory_id, description, memory_type, importance FROM memories WHERE memory_id IN ({})".format(
                                    placeholders
                                )
                                async with conn.execute(
                                    context_query, similar_ids
                                ) as context_cursor:
                                    context_rows = await context_cursor.fetchall()
                                    # Order by original similarity score
                                    ordered_context = sorted(
                                        context_rows,
                                        key=lambda r: score_map.get(r["memory_id"], -1.0),
                                        reverse=True,
                                    )
                                    for context_row in ordered_context:
                                        result_memory["semantic_context"].append(
                                            {
                                                "memory_id": context_row["memory_id"],
                                                "description": context_row["description"],
                                                "memory_type": context_row["memory_type"],
                                                "importance": context_row["importance"],
                                                "similarity": score_map.get(
                                                    context_row["memory_id"], 0.0
                                                ),
                                            }
                                        )
                    except Exception as context_err:
                        logger.warning(
                            f"Could not retrieve semantic context for memory {memory_id}: {context_err}"
                        )
                        # Continue without semantic context if it fails

            # --- 7. Finalize and Return ---
            await conn.commit()  # Commit the access updates

            result_memory["success"] = True
            # Add consistently named timestamp keys
            result_memory["created_at_unix"] = result_memory["created_at"]
            result_memory["updated_at_unix"] = result_memory["updated_at"]
            result_memory["last_accessed_unix"] = result_memory[
                "last_accessed"
            ]  # Already updated by _update_memory_access

            result_memory["processing_time"] = time.time() - start_time

            logger.info(
                f"Retrieved memory {memory_id} with links={include_links}, context={include_context}",
                emoji_key="inbox_tray",
            )
            return result_memory  # Return the enhanced dictionary

    except (ToolInputError, ToolError):
        raise  # Re-raise specific handled errors
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id}: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Failed to get memory {memory_id}: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def search_semantic_memories(
    query: str,
    workflow_id: Optional[str] = None,  # Allow searching across workflows if None
    limit: int = 5,
    threshold: float = agent_memory_config.similarity_threshold,  # Use constant
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,  # Filter by type is now handled by _find_similar_memories
    include_content: bool = True,  # Control whether full content is returned
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Searches memories based on semantic similarity using EmbeddingService.

    Retrieves memories whose embeddings are semantically close to the query text's embedding.
    Optionally filters by workflow, memory level, and type. Updates access stats for retrieved memories.
    """
    # Input validation
    if not query:
        raise ToolInputError("Search query required.", param_name="query")
    if not isinstance(limit, int) or limit < 1:
        raise ToolInputError("Limit must be positive integer.", param_name="limit")
    if not 0.0 <= threshold <= 1.0:
        raise ToolInputError("Threshold must be 0.0-1.0.", param_name="threshold")
    if memory_level:
        try:
            MemoryLevel(memory_level.lower())  # Validate enum value
        except ValueError as e:
            raise ToolInputError("Invalid memory_level.", param_name="memory_level") from e
    if memory_type:
        try:
            MemoryType(memory_type.lower())  # Validate enum value
        except ValueError as e:
            raise ToolInputError("Invalid memory_type.", param_name="memory_type") from e

    start_time = time.time()
    try:
        async with DBConnection(db_path) as conn:
            # Verify workflow exists only if a specific one is provided
            if workflow_id:
                async with conn.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Workflow {workflow_id} not found.", param_name="workflow_id"
                        )

            # --- Step 1: Find similar memory IDs and scores ---
            # Pass the memory_type filter to the helper function now
            similar_results: List[Tuple[str, float]] = await _find_similar_memories(
                conn=conn,
                query_text=query,
                workflow_id=workflow_id,
                limit=limit,
                threshold=threshold,
                memory_level=memory_level,
                memory_type=memory_type,  # Pass the filter here
            )

            # If no similar memories found, return early
            if not similar_results:
                logger.info(
                    f"Semantic search for '{query[:50]}...' found no results matching filters above threshold {threshold}.",
                    emoji_key="zzz",
                )
                return {
                    "memories": [],
                    "query": query,
                    "workflow_id": workflow_id,
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # --- Step 2: Fetch full details for the matching memories ---
            memory_ids = [mem_id for mem_id, score in similar_results]
            placeholders = ", ".join(["?"] * len(memory_ids))

            # Define columns to select based on include_content flag
            select_cols = "memory_id, workflow_id, description, memory_type, memory_level, importance, confidence, created_at, tags, action_id, thought_id, artifact_id"
            if include_content:
                select_cols += ", content"

            # Create a score mapping for ordering results correctly later
            score_map = dict(similar_results)

            # Fetch memory data from the database
            memories_data = []
            async with conn.execute(
                f"SELECT {select_cols} FROM memories WHERE memory_id IN ({placeholders})",
                memory_ids,
            ) as cursor:
                rows = await cursor.fetchall()

                # --- Step 3: Process and format results, update access stats ---
                # Order rows based on the similarity score (desc) and then memory_id (asc) for stability
                ordered_rows = sorted(
                    rows,
                    key=lambda r: (score_map.get(r["memory_id"], -1.0), r["memory_id"]),
                    reverse=True,  # Sort primarily by score descending
                )
                # Note: Secondary sort by memory_id will be ascending due to how tuple sorting works with reverse=True

                for row in ordered_rows:
                    # Type filter already applied in _find_similar_memories, no need to check again here

                    mem_dict = dict(row)  # Convert row to dict
                    mem_dict["similarity"] = score_map.get(
                        row["memory_id"], 0.0
                    )  # Add similarity score
                    mem_dict["created_at_unix"] = row["created_at"]  # Keep unix ts name consistent
                    mem_dict["tags"] = await MemoryUtils.deserialize(
                        mem_dict.get("tags")
                    )  # Deserialize tags

                    # Update access time and log operation for this retrieved memory
                    await MemoryUtils._update_memory_access(conn, row["memory_id"])
                    await MemoryUtils._log_memory_operation(
                        conn,
                        row["workflow_id"],
                        "semantic_access",
                        row["memory_id"],
                        None,
                        {"query": query[:100], "score": mem_dict["similarity"]},
                    )

                    memories_data.append(mem_dict)

                    # Limit should have been handled by _find_similar_memories, but double-check
                    if len(memories_data) >= limit:
                        break

            # Commit the access updates
            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Semantic search found {len(memories_data)} results for query: '{query[:50]}...'",
                emoji_key="mag",
                time=processing_time,
            )

            # Return the formatted results
            return {
                "memories": memories_data,  # Return list of full memory dictionaries
                "query": query,
                "workflow_id": workflow_id,  # Indicate which workflow was searched, or None
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise  # Re-raise specific input errors
    except Exception as e:
        logger.error(f"Failed semantic search: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Failed semantic search: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def hybrid_search_memories(
    query: str,
    workflow_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    semantic_weight: float = 0.6,  # Default weight for semantic score
    keyword_weight: float = 0.4,  # Default weight for keyword/relevance score
    # Filters from query_memories
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_importance: Optional[float] = None,
    max_importance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_created_at_unix: Optional[int] = None,
    max_created_at_unix: Optional[int] = None,
    # Control flags
    include_content: bool = True,
    include_links: bool = False,  # Keep False by default for performance in search
    link_direction: str = "outgoing",  # 'outgoing', 'incoming', 'both' - Determines which links to fetch if include_links=True
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Performs a hybrid search combining semantic similarity and keyword/filtered relevance.

    Retrieves memories ranked by a weighted combination of their semantic similarity
    to the query and their relevance based on keywords (FTS) and other attributes
    (importance, recency, confidence).

    Args:
        query: The search query text (used for both semantic and keyword/FTS search).
        workflow_id: (Optional) ID of the workflow to search within. If None, searches globally.
        limit: (Optional) Maximum number of results to return. Default 10.
        offset: (Optional) Number of results to skip for pagination. Default 0.
        semantic_weight: (Optional) Weight (0.0-1.0) for the semantic similarity score. Default 0.6.
        keyword_weight: (Optional) Weight (0.0-1.0) for the keyword/attribute relevance score. Default 0.4.
        memory_level: (Optional) Filter by memory level.
        memory_type: (Optional) Filter by memory type.
        tags: (Optional) Filter memories containing ALL specified tags.
        min_importance: (Optional) Minimum importance score.
        max_importance: (Optional) Maximum importance score.
        min_confidence: (Optional) Minimum confidence score.
        min_created_at_unix: (Optional) Minimum creation timestamp (Unix seconds).
        max_created_at_unix: (Optional) Maximum creation timestamp (Unix seconds).
        include_content: (Optional) Whether to include full memory content. Default True.
        include_links: (Optional) Whether to include detailed link info. Default False.
        link_direction: (Optional) Direction for links if included ('outgoing', 'incoming', 'both'). Default 'outgoing'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the ranked list of matching memories and scores.
        {
            "memories": [
                {
                    ...memory_details...,
                    "hybrid_score": 0.85,
                    "semantic_score": 0.92,
                    "keyword_relevance_score": 0.75, # Normalized 0-1
                    "links": { # Populated if include_links=True
                        "outgoing": [ ... ],
                        "incoming": [ ... ]
                    }
                },
                ...
            ],
            "total_candidates_considered": 55, # Total unique memories found by either search before ranking/limit
            "success": true,
            "processing_time": 0.45
        }

    Raises:
        ToolInputError: If parameters are invalid (weights out of range, bad filters, etc.).
        ToolError: If the database operation or semantic search fails.
    """
    start_time = time.time()

    # --- Input Validation ---
    if not query:
        raise ToolInputError("Query string cannot be empty.", param_name="query")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ToolInputError(
            "semantic_weight must be between 0.0 and 1.0", param_name="semantic_weight"
        )
    if not 0.0 <= keyword_weight <= 1.0:
        raise ToolInputError(
            "keyword_weight must be between 0.0 and 1.0", param_name="keyword_weight"
        )
    if semantic_weight + keyword_weight <= 0:
        raise ToolInputError(
            "Sum of semantic_weight and keyword_weight must be positive.",
            param_name="semantic_weight",
        )
    if limit < 1:
        raise ToolInputError("Limit must be >= 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("Offset must be >= 0", param_name="offset")
    # Reuse validation for filters from query_memories if needed, or perform here
    if memory_level:
        try:
            MemoryLevel(memory_level.lower())
        except ValueError as e:
            raise ToolInputError("Invalid memory_level.", param_name="memory_level") from e
    if memory_type:
        try:
            MemoryType(memory_type.lower())
        except ValueError as e:
            raise ToolInputError("Invalid memory_type.", param_name="memory_type") from e
    valid_link_directions = ["outgoing", "incoming", "both"]
    link_direction_lower = link_direction.lower()  # Use lower case consistently
    if link_direction_lower not in valid_link_directions:
        raise ToolInputError(
            f"link_direction must be one of: {', '.join(valid_link_directions)}",
            param_name="link_direction",
        )

    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    norm_sem_weight = semantic_weight / total_weight
    norm_key_weight = keyword_weight / total_weight

    # Dictionary to hold scores for each memory ID
    # Structure: { memory_id: {"semantic": score, "keyword": score, "hybrid": score} }
    combined_scores: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"semantic": 0.0, "keyword": 0.0, "hybrid": 0.0}
    )

    try:
        async with DBConnection(db_path) as conn:
            # --- Step 1: Semantic Search ---
            semantic_results: List[Tuple[str, float]] = []
            if norm_sem_weight > 0:
                try:
                    # Use internal helper _find_similar_memories
                    # Fetch more candidates than final limit initially to allow keyword scores to influence ranking
                    semantic_candidate_limit = min(max(limit * 5, 50), agent_memory_config.max_semantic_candidates)
                    semantic_results = await _find_similar_memories(
                        conn=conn,
                        query_text=query,
                        workflow_id=workflow_id,
                        limit=semantic_candidate_limit,  # Fetch more initially
                        threshold=0.1,  # Lower threshold initially to get more candidates
                        memory_level=memory_level,
                        memory_type=memory_type,
                    )
                    for mem_id, score in semantic_results:
                        combined_scores[mem_id]["semantic"] = score
                    logger.debug(
                        f"Hybrid search: Found {len(semantic_results)} semantic candidates."
                    )
                except Exception as sem_err:
                    logger.warning(
                        f"Semantic search part failed in hybrid search: {sem_err}", exc_info=True
                    )
                    # Allow hybrid search to continue with keyword results if semantic fails

            # --- Step 2: Keyword/Filtered Search & Relevance Score ---
            if norm_key_weight > 0:
                # Build query similar to query_memories, but only fetch IDs and scoring components
                select_cols = "m.memory_id, m.importance, m.confidence, m.created_at, m.access_count, m.last_accessed"
                data_query = f"SELECT {select_cols} FROM memories m"
                where_clauses = ["1=1"]
                params: List[Any] = []
                fts_params: List[Any] = []
                joins = ""  # Keep track of needed joins

                # Apply filters (same logic as query_memories)
                if workflow_id:
                    where_clauses.append("m.workflow_id = ?")
                    params.append(workflow_id)
                if memory_level:
                    where_clauses.append("m.memory_level = ?")
                    params.append(memory_level.lower())
                if memory_type:
                    where_clauses.append("m.memory_type = ?")
                    params.append(memory_type.lower())
                if min_importance is not None:
                    where_clauses.append("m.importance >= ?")
                    params.append(min_importance)
                if max_importance is not None:
                    where_clauses.append("m.importance <= ?")
                    params.append(max_importance)
                if min_confidence is not None:
                    where_clauses.append("m.confidence >= ?")
                    params.append(min_confidence)
                if min_created_at_unix is not None:
                    where_clauses.append("m.created_at >= ?")
                    params.append(min_created_at_unix)
                if max_created_at_unix is not None:
                    where_clauses.append("m.created_at <= ?")
                    params.append(max_created_at_unix)
                now_unix = int(time.time())
                where_clauses.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
                params.append(now_unix)
                if tags and isinstance(tags, list) and len(tags) > 0:
                    tags_json = json.dumps([str(tag).lower() for tag in tags])
                    where_clauses.append("json_contains_all(m.tags, ?)")
                    params.append(tags_json)
                if query:  # Use the main query for FTS search
                    # Important: FTS requires JOINing the FTS table
                    if "memory_fts" not in joins:
                        joins += " JOIN memory_fts fts ON m.rowid = fts.rowid"
                    where_clauses.append("fts.memory_fts MATCH ?")
                    fts_query_term = " OR ".join(query.strip().split())  # Basic query formation
                    fts_params.append(fts_query_term)

                # Combine WHERE clauses and add joins
                where_sql = " WHERE " + " AND ".join(where_clauses)
                final_query = data_query + joins + where_sql

                # Fetch *all* matching results for keyword scoring, don't limit yet
                # We limit *after* hybrid scoring
                keyword_candidates = []
                cursor = await conn.execute(final_query, params + fts_params)
                keyword_candidates = await cursor.fetchall()
                await cursor.close()  # Close cursor

                # Calculate raw keyword relevance scores (0-10 range)
                raw_keyword_scores = {}
                for row in keyword_candidates:
                    mem_id = row["memory_id"]
                    kw_relevance = _compute_memory_relevance(
                        row["importance"],
                        row["confidence"],
                        row["created_at"],
                        row["access_count"],
                        row["last_accessed"],
                    )
                    # Store the raw score (0-10 range)
                    raw_keyword_scores[mem_id] = kw_relevance
                    # Initialize the keyword score in combined_scores (will be normalized later)
                    if mem_id not in combined_scores:
                        combined_scores[
                            mem_id
                        ]  # Ensure entry exists if only found by keyword search
                    combined_scores[mem_id]["keyword"] = 0.0  # Placeholder

                # Find the maximum observed raw keyword score
                max_kw_score = 0.0
                if raw_keyword_scores:
                    max_kw_score = max(raw_keyword_scores.values())

                # Normalize keyword scores based on the observed maximum (or 10 if max is 0)
                # Avoid division by zero or near-zero scores causing massive inflation.
                normalization_factor = max(
                    max_kw_score, 1e-6
                )  # Use a small epsilon if max_kw_score is 0

                for mem_id, raw_score in raw_keyword_scores.items():
                    normalized_kw_score = min(max(raw_score / normalization_factor, 0.0), 1.0)
                    combined_scores[mem_id]["keyword"] = normalized_kw_score

                logger.debug(
                    f"Hybrid search: Found and scored {len(keyword_candidates)} keyword/filtered candidates (Max raw score: {max_kw_score:.2f})."
                )

            # --- Step 3: Calculate Hybrid Score ---
            final_ranked_ids = []
            final_scores_map = {}
            if not combined_scores:
                logger.info(
                    "Hybrid search yielded no candidates from either semantic or keyword search."
                )
            else:
                for _mem_id, scores in combined_scores.items():
                    scores["hybrid"] = (scores["semantic"] * norm_sem_weight) + (
                        scores["keyword"] * norm_key_weight
                    )

                # Sort by hybrid score
                sorted_ids_scores = sorted(
                    combined_scores.items(), key=lambda item: item[1]["hybrid"], reverse=True
                )

                # Apply pagination *after* ranking
                paginated_ids_scores = sorted_ids_scores[offset : offset + limit]
                final_ranked_ids = [item[0] for item in paginated_ids_scores]  # Get just the IDs
                final_scores_map = {
                    item[0]: item[1] for item in paginated_ids_scores
                }  # Keep scores for final list

            # --- Step 4: Fetch Full Details for Ranked & Paginated IDs ---
            memories_results = []  # Final list of processed memories
            total_candidates_considered = len(combined_scores)  # Total unique matches found
            rows_map = {}  # To store fetched memory data

            if final_ranked_ids:
                placeholders = ",".join("?" * len(final_ranked_ids))
                # Select columns based on include_content
                select_cols_final = "m.memory_id, m.workflow_id, m.memory_level, m.memory_type, m.importance, m.confidence, m.description, m.reasoning, m.source, m.tags, m.created_at, m.updated_at, m.last_accessed, m.access_count, m.ttl, m.action_id, m.thought_id, m.artifact_id"
                if include_content:
                    select_cols_final += ", m.content"

                # Fetch data
                query_final = f"SELECT {select_cols_final} FROM memories m WHERE m.memory_id IN ({placeholders})"
                cursor = await conn.execute(query_final, final_ranked_ids)
                # Store fetched data in a map for easy access
                rows_map = {row["memory_id"]: dict(row) for row in await cursor.fetchall()}
                await cursor.close()

            # --- Step 5: Batch Fetch Links if Requested ---
            links_map = defaultdict(lambda: {"outgoing": [], "incoming": []})
            if include_links and final_ranked_ids:
                placeholders = ",".join("?" * len(final_ranked_ids))

                # Fetch Outgoing Links
                if link_direction_lower in ["outgoing", "both"]:
                    outgoing_query = f"""
                    SELECT ml.link_id, ml.source_memory_id, ml.target_memory_id, ml.link_type, ml.strength, ml.description AS link_description,
                           target_mem.description AS target_description, target_mem.memory_type AS target_type
                    FROM memory_links ml JOIN memories target_mem ON ml.target_memory_id = target_mem.memory_id
                    WHERE ml.source_memory_id IN ({placeholders})
                    """
                    outgoing_cursor = await conn.execute(outgoing_query, final_ranked_ids)
                    async for link_row in outgoing_cursor:
                        links_map[link_row["source_memory_id"]]["outgoing"].append(dict(link_row))
                    await outgoing_cursor.close()

                # Fetch Incoming Links
                if link_direction_lower in ["incoming", "both"]:
                    incoming_query = f"""
                    SELECT ml.link_id, ml.source_memory_id, ml.target_memory_id, ml.link_type, ml.strength, ml.description AS link_description,
                           source_mem.description AS source_description, source_mem.memory_type AS source_type
                    FROM memory_links ml JOIN memories source_mem ON ml.source_memory_id = source_mem.memory_id
                    WHERE ml.target_memory_id IN ({placeholders})
                    """
                    incoming_cursor = await conn.execute(incoming_query, final_ranked_ids)
                    async for link_row in incoming_cursor:
                        links_map[link_row["target_memory_id"]]["incoming"].append(dict(link_row))
                    await incoming_cursor.close()

            # --- Step 6: Reconstruct Results, Attach Links, Update Access Stats ---
            update_access_tasks = []  # Tasks for updating access stats
            if final_ranked_ids:
                # Reconstruct the list in the final ranked order and add scores/links
                for mem_id in final_ranked_ids:
                    if mem_id in rows_map:
                        memory_dict = rows_map[mem_id]
                        scores = final_scores_map.get(mem_id, {})  # Get scores for this ID
                        memory_dict["hybrid_score"] = round(scores.get("hybrid", 0.0), 4)
                        memory_dict["semantic_score"] = round(scores.get("semantic", 0.0), 4)
                        memory_dict["keyword_relevance_score"] = round(
                            scores.get("keyword", 0.0), 4
                        )  # Already normalized 0-1

                        # Add other standard fields
                        memory_dict["tags"] = await MemoryUtils.deserialize(memory_dict.get("tags"))
                        memory_dict["created_at_unix"] = memory_dict.get("created_at")
                        memory_dict["updated_at_unix"] = memory_dict.get("updated_at")
                        memory_dict["last_accessed_unix"] = memory_dict.get("last_accessed")

                        # Attach links from the pre-fetched map
                        if include_links:
                            memory_dict["links"] = links_map[mem_id]

                        memories_results.append(memory_dict)

                        # Prepare access update tasks
                        update_access_tasks.append(MemoryUtils._update_memory_access(conn, mem_id))
                        update_access_tasks.append(
                            MemoryUtils._log_memory_operation(
                                conn,
                                memory_dict["workflow_id"],
                                "hybrid_access",
                                mem_id,
                                None,
                                {"query": query[:100], "hybrid_score": memory_dict["hybrid_score"]},
                            )
                        )

            # --- Step 7: Update Access Stats Concurrently & Commit ---
            if update_access_tasks:
                await asyncio.gather(*update_access_tasks)
                await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Hybrid search returned {len(memories_results)} results for query '{query[:50]}...'",
                emoji_key="magic_wand",
                time=processing_time,
            )

            return {
                "memories": memories_results,
                "total_candidates_considered": total_candidates_considered,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Hybrid search failed: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def create_memory_link(
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,  # Use LinkType enum
    strength: float = 1.0,
    description: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Creates an associative link between two memories.
    (Adapted from cognitive_memory.create_memory_link).
    """
    if source_memory_id == target_memory_id:
        raise ToolInputError("Cannot link memory to itself.", param_name="source_memory_id")
    try:
        link_type_enum = LinkType(link_type.lower())
    except ValueError as e:
        raise ToolInputError(
            f"Invalid link_type. Use one of: {[lt.value for lt in LinkType]}",
            param_name="link_type",
        ) from e
    if not 0.0 <= strength <= 1.0:
        raise ToolInputError("Strength must be 0.0-1.0.", param_name="strength")

    link_id = MemoryUtils.generate_id()
    now_unix = int(time.time())

    try:
        async with DBConnection(db_path) as conn:
            # Check memories exist and get workflow_id (use source memory's workflow)
            async with conn.execute(
                "SELECT workflow_id FROM memories WHERE memory_id = ?", (source_memory_id,)
            ) as cursor:
                source_row = await cursor.fetchone()
                if not source_row:
                    raise ToolInputError(
                        f"Source memory {source_memory_id} not found.",
                        param_name="source_memory_id",
                    )
                workflow_id = source_row["workflow_id"]
            async with conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (target_memory_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Target memory {target_memory_id} not found.",
                        param_name="target_memory_id",
                    )

            # Insert or Replace link (handle existing links gracefully)
            # Using INSERT OR REPLACE requires unique constraint on (source, target, type)
            await conn.execute(
                """
                INSERT OR REPLACE INTO memory_links
                (link_id, source_memory_id, target_memory_id, link_type, strength, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    source_memory_id,
                    target_memory_id,
                    link_type_enum.value,
                    strength,
                    description or "",
                    now_unix,
                ),
            )

            # Log operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "link",
                source_memory_id,
                None,
                {
                    "target_memory_id": target_memory_id,
                    "link_type": link_type_enum.value,
                    "link_id": link_id,
                },
            )

            await conn.commit()

            result = {
                "link_id": link_id,
                "source_memory_id": source_memory_id,
                "target_memory_id": target_memory_id,
                "link_type": link_type_enum.value,
                "strength": strength,
                "created_at_unix": now_unix,
                "success": True,
            }
            logger.info(
                f"Created link {link_id} from {source_memory_id} to {target_memory_id}",
                emoji_key="link",
            )
            return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Failed to create memory link: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Failed to create memory link: {str(e)}") from e


# --- 7. Core Memory Retrieval ---
@with_tool_metrics
@with_error_handling
async def query_memories(
    workflow_id: Optional[str] = None,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    search_text: Optional[str] = None,  # Keyword/FTS search
    tags: Optional[List[str]] = None,  # Filter by tags
    min_importance: Optional[float] = None,
    max_importance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_created_at_unix: Optional[int] = None,
    max_created_at_unix: Optional[int] = None,
    sort_by: str = "relevance",  # relevance, importance, created_at, updated_at, confidence, last_accessed
    sort_order: str = "DESC",
    include_content: bool = True,
    include_links: bool = False,  # Flag to include detailed links
    link_direction: str = "outgoing",  # 'outgoing', 'incoming', 'both' - Determines which links to fetch if include_links=True
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves memories based on various criteria like level, type, tags, text, importance, etc.

    This is the primary tool for filtering and retrieving memories from the knowledge base
    using structured criteria and keyword search (distinct from pure semantic search).
    Includes option to fetch detailed information about linked memories.

    Args:
        workflow_id: (Optional) ID of the workflow to query. If None, searches across all accessible workflows.
        memory_level: (Optional) Filter by memory level (e.g., 'episodic', 'semantic').
        memory_type: (Optional) Filter by memory type (e.g., 'insight', 'fact').
        search_text: (Optional) Full-text search query for content, description, reasoning, tags.
        tags: (Optional) Filter memories containing ALL specified tags.
        min_importance: (Optional) Minimum importance score (1.0-10.0).
        max_importance: (Optional) Maximum importance score (1.0-10.0).
        min_confidence: (Optional) Minimum confidence score (0.0-1.0).
        min_created_at_unix: (Optional) Minimum creation timestamp (Unix seconds).
        max_created_at_unix: (Optional) Maximum creation timestamp (Unix seconds).
        sort_by: (Optional) Field to sort by. Options: 'relevance', 'importance', 'created_at',
                 'updated_at', 'confidence', 'last_accessed', 'access_count'. Default 'relevance'.
        sort_order: (Optional) Sort direction ('ASC' or 'DESC'). Default 'DESC'.
        include_content: (Optional) Whether to include the full memory content. Default True.
        include_links: (Optional) Whether to include detailed info about linked memories. Default False.
        link_direction: (Optional) Which links to fetch if include_links is True:
                        'outgoing', 'incoming', or 'both'. Default 'outgoing'.
        limit: (Optional) Maximum number of memories to return. Default 10.
        offset: (Optional) Number of memories to skip for pagination. Default 0.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the list of matching memories and total count.
        {
            "memories": [
                {
                    ...memory_details...,
                    "links": {
                        "outgoing": [ { "link_id": ..., "target_memory_id": ..., "target_description": ..., "link_type": ..., "strength": ... }, ... ],
                        "incoming": [ { "link_id": ..., "source_memory_id": ..., "source_description": ..., "link_type": ..., "strength": ... }, ... ]
                    } # Populated if include_links=True
                },
                ...
            ],
            "total_matching_count": 42,
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If filter parameters are invalid.
        ToolError: If the database operation fails.
    """
    start_time = time.time()

    # --- Input Validation ---
    valid_sort_fields = [
        "relevance",
        "importance",
        "created_at",
        "updated_at",
        "confidence",
        "last_accessed",
        "access_count",
    ]
    if sort_by not in valid_sort_fields:
        raise ToolInputError(
            f"sort_by must be one of: {', '.join(valid_sort_fields)}", param_name="sort_by"
        )
    if sort_order.upper() not in ["ASC", "DESC"]:
        raise ToolInputError("sort_order must be 'ASC' or 'DESC'", param_name="sort_order")
    sort_order = sort_order.upper()

    valid_link_directions = ["outgoing", "incoming", "both"]
    if link_direction.lower() not in valid_link_directions:
        raise ToolInputError(
            f"link_direction must be one of: {', '.join(valid_link_directions)}",
            param_name="link_direction",
        )
    link_direction = link_direction.lower()

    if limit < 1:
        raise ToolInputError("Limit must be >= 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("Offset must be >= 0", param_name="offset")

    if memory_level:
        try:
            MemoryLevel(memory_level.lower())
        except ValueError as e:
            raise ToolInputError("Invalid memory_level.", param_name="memory_level") from e
    if memory_type:
        try:
            MemoryType(memory_type.lower())
        except ValueError as e:
            raise ToolInputError("Invalid memory_type.", param_name="memory_type") from e

    try:
        async with DBConnection(db_path) as conn:
            # Verify workflow exists if provided
            if workflow_id:
                async with conn.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Workflow {workflow_id} not found.", param_name="workflow_id"
                        )

            # --- Build Query ---
            select_clause = "m.memory_id, m.workflow_id, m.memory_level, m.memory_type, m.importance, m.confidence, m.description, m.reasoning, m.source, m.tags, m.created_at, m.updated_at, m.last_accessed, m.access_count, m.ttl, m.action_id, m.thought_id, m.artifact_id"
            if include_content:
                select_clause += ", m.content"

            count_query = "SELECT COUNT(m.memory_id) FROM memories m"
            data_query = f"SELECT {select_clause} FROM memories m"

            where_clauses = ["1=1"]
            params: List[Any] = []
            fts_params: List[Any] = []  # Params specific to FTS match

            # --- Apply Filters (same logic as before) ---
            if workflow_id:
                where_clauses.append("m.workflow_id = ?")
                params.append(workflow_id)
            if memory_level:
                where_clauses.append("m.memory_level = ?")
                params.append(memory_level.lower())
            if memory_type:
                where_clauses.append("m.memory_type = ?")
                params.append(memory_type.lower())
            if min_importance is not None:
                where_clauses.append("m.importance >= ?")
                params.append(min_importance)
            if max_importance is not None:
                where_clauses.append("m.importance <= ?")
                params.append(max_importance)
            if min_confidence is not None:
                where_clauses.append("m.confidence >= ?")
                params.append(min_confidence)
            if min_created_at_unix is not None:
                where_clauses.append("m.created_at >= ?")
                params.append(min_created_at_unix)
            if max_created_at_unix is not None:
                where_clauses.append("m.created_at <= ?")
                params.append(max_created_at_unix)

            now_unix = int(time.time())
            where_clauses.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
            params.append(now_unix)

            if tags and isinstance(tags, list) and len(tags) > 0:
                tags_json = json.dumps([str(tag).lower() for tag in tags])
                where_clauses.append("json_contains_all(m.tags, ?)")
                params.append(tags_json)

            if search_text:
                count_query += " JOIN memory_fts fts ON m.rowid = fts.rowid"
                data_query += " JOIN memory_fts fts ON m.rowid = fts.rowid"
                where_clauses.append("fts.memory_fts MATCH ?")
                fts_query_term = " OR ".join(search_text.strip().split())
                fts_params.append(fts_query_term)

            # Combine WHERE clauses
            where_sql = ""
            if len(where_clauses) > 1:
                where_sql = " WHERE " + " AND ".join(where_clauses)
                count_query += where_sql
                data_query += where_sql

            # --- Get Total Count ---
            async with conn.execute(count_query, params + fts_params) as cursor:
                row = await cursor.fetchone()
                total_matching_count = row[0] if row else 0

            # --- Apply Sorting ---
            order_clause = ""
            if sort_by == "relevance":
                order_clause = " ORDER BY compute_memory_relevance(m.importance, m.confidence, m.created_at, m.access_count, m.last_accessed)"
            elif sort_by in [
                "created_at",
                "updated_at",
                "importance",
                "confidence",
                "last_accessed",
                "access_count",
            ]:
                order_clause = f" ORDER BY m.{sort_by}"
            else:
                order_clause = " ORDER BY m.created_at"  # Default sort fallback
            order_clause += f" {sort_order}"

            # --- Apply Pagination ---
            limit_clause = " LIMIT ? OFFSET ?"

            # --- Execute Data Query ---
            final_query = data_query + order_clause + limit_clause
            final_params = params + fts_params + [limit, offset]

            memories_results = []  # Final list of processed memories
            async with conn.execute(final_query, final_params) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    memory_dict = dict(row)  # Convert row to dict
                    memory_dict["tags"] = await MemoryUtils.deserialize(memory_dict.get("tags"))
                    memory_dict["created_at_unix"] = memory_dict.get("created_at")
                    memory_dict["updated_at_unix"] = memory_dict.get("updated_at")
                    memory_dict["last_accessed_unix"] = memory_dict.get("last_accessed")

                    # --- Fetch Detailed Links (Full Implementation) ---
                    if include_links:
                        current_memory_id = memory_dict["memory_id"]
                        memory_dict["links"] = {
                            "outgoing": [],
                            "incoming": [],
                        }  # Initialize structure

                        # Fetch Outgoing Links
                        if link_direction in ["outgoing", "both"]:
                            outgoing_query = """
                            SELECT ml.link_id, ml.target_memory_id, ml.link_type, ml.strength, ml.description AS link_description,
                                   target_mem.description AS target_description, target_mem.memory_type AS target_type
                            FROM memory_links ml
                            JOIN memories target_mem ON ml.target_memory_id = target_mem.memory_id
                            WHERE ml.source_memory_id = ?
                            """
                            async with conn.execute(
                                outgoing_query, (current_memory_id,)
                            ) as link_cursor:
                                async for link_row in link_cursor:
                                    memory_dict["links"]["outgoing"].append(dict(link_row))

                        # Fetch Incoming Links
                        if link_direction in ["incoming", "both"]:
                            incoming_query = """
                            SELECT ml.link_id, ml.source_memory_id, ml.link_type, ml.strength, ml.description AS link_description,
                                   source_mem.description AS source_description, source_mem.memory_type AS source_type
                            FROM memory_links ml
                            JOIN memories source_mem ON ml.source_memory_id = source_mem.memory_id
                            WHERE ml.target_memory_id = ?
                            """
                            async with conn.execute(
                                incoming_query, (current_memory_id,)
                            ) as link_cursor:
                                async for link_row in link_cursor:
                                    memory_dict["links"]["incoming"].append(dict(link_row))

                    # Update access stats for the primary retrieved memory
                    await MemoryUtils._update_memory_access(conn, memory_dict["memory_id"])
                    await MemoryUtils._log_memory_operation(
                        conn,
                        memory_dict["workflow_id"],
                        "query_access",
                        memory_dict["memory_id"],
                        None,
                        {"query_filters": {"sort": sort_by, "limit": limit}},
                    )

                    memories_results.append(memory_dict)

            # Commit access updates
            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Memory query returned {len(memories_results)} of {total_matching_count} results.",
                emoji_key="search",
                time=processing_time,
            )

            return {
                "memories": memories_results,
                "total_matching_count": total_matching_count,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Failed to query memories: {str(e)}", emoji_key="x", exc_info=True)
        raise ToolError(f"Failed to query memories: {str(e)}") from e


# --- 8. Workflow Listing & Details ---

@with_tool_metrics
@with_error_handling
async def list_workflows(
    status: Optional[str] = None,
    tag: Optional[str] = None,
    after_date: Optional[str] = None,  # ISO Format string for filtering
    before_date: Optional[str] = None,  # ISO Format string for filtering
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Lists workflows matching specified criteria.
    Timestamps are returned as ISO 8601 strings.
    """
    try:
        # Validate status
        if status:
            try:
                WorkflowStatus(status.lower())
            except ValueError as e:
                raise ToolInputError(f"Invalid status: {status}", param_name="status") from e

        # Convert filter dates from ISO strings to Unix timestamps for querying
        after_ts: Optional[int] = None
        if after_date:
            try:
                # Use datetime.fromisoformat for robustness with timezone info
                dt_obj = datetime.fromisoformat(after_date.replace("Z", "+00:00"))
                after_ts = int(dt_obj.timestamp())
            except ValueError as e:
                raise ToolInputError(
                    "Invalid after_date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).", param_name="after_date"
                ) from e
        before_ts: Optional[int] = None
        if before_date:
            try:
                dt_obj = datetime.fromisoformat(before_date.replace("Z", "+00:00"))
                before_ts = int(dt_obj.timestamp())
            except ValueError as e:
                raise ToolInputError(
                    "Invalid before_date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).", param_name="before_date"
                ) from e

        # Validate pagination
        if limit < 1:
            raise ToolInputError("Limit must be >= 1", param_name="limit")
        if offset < 0:
            raise ToolInputError("Offset must be >= 0", param_name="offset")

        async with DBConnection(db_path) as conn:
            base_query = """
            SELECT DISTINCT w.workflow_id, w.title, w.description, w.goal, w.status,
                   w.created_at, w.updated_at, w.completed_at -- Fetch potentially mixed timestamp types
            FROM workflows w
            """
            count_query = "SELECT COUNT(DISTINCT w.workflow_id) FROM workflows w"
            joins = ""
            where_clauses = ["1=1"]
            params = []

            if tag:
                joins += " JOIN workflow_tags wt ON w.workflow_id = wt.workflow_id JOIN tags t ON wt.tag_id = t.tag_id"
                where_clauses.append("t.name = ?")
                params.append(tag)

            if status:
                where_clauses.append("w.status = ?")
                params.append(status.lower())
            if after_ts is not None:  # Use timestamp for filtering
                where_clauses.append("w.created_at >= ?")
                params.append(after_ts)
            if before_ts is not None:  # Use timestamp for filtering
                where_clauses.append("w.created_at <= ?")
                params.append(before_ts)

            where_sql = " WHERE " + " AND ".join(where_clauses)
            full_base_query = base_query + joins + where_sql
            full_count_query = count_query + joins + where_sql

            # Get total count
            cursor = await conn.execute(full_count_query, params)
            row = await cursor.fetchone()
            await cursor.close()
            total_count = row[0] if row else 0

            # Get workflow data
            data_query = full_base_query + " ORDER BY w.updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            workflows_list = []
            workflow_ids_fetched = []

            cursor = await conn.execute(data_query, params)
            rows = await cursor.fetchall()
            await cursor.close()

            for row in rows:
                wf_data = dict(row)
                # Keep raw timestamps for now
                wf_data["tags"] = []  # Initialize tags list
                workflows_list.append(wf_data)
                workflow_ids_fetched.append(wf_data["workflow_id"])

            # Batch fetch tags for fetched workflows
            if workflow_ids_fetched:
                placeholders = ",".join("?" * len(workflow_ids_fetched))
                tags_query = f"""
                 SELECT wt.workflow_id, t.name
                 FROM tags t JOIN workflow_tags wt ON t.tag_id = wt.tag_id
                 WHERE wt.workflow_id IN ({placeholders})
                 """
                tags_map = defaultdict(list)
                tags_cursor = await conn.execute(tags_query, workflow_ids_fetched)
                async for tag_row in tags_cursor:
                    tags_map[tag_row["workflow_id"]].append(tag_row["name"])
                await tags_cursor.close()

                # Assign tags to workflows
                for wf in workflows_list:
                    wf["tags"] = tags_map.get(wf["workflow_id"], [])

            # --- Perform FINAL timestamp conversion before returning ---
            for wf_data in workflows_list:
                 for ts_key in ["created_at", "updated_at", "completed_at"]:
                     # Apply the safe formatter
                     wf_data[ts_key] = safe_format_timestamp(wf_data.get(ts_key))

            result = {"workflows": workflows_list, "total_count": total_count, "success": True}
            logger.info(
                f"Listed {len(workflows_list)} workflows (total matching: {total_count})",
                emoji_key="scroll",
            )
            return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error listing workflows: {e}", exc_info=True)
        raise ToolError(f"Failed to list workflows: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_workflow_details(
    workflow_id: str,
    include_actions: bool = True,  # Default to True now
    include_artifacts: bool = True,
    include_thoughts: bool = True,
    include_memories: bool = False,  # Keep memories optional for performance
    memories_limit: int = 20,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves comprehensive details about a specific workflow, including related items.
    Timestamps are returned as ISO 8601 strings.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    try:
        async with DBConnection(db_path) as conn:
            # --- Get Workflow Core Info & Tags ---
            cursor = await conn.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_row = await cursor.fetchone()
            await cursor.close()
            if not wf_row:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            workflow_details = dict(wf_row) # Keep raw timestamps

            workflow_details["metadata"] = await MemoryUtils.deserialize(
                workflow_details.get("metadata")
            )

            # Fetch tags
            workflow_details["tags"] = []
            tags_cursor = await conn.execute(
                "SELECT t.name FROM tags t JOIN workflow_tags wt ON t.tag_id = wt.tag_id WHERE wt.workflow_id = ?",
                (workflow_id,),
            )
            workflow_details["tags"] = [row["name"] for row in await tags_cursor.fetchall()]
            await tags_cursor.close()

            # --- Get Actions ---
            if include_actions:
                workflow_details["actions"] = []
                actions_query = """
                SELECT a.*, GROUP_CONCAT(t.name) as tags_str
                FROM actions a
                LEFT JOIN action_tags at ON a.action_id = at.action_id
                LEFT JOIN tags t ON at.tag_id = t.tag_id
                WHERE a.workflow_id = ?
                GROUP BY a.action_id
                ORDER BY a.sequence_number ASC
                """
                actions_cursor = await conn.execute(actions_query, (workflow_id,))
                async for row in actions_cursor:
                    action = dict(row) # Keep raw timestamps
                    action["tool_args"] = await MemoryUtils.deserialize(action.get("tool_args"))
                    action["tool_result"] = await MemoryUtils.deserialize(action.get("tool_result"))
                    action["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                    action.pop("tags_str", None)
                    workflow_details["actions"].append(action)
                await actions_cursor.close()

            # --- Get Artifacts ---
            if include_artifacts:
                workflow_details["artifacts"] = []
                artifacts_query = """
                SELECT a.*, GROUP_CONCAT(t.name) as tags_str
                FROM artifacts a
                LEFT JOIN artifact_tags att ON a.artifact_id = att.artifact_id
                LEFT JOIN tags t ON att.tag_id = t.tag_id
                WHERE a.workflow_id = ?
                GROUP BY a.artifact_id
                ORDER BY a.created_at ASC
                """
                artifacts_cursor = await conn.execute(artifacts_query, (workflow_id,))
                async for row in artifacts_cursor:
                    artifact = dict(row) # Keep raw timestamp
                    artifact["metadata"] = await MemoryUtils.deserialize(artifact.get("metadata"))
                    artifact["is_output"] = bool(artifact["is_output"])
                    artifact["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                    artifact.pop("tags_str", None)
                    if artifact.get("content") and len(artifact["content"]) > 200:
                        artifact["content_preview"] = artifact["content"][:197] + "..."
                    workflow_details["artifacts"].append(artifact)
                await artifacts_cursor.close()

            # --- Get Thought Chains & Thoughts ---
            if include_thoughts:
                workflow_details["thought_chains"] = []
                chain_cursor = await conn.execute(
                    "SELECT * FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC",
                    (workflow_id,),
                )
                async for chain_row_data in chain_cursor:
                    thought_chain = dict(chain_row_data) # Keep raw timestamp
                    thought_chain["thoughts"] = []
                    thought_cursor = await conn.execute(
                        "SELECT * FROM thoughts WHERE thought_chain_id = ? ORDER BY sequence_number ASC",
                        (thought_chain["thought_chain_id"],),
                    )
                    async for thought_row_data in thought_cursor:
                        thought = dict(thought_row_data) # Keep raw timestamp
                        thought_chain["thoughts"].append(thought)
                    await thought_cursor.close()
                    workflow_details["thought_chains"].append(thought_chain)
                await chain_cursor.close()

            # --- Get Recent/Important Memories (Optional) ---
            if include_memories:
                memories_query = """
                 SELECT memory_id, content, memory_type, memory_level, importance, created_at
                 FROM memories
                 WHERE workflow_id = ?
                 ORDER BY importance DESC, created_at DESC
                 LIMIT ?
                 """
                workflow_details["memories_sample"] = []
                mem_cursor = await conn.execute(memories_query, (workflow_id, memories_limit))
                async for row in mem_cursor:
                    mem = dict(row) # Keep raw timestamp
                    # Add iso format specifically for the sample if needed, or keep unix
                    # Let's keep unix for consistency internally, formatter will handle later
                    # if mem.get("created_at"):
                    #     mem["created_at_iso"] = to_iso_z(mem["created_at"])
                    mem["created_at_unix"] = mem.get("created_at")
                    if mem.get("content") and len(mem["content"]) > 150:
                        mem["content_preview"] = mem["content"][:147] + "..."
                    workflow_details["memories_sample"].append(mem)
                await mem_cursor.close()

            workflow_details["success"] = True
            logger.info(f"Retrieved details for workflow {workflow_id}", emoji_key="books")

            # --- Perform FINAL timestamp conversion JUST before returning ---
            timestamp_keys_to_convert = {
                "workflow": ["created_at", "updated_at", "completed_at"],
                "action": ["started_at", "completed_at"],
                "artifact": ["created_at"],
                "thought_chain": ["created_at"],
                "thought": ["created_at"],
            }

            # Apply conversion safely using the helper
            for key in timestamp_keys_to_convert["workflow"]:
                if key in workflow_details:
                    workflow_details[key] = safe_format_timestamp(workflow_details[key])

            for action in workflow_details.get("actions", []):
                for key in timestamp_keys_to_convert["action"]:
                    if key in action:
                        action[key] = safe_format_timestamp(action[key])

            for artifact in workflow_details.get("artifacts", []):
                for key in timestamp_keys_to_convert["artifact"]:
                     if key in artifact:
                         artifact[key] = safe_format_timestamp(artifact[key])

            for chain in workflow_details.get("thought_chains", []):
                for key in timestamp_keys_to_convert["thought_chain"]:
                    if key in chain:
                         chain[key] = safe_format_timestamp(chain[key])
                for thought in chain.get("thoughts", []):
                    for key in timestamp_keys_to_convert["thought"]:
                         if key in thought:
                             thought[key] = safe_format_timestamp(thought[key])

            return workflow_details # Return the fully formatted details
    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow details for {workflow_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to get workflow details: {str(e)}") from e


# --- 9. Action Details ---
@with_tool_metrics
@with_error_handling
async def get_recent_actions(
    workflow_id: str,
    limit: int = 5,
    action_type: Optional[str] = None,
    status: Optional[str] = None,
    include_tool_results: bool = True,
    include_reasoning: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves the most recent actions for a workflow, optionally filtered.

    Use this tool to refresh your understanding of what has been done recently in a workflow.
    This helps maintain context when working on complex tasks or when resuming a workflow
    after an interruption. By default, includes tool results and reasoning.

    Args:
        workflow_id: The ID of the workflow.
        limit: (Optional) Maximum number of actions to return. Default 5.
        action_type: (Optional) Filter by action type (e.g., 'tool_use', 'reasoning').
        status: (Optional) Filter by action status (e.g., 'completed', 'failed').
        include_tool_results: (Optional) Whether to include tool results. Default True.
        include_reasoning: (Optional) Whether to include reasoning. Default True.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        A dictionary containing the list of recent actions:
        {
            "actions": [
                {
                    "action_id": "uuid-string",
                    "action_type": "tool_use",
                    "title": "Reading customer data",
                    "tool_name": "read_file",
                    "tool_args": { "path": "example.txt" },
                    "tool_result": { "content": "file content..." }, // Included by default
                    "reasoning": "Reading this file to extract project requirements", // Included by default
                    "status": "completed",
                    "started_at": "2025-04-13T12:34:56Z",
                    "completed_at": "2025-04-13T12:35:01Z",
                    "sequence_number": 3,
                    "parent_action_id": null,
                    "tags": ["data_processing"]
                },
                ...
            ],
            "workflow_title": "Data Analysis Project",
            "workflow_id": "uuid-string",
            "success": true
        }

    Raises:
        ToolInputError: If the workflow doesn't exist or parameters are invalid.
        ToolError: If the database operation fails.
    """
    try:
        # --- Validations ---
        if not isinstance(limit, int) or limit < 1:
            raise ToolInputError("Limit must be a positive integer", param_name="limit")
        if action_type:
            try:
                ActionType(action_type.lower())
            except ValueError as e:
                raise ToolInputError(
                    f"Invalid action_type '{action_type}'. Must be one of: {[t.value for t in ActionType]}",
                    param_name="action_type",
                ) from e
        if status:
            try:
                ActionStatus(status.lower())
            except ValueError as e:
                raise ToolInputError(
                    f"Invalid status '{status}'. Must be one of: {[s.value for s in ActionStatus]}",
                    param_name="status",
                ) from e

        async with DBConnection(db_path) as conn:
            # --- Check Workflow & Get Title ---
            cursor = await conn.execute(
                "SELECT title FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_row = await cursor.fetchone()
            await cursor.close()
            if not wf_row:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")
            workflow_title = wf_row["title"]

            # --- Build Query ---
            select_fields = [
                "a.action_id",
                "a.action_type",
                "a.title",
                "a.tool_name",
                "a.tool_args",
                "a.status",
                "a.started_at",
                "a.completed_at",
                "a.sequence_number",
                "a.parent_action_id",
                "GROUP_CONCAT(t.name) as tags_str",
            ]
            if include_reasoning:
                select_fields.append("a.reasoning")
            if include_tool_results:
                select_fields.append("a.tool_result")

            query = f"SELECT {', '.join(select_fields)} FROM actions a LEFT JOIN action_tags at ON a.action_id = at.action_id LEFT JOIN tags t ON at.tag_id = t.tag_id WHERE a.workflow_id = ?"
            params: List[Any] = [workflow_id]

            if action_type:
                query += " AND a.action_type = ?"
                params.append(action_type.lower())
            if status:
                query += " AND a.status = ?"
                params.append(status.lower())

            query += " GROUP BY a.action_id ORDER BY a.sequence_number DESC LIMIT ?"
            params.append(limit)

            # --- Execute Query & Process Results ---
            actions_list = []
            cursor = await conn.execute(query, params)
            async for row in cursor:
                action = dict(row)
                if action.get("started_at"):
                    action["started_at"] = to_iso_z(action["started_at"])
                if action.get("completed_at"):
                    action["completed_at"] = to_iso_z(action["completed_at"])

                action["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                action.pop("tags_str", None)

                if "tool_args" in action:
                    action["tool_args"] = await MemoryUtils.deserialize(action.get("tool_args"))
                if "tool_result" in action:
                    action["tool_result"] = await MemoryUtils.deserialize(action.get("tool_result"))

                actions_list.append(action)
            await cursor.close()

            # --- Prepare Final Result ---
            result = {
                "actions": actions_list,
                "workflow_title": workflow_title,
                "workflow_id": workflow_id,
                "success": True,
            }
            logger.info(
                f"Retrieved {len(actions_list)} recent actions for workflow {workflow_id}",
                emoji_key="rewind",
            )
            return result

    except ToolInputError:
        raise  # Propagate validation errors
    except Exception as e:
        logger.error(f"Error getting recent actions for workflow {workflow_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to get recent actions: {str(e)}") from e


# --- 10. Artifact Details ---
@with_tool_metrics
@with_error_handling
async def get_artifacts(
    workflow_id: str,
    artifact_type: Optional[str] = None,
    tag: Optional[str] = None,
    is_output: Optional[bool] = None,
    include_content: bool = False,  # Default False for list view
    limit: int = 10,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves artifacts created during a workflow, optionally filtered.
    Timestamps are returned as ISO 8601 strings.
    """
    try:
        # Validations
        if limit < 1:
            raise ToolInputError("Limit must be >= 1", param_name="limit")
        if artifact_type:
            try:
                ArtifactType(artifact_type.lower())
            except ValueError as e:
                raise ToolInputError(
                    f"Invalid artifact_type: {artifact_type}", param_name="artifact_type"
                ) from e

        async with DBConnection(db_path) as conn:
            # Check workflow
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()
            if not wf_exists:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            # Build query
            select_cols = "a.artifact_id, a.action_id, a.artifact_type, a.name, a.description, a.path, a.metadata, a.created_at, a.is_output, GROUP_CONCAT(t.name) as tags_str"
            if include_content:
                select_cols += ", a.content"

            query = f"SELECT {select_cols} FROM artifacts a"
            joins = " LEFT JOIN artifact_tags att ON a.artifact_id = att.artifact_id LEFT JOIN tags t ON att.tag_id = t.tag_id"
            where_clauses = ["a.workflow_id = ?"]
            params = [workflow_id]

            if tag:
                if "LEFT JOIN artifact_tags" not in joins:
                    joins += " LEFT JOIN artifact_tags att ON a.artifact_id = att.artifact_id LEFT JOIN tags t ON att.tag_id = t.tag_id"
                where_clauses.append("t.name = ?")
                params.append(tag)

            if artifact_type:
                where_clauses.append("a.artifact_type = ?")
                params.append(artifact_type.lower())
            if is_output is not None:
                where_clauses.append("a.is_output = ?")
                params.append(1 if is_output else 0)

            where_sql = " WHERE " + " AND ".join(where_clauses)
            group_by = " GROUP BY a.artifact_id"
            order_by = " ORDER BY a.created_at DESC"
            limit_sql = " LIMIT ?"
            params.append(limit)

            final_query = query + joins + where_sql + group_by + order_by + limit_sql

            artifacts_list = []
            cursor = await conn.execute(final_query, params)
            async for row in cursor:
                artifact = dict(row)
                if artifact.get("created_at"):
                    artifact["created_at"] = to_iso_z(artifact["created_at"])
                artifact["metadata"] = await MemoryUtils.deserialize(artifact.get("metadata"))
                artifact["is_output"] = bool(artifact["is_output"])
                artifact["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                artifact.pop("tags_str", None)
                if (
                    not include_content and "content" in artifact and artifact.get("content")
                ):  # Check key exists before accessing
                    if len(artifact["content"]) > 100:
                        artifact["content_preview"] = artifact["content"][:97] + "..."
                    if not include_content:
                        del artifact["content"]  # Remove original content if not requested
                artifacts_list.append(artifact)
            await cursor.close()

            result = {"artifacts": artifacts_list, "workflow_id": workflow_id, "success": True}
            logger.info(
                f"Retrieved {len(artifacts_list)} artifacts for workflow {workflow_id}",
                emoji_key="open_file_folder",
            )
            return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting artifacts: {e}", exc_info=True)
        raise ToolError(f"Failed to get artifacts: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_artifact_by_id(
    artifact_id: str, include_content: bool = True, db_path: str = agent_memory_config.db_path
) -> Dict[str, Any]:
    """Retrieves a specific artifact by its ID.
    Timestamps are returned as ISO 8601 strings.
    """
    if not artifact_id:
        raise ToolInputError("Artifact ID required.", param_name="artifact_id")
    try:
        async with DBConnection(db_path) as conn:
            select_cols = "a.*, GROUP_CONCAT(t.name) as tags_str"
            query = f"""
             SELECT {select_cols}
             FROM artifacts a
             LEFT JOIN artifact_tags att ON a.artifact_id = att.artifact_id
             LEFT JOIN tags t ON att.tag_id = t.tag_id
             WHERE a.artifact_id = ?
             GROUP BY a.artifact_id
             """
            cursor = await conn.execute(query, (artifact_id,))
            row = await cursor.fetchone()
            await cursor.close()
            if not row:
                raise ToolInputError(f"Artifact {artifact_id} not found.", param_name="artifact_id")

            artifact = dict(row)
            if artifact.get("created_at"):
                artifact["created_at"] = to_iso_z(artifact["created_at"])
            artifact["metadata"] = await MemoryUtils.deserialize(artifact.get("metadata"))
            artifact["is_output"] = bool(artifact["is_output"])
            artifact["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
            artifact.pop("tags_str", None)

            if not include_content:
                if "content" in artifact:  # Check key exists before deleting
                    del artifact["content"]

            # Update access stats for related memory if possible
            mem_cursor = await conn.execute(
                "SELECT memory_id, workflow_id FROM memories WHERE artifact_id = ?", (artifact_id,)
            )
            mem_row = await mem_cursor.fetchone()
            await mem_cursor.close()
            if mem_row:
                # Use workflow_id from memory for logging if artifact one missing? No, artifact should have one.
                artifact_workflow_id = artifact.get(
                    "workflow_id"
                )  # Get workflow_id from artifact data
                if artifact_workflow_id:
                    await MemoryUtils._update_memory_access(conn, mem_row["memory_id"])
                    await MemoryUtils._log_memory_operation(
                        conn,
                        artifact_workflow_id,
                        "access_via_artifact",
                        mem_row["memory_id"],
                        None,
                        {"artifact_id": artifact_id},
                    )
                    await conn.commit()
                else:
                    logger.warning(
                        f"Cannot log memory access via artifact {artifact_id} as workflow_id is missing from artifact record."
                    )

            artifact["success"] = True
            logger.info(f"Retrieved artifact {artifact_id}", emoji_key="page_facing_up")
            return artifact

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving artifact {artifact_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve artifact: {str(e)}") from e


# --- 11. Thought Details ---
@with_tool_metrics
@with_error_handling
async def create_thought_chain(
    workflow_id: str,
    title: str,
    initial_thought: Optional[str] = None,
    initial_thought_type: str = "goal",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Creates a new reasoning chain for tracking related thoughts.

    This operation is atomic: the chain and optional initial thought are
    created within a single database transaction.

    Args:
        workflow_id: The ID of the workflow this chain belongs to.
        title: A descriptive title for the thought chain.
        initial_thought: (Optional) Content for the first thought in the chain.
        initial_thought_type: (Optional) Type for the initial thought. Default 'goal'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing information about the created thought chain.
        Timestamps are returned as ISO 8601 strings.
        {
            "thought_chain_id": "uuid-string",
            "workflow_id": "workflow-uuid",
            "title": "Reasoning for X",
            "created_at": "iso-timestampZ",
            "initial_thought_id": "uuid-string" | None,
            "success": true
        }

    Raises:
        ToolInputError: If title is empty, type is invalid, or workflow not found.
        ToolError: If the database operation fails.
    """
    try:
        if not title:
            raise ToolInputError("Thought chain title required", param_name="title")
        initial_thought_type_enum = None
        if initial_thought:
            try:
                initial_thought_type_enum = ThoughtType(initial_thought_type.lower())
            except ValueError as e:
                valid_types = [t.value for t in ThoughtType]
                raise ToolInputError(
                    f"Invalid initial_thought_type '{initial_thought_type}'. Must be one of: {', '.join(valid_types)}",
                    param_name="initial_thought_type",
                ) from e

        thought_chain_id = MemoryUtils.generate_id()
        now_unix = int(time.time())
        thought_id = None  # Initialize thought_id

        # Use the transaction manager for atomicity
        db_manager = DBConnection(db_path)
        async with db_manager.transaction() as conn:
            # Check workflow exists (using the transaction connection)
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()  # Close cursor after fetch
            if not wf_exists:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            # Insert chain using Unix timestamp
            await conn.execute(
                "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, ?, ?)",
                (thought_chain_id, workflow_id, title, now_unix),
            )
            # Update workflow using Unix timestamp
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            )

            # Add initial thought if specified, passing the transaction connection
            if initial_thought and initial_thought_type_enum:
                thought_result = await record_thought(
                    workflow_id=workflow_id,
                    content=initial_thought,
                    thought_type=initial_thought_type_enum.value,
                    thought_chain_id=thought_chain_id,
                    db_path=db_path,
                    conn=conn,  # Pass the active transaction connection
                )
                thought_id = thought_result.get("thought_id")

        # Prepare result dictionary outside the transaction block
        result = {
            "thought_chain_id": thought_chain_id,
            "workflow_id": workflow_id,
            "title": title,
            "created_at": to_iso_z(now_unix),  # Format the timestamp used for creation for output
            "initial_thought_id": thought_id,  # Include if created
            "success": True,
        }
        logger.info(
            f"Created thought chain '{title}' ({thought_chain_id}) in workflow {workflow_id}",
            emoji_key="thought_balloon",
        )
        return result

    except ToolInputError:
        raise
    except Exception as e:
        # Log the error context
        logger.error(
            f"Error creating thought chain '{title}' for workflow {workflow_id}: {e}", exc_info=True
        )
        # Re-raise as a ToolError for consistent error handling
        raise ToolError(f"Failed to create thought chain: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def get_thought_chain(
    thought_chain_id: str, include_thoughts: bool = True, db_path: str = agent_memory_config.db_path) -> Dict[str, Any]:
    """Retrieves a thought chain and optionally its thoughts."""
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")
    try:
        async with DBConnection(db_path) as conn:
            # Get chain info
            cursor = await conn.execute(
                "SELECT * FROM thought_chains WHERE thought_chain_id = ?", (thought_chain_id,)
            )
            chain_row = await cursor.fetchone()
            await cursor.close()
            if not chain_row:
                raise ToolInputError(
                    f"Thought chain {thought_chain_id} not found.", param_name="thought_chain_id"
                )

            thought_chain_details = dict(chain_row)
            if thought_chain_details.get("created_at"):
                thought_chain_details["created_at"] = to_iso_z(thought_chain_details["created_at"])

            # Get thoughts
            thought_chain_details["thoughts"] = []
            if include_thoughts:
                thought_cursor = await conn.execute(
                    "SELECT * FROM thoughts WHERE thought_chain_id = ? ORDER BY sequence_number ASC",
                    (thought_chain_id,),
                )
                async for row in thought_cursor:
                    thought = dict(row)
                    if thought.get("created_at"):
                        thought["created_at"] = to_iso_z(thought["created_at"])
                    thought_chain_details["thoughts"].append(thought)
                await thought_cursor.close()

            thought_chain_details["success"] = True
            logger.info(
                f"Retrieved thought chain {thought_chain_id} with {len(thought_chain_details.get('thoughts', []))} thoughts",
                emoji_key="left_speech_bubble",
            )
            return thought_chain_details

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting thought chain {thought_chain_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to get thought chain: {str(e)}") from e
    

# ======================================================
# Helper Function for Working Memory Management 
# ======================================================


async def _add_to_active_memories(
    conn: aiosqlite.Connection, context_id: str, memory_id: str
) -> bool:
    """Adds a memory to the working memory list for a context, enforcing size limits.
       Internal helper function. Assumes context_id exists.

    Args:
        conn: Active database connection.
        context_id: Context ID (maps to state_id in cognitive_states).
        memory_id: Memory ID to add.

    Returns:
        True if successful (even if no change needed), False otherwise.
    """
    try:
        # Get current working memory list and workflow_id
        async with conn.execute(
            "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        ) as cursor:
            row = await cursor.fetchone()
            await cursor.close() # Ensure cursor is closed
            if not row:
                logger.warning(
                    f"Context {context_id} not found while trying to add memory {memory_id}."
                )
                return False  # Context must exist
            workflow_id = row["workflow_id"]
            current_working_memory_ids = await MemoryUtils.deserialize(row["working_memory"]) or []

        if memory_id in current_working_memory_ids:
            logger.debug(f"Memory {memory_id} already in working memory for context {context_id}.")
            return True  # Already present

        # Check if memory exists before adding
        async with conn.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
        ) as cursor:
             mem_exists = await cursor.fetchone()
             await cursor.close() # Ensure cursor is closed
             if not mem_exists:
                logger.warning(
                    f"Memory {memory_id} not found, cannot add to working memory for context {context_id}."
                )
                return False

        removed_id = None
        if len(current_working_memory_ids) >= agent_memory_config.max_working_memory_size:
            if not current_working_memory_ids:
                logger.warning(
                    f"Working memory limit reached ({agent_memory_config.max_working_memory_size}) but list is empty for context {context_id}. Adding new memory."
                )
                # Fallthrough to append if list is somehow empty despite check
                current_working_memory_ids.append(memory_id)

            else:
                # Fetch relevance scores for current working memories
                placeholders = ", ".join(["?"] * len(current_working_memory_ids))
                query = f"""
                 SELECT memory_id,
                        compute_memory_relevance(importance, confidence, created_at, access_count, last_accessed) as relevance
                 FROM memories
                 WHERE memory_id IN ({placeholders})
                 """
                relevance_scores = []
                async with conn.execute(query, current_working_memory_ids) as cursor:
                    async for score_row in cursor: # Use different variable name
                        relevance_scores.append((score_row["memory_id"], score_row["relevance"]))
                # Cursor closed automatically by async with

                if relevance_scores:
                    # Sort by relevance (ascending) to find the least relevant
                    relevance_scores.sort(key=lambda x: x[1])
                    removed_id = relevance_scores[0][0]
                    try:
                        current_working_memory_ids.remove(removed_id)
                        logger.debug(
                            f"Removed least relevant memory {_fmt_id(removed_id)} from context {context_id} working memory."
                        )
                    except ValueError:
                        # This might happen in race conditions if another process modified the list
                        logger.warning(
                            f"Tried to remove memory {_fmt_id(removed_id)} from working memory {context_id}, but it was not found in the list."
                        )
                        removed_id = None # Ensure removed_id is None if removal failed

                    # Log the removal if successful
                    if removed_id:
                         await MemoryUtils._log_memory_operation(
                            conn,
                            workflow_id,
                            "remove_from_working",
                            removed_id,
                            None,
                            {
                                "context_id": context_id,
                                "reason": "working_memory_limit",
                                "removed_relevance": relevance_scores[0][1],
                            },
                        )
                else:
                    logger.warning(f"Could not fetch relevance scores to determine which memory to remove from context {context_id}. Cannot add new memory.")
                    return False # Don't add if we couldn't determine which to remove

                # Add the new memory AFTER potentially removing the old one
                current_working_memory_ids.append(memory_id)

        else:
            # Space available, just append
            current_working_memory_ids.append(memory_id)

        # Update the cognitive state
        await conn.execute(
            "UPDATE cognitive_states SET working_memory = ?, last_active = ? WHERE state_id = ?",
            (
                await MemoryUtils.serialize(current_working_memory_ids),
                int(time.time()),
                context_id,
            ),
        )
        logger.debug(
            f"Added memory {_fmt_id(memory_id)} to context {context_id} working memory. New count: {len(current_working_memory_ids)}"
        )
        # Log the addition
        await MemoryUtils._log_memory_operation(
            conn, workflow_id, "add_to_working", memory_id, None, {"context_id": context_id}
        )

        return True

    except Exception as e:
        # Log with more specific context if possible
        logger.error(
            f"Error in _add_to_active_memories for context {context_id}, memory {memory_id}: {e}", exc_info=True
        )
        return False


# --- 12. Working Memory Management ---

@with_tool_metrics
@with_error_handling
async def get_working_memory(
    context_id: str,
    include_content: bool = True,
    include_links: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves the current working memory (list of memory IDs and their details) for a given context.

    Working memory holds the information the agent is actively using. This tool fetches
    the details of memories currently stored in the working_memory list for a specific context ID,
    including their outgoing links if requested. Access statistics are updated for retrieved memories.

    Args:
        context_id: The context identifier (maps to state_id in cognitive_states).
        include_content: (Optional) Whether to include the full content of memories. Default True.
        include_links: (Optional) Whether to include outgoing links from memories. Default True.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the working memory details:
        {
            "context_id": "context-uuid",
            "workflow_id": "workflow-uuid",
            "focal_memory_id": "memory-uuid" | None,
            "working_memories": [
                {
                    "memory_id": "...",
                    ...,
                    "links": [ { "target_memory_id": "...", "link_type": "...", ... } ] # If include_links=True
                },
                ...
            ],
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If the context ID is not provided or not found.
        ToolError: If the database operation fails.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Get Cognitive State & Working Memory IDs ---
            # Select all relevant columns from cognitive_states
            async with conn.execute(
                "SELECT * FROM cognitive_states WHERE state_id = ?", (context_id,)
            ) as cursor:
                state_row = await cursor.fetchone()
                await cursor.close() # Close cursor
                if not state_row:
                    logger.warning(
                        f"Cognitive state for context {context_id} not found. Returning empty working memory."
                    )
                    return {
                        "context_id": context_id,
                        "workflow_id": None,
                        "focal_memory_id": None, # Reflects that state wasn't found
                        "working_memories": [],
                        "success": True,
                        "processing_time": time.time() - start_time,
                    }

                state = dict(state_row)
                workflow_id = state.get("workflow_id")
                focal_memory_id = state.get("focal_memory_id") # Fetch the potentially set focal ID
                working_memory_ids = (
                    await MemoryUtils.deserialize(state.get("working_memory")) or []
                )

            working_memories_list = []
            if working_memory_ids:
                # --- 2. Fetch Memory Details ---
                placeholders = ", ".join(["?"] * len(working_memory_ids))
                # Define columns to select
                select_cols_list = [
                    "memory_id", "workflow_id", "description", "memory_type",
                    "memory_level", "importance", "confidence", "created_at",
                    "tags", "action_id", "thought_id", "artifact_id",
                    "reasoning", "source", "context", "updated_at",
                    "last_accessed", "access_count", "ttl", "embedding_id",
                ]
                if include_content:
                    select_cols_list.append("content")
                select_cols = ", ".join(select_cols_list)

                memory_map = {}  # Use a map for efficient link attachment later
                query = f"SELECT {select_cols} FROM memories WHERE memory_id IN ({placeholders})"
                async with conn.execute(query, working_memory_ids) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        mem_dict = dict(row)
                        mem_dict["tags"] = await MemoryUtils.deserialize(mem_dict.get("tags"))
                        mem_dict["context"] = await MemoryUtils.deserialize(mem_dict.get("context"))
                        mem_dict["created_at_unix"] = row["created_at"]
                        mem_dict["updated_at_unix"] = row["updated_at"]
                        mem_dict["last_accessed_unix"] = row["last_accessed"]
                        memory_map[row["memory_id"]] = mem_dict

                # --- 3. Fetch Links if Requested ---
                if include_links:
                    links_map = defaultdict(list)
                    links_query = f"""
                        SELECT source_memory_id, target_memory_id, link_type, strength, description, link_id, created_at
                        FROM memory_links
                        WHERE source_memory_id IN ({placeholders})
                    """
                    async with conn.execute(links_query, working_memory_ids) as link_cursor:
                        async for link_row in link_cursor:
                            link_data = dict(link_row)
                            link_data["created_at_unix"] = link_row["created_at"]
                            links_map[link_row["source_memory_id"]].append(link_data)

                    # Attach links to memories
                    for mem_id in memory_map:
                        memory_map[mem_id]["links"] = links_map.get(mem_id, [])

                # --- 4. Reconstruct List & Update Access Stats ---
                # Preserve order from the working_memory list if possible
                update_tasks = []
                for mem_id in working_memory_ids:
                    if mem_id in memory_map:
                        working_memories_list.append(memory_map[mem_id])
                        # Create tasks for updating access stats
                        update_tasks.append(MemoryUtils._update_memory_access(conn, mem_id))
                        if workflow_id: # Check if workflow_id is known
                           update_tasks.append(
                               MemoryUtils._log_memory_operation(
                                   conn, workflow_id, "access_working", mem_id, None, {"context_id": context_id}
                               )
                           )

                # --- 5. Commit Access Updates (Run concurrently) ---
                if update_tasks:
                    await asyncio.gather(*update_tasks)
                    await conn.commit()

            # --- 6. Return Result ---
            processing_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(working_memories_list)} working memories for context {context_id}",
                emoji_key="brain",
            )
            return {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "focal_memory_id": focal_memory_id,
                "working_memories": working_memories_list,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting working memory for {context_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to get working memory: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def focus_memory(
    memory_id: str, context_id: str, add_to_working: bool = True, db_path: str = agent_memory_config.db_path
) -> Dict[str, Any]:
    """Sets a specific memory as the current focus for a context.

    Marks a memory as the primary item of attention by updating the `focal_memory_id`
    in the `cognitive_states` table. Optionally adds the memory to the working memory list
    using the `_add_to_active_memories` helper if not already present.

    Args:
        memory_id: ID of the memory to focus on.
        context_id: The context identifier (maps to state_id in cognitive_states).
        add_to_working: (Optional) If True, add the memory to the working memory list
                       if it's not already there (may evict another memory). Default True.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary confirming the focus update.
        {
            "context_id": "context-uuid",
            "focused_memory_id": "memory-uuid",
            "workflow_id": "workflow-uuid",
            "added_to_working": true | false,
            "success": true,
            "processing_time": 0.05
        }

    Raises:
        ToolInputError: If memory or context doesn't exist, or if they belong to different workflows.
        ToolError: If the database operation fails.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # Check memory exists and get its workflow_id
            async with conn.execute(
                "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
            ) as cursor:
                mem_row = await cursor.fetchone()
                await cursor.close() # Close cursor
                if not mem_row:
                    raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
                mem_workflow_id = mem_row["workflow_id"]

            # Check context exists and belongs to the same workflow
            async with conn.execute(
                "SELECT workflow_id FROM cognitive_states WHERE state_id = ?", (context_id,)
            ) as cursor:
                state_row = await cursor.fetchone()
                await cursor.close() # Close cursor
                if not state_row:
                    raise ToolInputError(
                        f"Context {context_id} not found.", param_name="context_id"
                    )
                if state_row["workflow_id"] != mem_workflow_id:
                    raise ToolInputError(
                        f"Memory {memory_id} (wf={mem_workflow_id}) does not belong to context {context_id}'s workflow ({state_row['workflow_id']})"
                    )

            # Add to working memory if requested (uses helper which knows the correct column name)
            added = False
            if add_to_working:
                added = await _add_to_active_memories(conn, context_id, memory_id)
                if not added:
                    logger.warning(
                        f"Failed to add memory {_fmt_id(memory_id)} to working set for context {context_id}, but proceeding to set focus."
                    )
                    added = False

            # Update the focal memory and last_active timestamp in the cognitive state
            now_unix = int(time.time())
            await conn.execute(
                "UPDATE cognitive_states SET focal_memory_id = ?, last_active = ? WHERE state_id = ?",
                (memory_id, now_unix, context_id),
            )

            # Log focus operation
            await MemoryUtils._log_memory_operation(
                conn, mem_workflow_id, "focus", memory_id, None, {"context_id": context_id}
            )

            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Set memory {_fmt_id(memory_id)} as focus for context {context_id}", emoji_key="target"
            )
            return {
                "context_id": context_id,
                "focused_memory_id": memory_id,
                "workflow_id": mem_workflow_id,
                "added_to_working": added,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(
            f"Error focusing memory {memory_id} for context {context_id}: {e}", exc_info=True
        )
        raise ToolError(f"Failed to focus memory: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def optimize_working_memory(
    context_id: str,
    target_size: int = agent_memory_config.max_working_memory_size,
    strategy: str = "balanced",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Optimizes the working memory for a context by retaining the most relevant items.

    Reduces the number of memory IDs in the `working_memory` list down to the target size
    based on the chosen strategy.

    Args:
        context_id: The context identifier (maps to state_id in cognitive_states).
        target_size: (Optional) Desired number of memories after optimization. Default MAX_WORKING_MEMORY_SIZE.
        strategy: (Optional) Optimization strategy: 'balanced', 'importance', 'recency', 'diversity'. Default 'balanced'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary detailing the optimization results.
        { ... same structure as before ... }

    Raises:
        ToolInputError: If context not found or strategy is invalid.
        ToolError: If the database operation fails.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if target_size < 0:
        raise ToolInputError("Target size cannot be negative.", param_name="target_size")
    valid_strategies = ["balanced", "importance", "recency", "diversity"]
    if strategy not in valid_strategies:
        raise ToolInputError(
            f"Strategy must be one of: {', '.join(valid_strategies)}", param_name="strategy"
        )
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # Get current context state using correct column name
            state_cursor = await conn.execute(
                "SELECT workflow_id, focal_memory_id, working_memory FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            )
            state_row = await state_cursor.fetchone()
            await state_cursor.close() # Close cursor
            if not state_row:
                raise ToolInputError(f"Context {context_id} not found.", param_name="context_id")

            workflow_id = state_row["workflow_id"]
            # Deserialize using the correct column name
            current_memory_ids = await MemoryUtils.deserialize(state_row["working_memory"]) or []

            before_count = len(current_memory_ids)
            if before_count <= target_size:
                logger.info(
                    f"Working memory for {context_id} already at or below target size ({before_count}/{target_size}). No optimization needed."
                )
                # Return block remains the same
                return {
                    "context_id": context_id,
                    "workflow_id": workflow_id,
                    "strategy_used": strategy,
                    "target_size": target_size,
                    "before_count": before_count,
                    "after_count": before_count,
                    "removed_count": 0,
                    "retained_memories": current_memory_ids,
                    "removed_memories": [],
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # Fetch details needed for scoring/diversity
            memories_to_consider = []
            if current_memory_ids:
                placeholders = ", ".join(["?"] * len(current_memory_ids))
                query = f"""
                SELECT memory_id, memory_type, importance, confidence, created_at, last_accessed, access_count
                FROM memories WHERE memory_id IN ({placeholders})
                """
                async with conn.execute(query, current_memory_ids) as mem_cursor: # Use async with
                    async for row in mem_cursor:
                        memories_to_consider.append(dict(row))
                # Cursor closed automatically by async with

            # Check if fetching details failed unexpectedly
            if not memories_to_consider and before_count > 0:
                logger.warning(
                    f"Working memory ID list for {context_id} was not empty ({before_count}), but no memory details found. Clearing working memory list."
                )
                await conn.execute(
                    "UPDATE cognitive_states SET working_memory = ?, last_active = ? WHERE state_id = ?",
                    ("[]", int(time.time()), context_id),
                )
                await conn.commit()

                return {
                    "context_id": context_id,
                    "workflow_id": workflow_id,
                    "strategy_used": strategy,
                    "target_size": target_size,
                    "before_count": before_count,
                    "after_count": 0,
                    "removed_count": before_count,
                    "retained_memories": [],
                    "removed_memories": current_memory_ids,
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # --- Scoring and Selection Logic (remains the same) ---
            scored_memories = []
            now = time.time()
            for memory in memories_to_consider:
                mem_id = memory["memory_id"]
                importance = memory["importance"]
                confidence = memory["confidence"]
                created_at = memory["created_at"]
                last_accessed = memory["last_accessed"]
                access_count = memory["access_count"] or 0 # Handle potential None
                mem_type = memory["memory_type"]

                relevance = _compute_memory_relevance(
                    importance, confidence, created_at, access_count, last_accessed
                )
                recency = 1.0 / (1.0 + (now - (last_accessed or created_at)) / 86400)

                score = 0.0
                if strategy == "balanced":
                    score = relevance
                elif strategy == "importance":
                    score = (importance * 0.7) + (confidence * 0.1) + (recency * 0.1) + (min(1.0, access_count / 5.0) * 0.1)
                elif strategy == "recency":
                    score = (recency * 0.6) + (importance * 0.2) + (min(1.0, access_count / 5.0) * 0.2)
                elif strategy == "diversity":
                    score = relevance # Base score for sorting within types

                scored_memories.append({"id": mem_id, "score": score, "type": mem_type})

            # Select memories to keep
            retained_memory_ids = []
            if strategy == "diversity":
                type_groups = defaultdict(list)
                for mem in scored_memories:
                    type_groups[mem["type"]].append(mem)
                for group in type_groups.values():
                    group.sort(key=lambda x: x["score"], reverse=True)
                group_iters = {mem_type: iter(group) for mem_type, group in type_groups.items()}
                active_groups = list(group_iters.keys())
                while len(retained_memory_ids) < target_size and active_groups:
                    group_type_to_select = active_groups.pop(0)
                    try:
                        selected_mem = next(group_iters[group_type_to_select])
                        retained_memory_ids.append(selected_mem["id"])
                        active_groups.append(group_type_to_select) # Add back for round-robin
                    except StopIteration:
                        pass # Group exhausted
                    if not active_groups and len(retained_memory_ids) < target_size:
                         break
            else:  # Balanced, Importance, Recency
                scored_memories.sort(key=lambda x: x["score"], reverse=True)
                retained_memory_ids = [m["id"] for m in scored_memories[:target_size]]
            # --- End Scoring and Selection Logic ---

            # Determine removed memories
            removed_memory_ids = list(set(current_memory_ids) - set(retained_memory_ids))
            after_count = len(retained_memory_ids)
            removed_count = len(removed_memory_ids)

            # Update cognitive state with the correct column name
            now_unix = int(time.time())
            await conn.execute(
                "UPDATE cognitive_states SET working_memory = ?, last_active = ? WHERE state_id = ?",
                (await MemoryUtils.serialize(retained_memory_ids), now_unix, context_id),
            )

            # Log operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "optimize_working_memory",
                None, None,
                {
                    "context_id": context_id, "strategy": strategy,
                    "target_size": target_size, "before_count": before_count,
                    "after_count": after_count, "removed_count": removed_count,
                },
            )

            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Optimized working memory for {context_id} using '{strategy}'. Retained: {after_count}, Removed: {removed_count}",
                emoji_key="recycle",
            )
            # Return block remains the same
            return {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "strategy_used": strategy,
                "target_size": target_size,
                "before_count": before_count,
                "after_count": after_count,
                "removed_count": removed_count,
                "retained_memories": retained_memory_ids,
                "removed_memories": removed_memory_ids,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error optimizing working memory for {context_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to optimize working memory: {str(e)}") from e
    

# --- 13. Cognitive State Persistence ---
@with_tool_metrics
@with_error_handling
async def save_cognitive_state(
    workflow_id: str,
    title: str,
    # Accept lists of IDs
    working_memory_ids: List[str],
    focus_area_ids: Optional[List[str]] = None,  # Assuming focus areas are primarily memory IDs now
    context_action_ids: Optional[List[str]] = None,
    current_goal_thought_ids: Optional[List[str]] = None,  # Assuming goals are thought IDs
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Saves the agent's current cognitive state as a checkpoint for a workflow.

    Creates a snapshot containing active memory IDs, focus memory IDs, relevant action IDs,
    and current goal thought IDs. Marks previous states for the workflow as not the latest.
    Validates that all provided IDs exist within the current workflow context.

    Args:
        workflow_id: The ID of the workflow.
        title: A descriptive title for this state (e.g., "Before attempting API integration").
        working_memory_ids: List of memory IDs currently in active working memory.
        focus_area_ids: (Optional) List of memory IDs representing the agent's current focus.
        context_action_ids: (Optional) List of recent/relevant action IDs providing context.
        current_goal_thought_ids: (Optional) List of thought IDs representing current goals.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary confirming the saved state.
        {
            "state_id": "state-uuid",
            "workflow_id": "workflow-uuid",
            "title": "State title",
            "created_at": "iso-timestamp",
            "success": true,
            "processing_time": 0.08
        }

    Raises:
        ToolInputError: If workflow not found, required parameters missing, or any provided IDs do not exist or belong to the workflow.
        ToolError: If the database operation fails.
    """
    if not title:
        raise ToolInputError("State title required.", param_name="title")

    state_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    start_time = time.time()

    # Combine all IDs for validation efficiently
    all_memory_ids = set(working_memory_ids + (focus_area_ids or []))
    all_action_ids = set(context_action_ids or [])
    all_thought_ids = set(current_goal_thought_ids or [])

    try:
        async with DBConnection(db_path) as conn:
            # --- Validation Step ---
            # 1. Check workflow exists
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()
            if not wf_exists:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # 2. Validate Memory IDs belong to this workflow
            if all_memory_ids:
                placeholders = ",".join("?" * len(all_memory_ids))
                query = f"SELECT memory_id FROM memories WHERE memory_id IN ({placeholders}) AND workflow_id = ?"
                params = list(all_memory_ids) + [workflow_id]
                cursor = await conn.execute(query, params)
                found_mem_ids = {row["memory_id"] for row in await cursor.fetchall()}
                await cursor.close()
                missing_mem_ids = all_memory_ids - found_mem_ids
                if missing_mem_ids:
                    raise ToolInputError(
                        f"Memory IDs not found or not in workflow {workflow_id}: {missing_mem_ids}",
                        param_name="working_memory_ids/focus_area_ids",
                    )

            # 3. Validate Action IDs belong to this workflow
            if all_action_ids:
                placeholders = ",".join("?" * len(all_action_ids))
                query = f"SELECT action_id FROM actions WHERE action_id IN ({placeholders}) AND workflow_id = ?"
                params = list(all_action_ids) + [workflow_id]
                cursor = await conn.execute(query, params)
                found_action_ids = {row["action_id"] for row in await cursor.fetchall()}
                await cursor.close()
                missing_action_ids = all_action_ids - found_action_ids
                if missing_action_ids:
                    raise ToolInputError(
                        f"Action IDs not found or not in workflow {workflow_id}: {missing_action_ids}",
                        param_name="context_action_ids",
                    )

            # 4. Validate Thought IDs belong to this workflow
            if all_thought_ids:
                placeholders = ",".join("?" * len(all_thought_ids))
                query = f"""
                    SELECT t.thought_id FROM thoughts t
                    JOIN thought_chains tc ON t.thought_chain_id = tc.thought_chain_id
                    WHERE t.thought_id IN ({placeholders}) AND tc.workflow_id = ?
                """
                params = list(all_thought_ids) + [workflow_id]
                cursor = await conn.execute(query, params)
                found_thought_ids = {row["thought_id"] for row in await cursor.fetchall()}
                await cursor.close()
                missing_thought_ids = all_thought_ids - found_thought_ids
                if missing_thought_ids:
                    raise ToolInputError(
                        f"Thought IDs not found or not in workflow {workflow_id}: {missing_thought_ids}",
                        param_name="current_goal_thought_ids",
                    )

            # --- Proceed with Saving State ---
            # Mark previous states as not latest
            await conn.execute(
                "UPDATE cognitive_states SET is_latest = 0 WHERE workflow_id = ?", (workflow_id,)
            )

            # Serialize state data (using the validated lists)
            working_mem_json = await MemoryUtils.serialize(working_memory_ids)
            focus_json = await MemoryUtils.serialize(focus_area_ids or [])
            context_actions_json = await MemoryUtils.serialize(context_action_ids or [])
            current_goals_json = await MemoryUtils.serialize(current_goal_thought_ids or [])

            # Insert new state
            await conn.execute(
                """
                INSERT INTO cognitive_states (state_id, workflow_id, title, working_memory, focus_areas,
                context_actions, current_goals, created_at, is_latest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state_id,
                    workflow_id,
                    title,
                    working_mem_json,
                    focus_json,
                    context_actions_json,
                    current_goals_json,
                    now_unix,
                    True,
                ),
            )

            # Update workflow timestamp
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now_unix, now_unix, workflow_id),
            ) 

            # Log operation
            log_data = {
                "state_id": state_id,
                "title": title,
                "working_memory_count": len(working_memory_ids),
                "focus_count": len(focus_area_ids or []),
                "action_context_count": len(context_action_ids or []),
                "goal_count": len(current_goal_thought_ids or []),
            }
            await MemoryUtils._log_memory_operation(
                conn, workflow_id, "save_state", None, None, log_data
            )

            await conn.commit()

            result = {
                "state_id": state_id,
                "workflow_id": workflow_id,
                "title": title,
                "created_at": to_iso_z(now_unix),
                "success": True,
                "processing_time": time.time() - start_time,
            }

            logger.info(
                f"Saved cognitive state '{title}' ({state_id}) for workflow {workflow_id}",
                emoji_key="save",
            )

            return result
    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error saving cognitive state: {e}", exc_info=True)
        raise ToolError(f"Failed to save cognitive state: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def load_cognitive_state(
    workflow_id: str,
    state_id: Optional[str] = None,  # If None, load latest
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Loads a previously saved cognitive state for a workflow.

    Restores context by retrieving a saved snapshot of working memory, focus, etc.

    Args:
        workflow_id: The ID of the workflow.
        state_id: (Optional) The ID of the specific state to load. If None, loads the latest state.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the loaded cognitive state data.
        {
            "state_id": "state-uuid",
            "workflow_id": "workflow-uuid",
            "title": "State title",
            "working_memory_ids": ["mem_id1", ...],
            "focus_areas": ["mem_id_focus", "topic description", ...],
            "context_action_ids": ["action_id1", ...],
            "current_goals": ["goal description", "goal_thought_id", ...],
            "created_at": "iso-timestamp",
            "success": true,
            "processing_time": 0.06
        }

    Raises:
        ToolInputError: If the workflow or specified state doesn't exist.
        ToolError: If the database operation fails.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # Check workflow exists
            cursor = await conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            wf_exists = await cursor.fetchone()
            await cursor.close()
            if not wf_exists:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            # Build query
            query = "SELECT * FROM cognitive_states WHERE workflow_id = ?"
            params: List[Any] = [workflow_id]
            if state_id:
                query += " AND state_id = ?"
                params.append(state_id)
            else:
                query += " ORDER BY created_at DESC, is_latest DESC LIMIT 1"  # Prefer is_latest if timestamps clash

            # Fetch state
            cursor = await conn.execute(query, params)
            row = await cursor.fetchone()
            await cursor.close()
            if not row:
                err_msg = (
                    f"State {state_id} not found."
                    if state_id
                    else f"No states found for workflow {workflow_id}."
                )
                raise ToolInputError(err_msg, param_name="state_id" if state_id else "workflow_id")

            state = dict(row)

            # Deserialize data and format timestamp
            created_at_unix = state["created_at"]  # Get the stored Unix timestamp
            result = {
                "state_id": state["state_id"],
                "workflow_id": state["workflow_id"],
                "title": state["title"],
                "working_memory_ids": await MemoryUtils.deserialize(state.get("working_memory"))
                or [],
                "focus_areas": await MemoryUtils.deserialize(state.get("focus_areas")) or [],
                "context_action_ids": await MemoryUtils.deserialize(state.get("context_actions"))
                or [],
                "current_goals": await MemoryUtils.deserialize(state.get("current_goals")) or [],
                "created_at": to_iso_z(created_at_unix),
                "success": True,
                "processing_time": time.time() - start_time,
            }

            # Log operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "load_state",
                None,
                None,
                {"state_id": state["state_id"], "title": state["title"]},
            )

            logger.info(
                f"Loaded cognitive state '{result['title']}' ({result['state_id']}) for workflow {workflow_id}",
                emoji_key="inbox_tray",
            )
            return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error loading cognitive state: {e}", exc_info=True)
        raise ToolError(f"Failed to load cognitive state: {str(e)}") from e


# --- 14. Comprehensive Context Retrieval (Ported from agent_memory) ---
@with_tool_metrics
@with_error_handling
async def get_workflow_context(
    workflow_id: str,
    recent_actions_limit: int = 10,  # Reduced default
    important_memories_limit: int = 5,
    key_thoughts_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves a comprehensive context summary for a workflow.

    Combines the latest cognitive state, recent actions, important memories, and key thoughts
    to provide a snapshot for resuming work or understanding the current situation.

    Args:
        workflow_id: The ID of the workflow.
        recent_actions_limit: (Optional) Max number of recent actions to include. Default 10.
        important_memories_limit: (Optional) Max number of important memories to include. Default 5.
        key_thoughts_limit: (Optional) Max number of key thoughts to include. Default 5.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the combined workflow context.
        {
            "workflow_id": "workflow-uuid",
            "workflow_title": "Workflow Title",
            "workflow_status": "active",
            "workflow_goal": "Goal description",
            "latest_cognitive_state": { ... } | None,
            "recent_actions": [ { ...action_summary... }, ... ],
            "important_memories": [ { ...memory_summary... }, ... ],
            "key_thoughts": [ { ...thought_summary... }, ... ],
            "success": true,
            "processing_time": 0.35
        }

    Raises:
        ToolInputError: If the workflow doesn't exist.
        ToolError: If the database operation fails.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # 1. Get basic workflow info
            async with conn.execute(
                "SELECT title, goal, status FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                wf_row = await cursor.fetchone()
                if not wf_row:
                    raise ToolInputError(
                        f"Workflow {workflow_id} not found.", param_name="workflow_id"
                    )
                context = {
                    "workflow_id": workflow_id,
                    "workflow_title": wf_row["title"],
                    "workflow_goal": wf_row["goal"],
                    "workflow_status": wf_row["status"],
                }

            # 2. Get Latest Cognitive State (using load_cognitive_state logic)
            try:
                # Reusing the logic by calling the tool function
                context["latest_cognitive_state"] = await load_cognitive_state(
                    workflow_id=workflow_id, state_id=None, db_path=db_path
                )
                # Remove success/timing info from nested call result for cleaner context
                context["latest_cognitive_state"].pop("success", None)
                context["latest_cognitive_state"].pop("processing_time", None)
            except ToolInputError:  # Catch if no state exists
                context["latest_cognitive_state"] = None
                logger.info(
                    f"No cognitive state found for workflow {workflow_id} during context retrieval."
                )
            except Exception as e:  # Catch other errors loading state
                logger.warning(
                    f"Could not load cognitive state for workflow {workflow_id} context: {e}"
                )
                context["latest_cognitive_state"] = {"error": f"Failed to load state: {e}"}

            # 3. Get Recent Actions (using get_recent_actions logic)
            try:
                actions_result = await get_recent_actions(
                    workflow_id=workflow_id,
                    limit=recent_actions_limit,
                    include_reasoning=False,
                    include_tool_results=False,  # Keep context concise
                    db_path=db_path,
                )
                context["recent_actions"] = actions_result.get("actions", [])
            except Exception as e:
                logger.warning(
                    f"Could not load recent actions for workflow {workflow_id} context: {e}"
                )
                context["recent_actions"] = [{"error": f"Failed to load actions: {e}"}]

            # 4. Get Important Memories (using query_memories logic)
            try:
                memories_result = await query_memories(
                    workflow_id=workflow_id,
                    limit=important_memories_limit,
                    sort_by="importance",
                    sort_order="DESC",
                    include_content=False,  # Exclude full content for context
                    db_path=db_path,
                )
                # Extract relevant fields for context
                context["important_memories"] = [
                    {
                        "memory_id": m["memory_id"],
                        "description": m.get("description"),
                        "memory_type": m.get("memory_type"),
                        "importance": m.get("importance"),
                    }
                    for m in memories_result.get("memories", [])
                ]
            except Exception as e:
                logger.warning(
                    f"Could not load important memories for workflow {workflow_id} context: {e}"
                )
                context["important_memories"] = [{"error": f"Failed to load memories: {e}"}]

            # 5. Get Key Thoughts (e.g., latest goals, decisions, summaries from main chain)
            try:
                # Find main thought chain
                async with conn.execute(
                    "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                    (workflow_id,),
                ) as cursor:
                    chain_row = await cursor.fetchone()
                    if chain_row:
                        thought_chain_id = chain_row["thought_chain_id"]
                        # Fetch important thought types, most recent first
                        async with conn.execute(
                            """SELECT thought_type, content, sequence_number, created_at
                                 FROM thoughts WHERE thought_chain_id = ?
                                 AND thought_type IN (?, ?, ?, ?)
                                 ORDER BY sequence_number DESC LIMIT ?""",
                            (
                                thought_chain_id,
                                ThoughtType.GOAL.value,
                                ThoughtType.DECISION.value,
                                ThoughtType.SUMMARY.value,
                                ThoughtType.REFLECTION.value,
                                key_thoughts_limit,
                            ),
                        ) as thought_cursor:
                            context["key_thoughts"] = [
                                dict(row) for row in await thought_cursor.fetchall()
                            ]
                    else:
                        context["key_thoughts"] = []
            except Exception as e:
                logger.warning(
                    f"Could not load key thoughts for workflow {workflow_id} context: {e}"
                )
                context["key_thoughts"] = [{"error": f"Failed to load thoughts: {e}"}]

            context["success"] = True
            context["processing_time"] = time.time() - start_time
            logger.info(
                f"Retrieved context summary for workflow {workflow_id}", emoji_key="compass"
            )
            return context

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow context for {workflow_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to get workflow context: {str(e)}") from e


# --- Helper: Scoring for Focus ---
def _calculate_focus_score(memory: Dict, recent_action_ids: List[str], now_unix: int) -> float:
    """Calculates a score for prioritizing focus, based on memory attributes."""
    score = 0.0

    # Base relevance score (importance, confidence, recency, usage)
    relevance = _compute_memory_relevance(
        memory.get("importance", 5.0),
        memory.get("confidence", 1.0),
        memory.get("created_at", now_unix),
        memory.get("access_count", 0),
        memory.get("last_accessed", None),
    )
    score += relevance * 0.6  # Base relevance is weighted heavily

    # Boost for being linked to recent actions
    if memory.get("action_id") and memory["action_id"] in recent_action_ids:
        score += 3.0  # Significant boost if directly related to recent work

    # Boost for certain types often indicating current context
    if memory.get("memory_type") in [
        MemoryType.QUESTION.value,
        MemoryType.PLAN.value,
        MemoryType.INSIGHT.value,
    ]:
        score += 1.5

    # Slight boost for higher memory levels (semantic/procedural over episodic)
    if memory.get("memory_level") == MemoryLevel.SEMANTIC.value:
        score += 0.5
    elif memory.get("memory_level") == MemoryLevel.PROCEDURAL.value:
        score += 0.7

    # Ensure score is not negative
    return max(0.0, score)


# --- Tool: Auto Update Focus ---
@with_tool_metrics
@with_error_handling
async def auto_update_focus(
    context_id: str,
    recent_actions_count: int = 3,  # How many recent actions to consider influential
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Automatically updates the focal memory for a context based on relevance and recent activity.

    Analyzes memories currently in the working set for the given context. It scores them based
    on importance, confidence, recency, usage, type, and linkage to recent actions.
    The memory with the highest score becomes the new focal memory.

    Args:
        context_id: The context identifier.
        recent_actions_count: (Optional) Number of most recent actions to consider for boosting relevance. Default 3.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary indicating the result of the focus update.
        {
            "context_id": "context-uuid",
            "workflow_id": "workflow-uuid",
            "previous_focal_memory_id": "old-focus-uuid" | None,
            "new_focal_memory_id": "new-focus-uuid" | None,
            "focus_changed": true | false,
            "reason": "Highest score based on relevance and recent activity." | "No suitable memory found." | "Focus unchanged.",
            "success": true,
            "processing_time": 0.1
        }

    Raises:
        ToolInputError: If context not found.
        ToolError: If the database operation fails.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if recent_actions_count < 0:
        raise ToolInputError(
            "Recent actions count cannot be negative.", param_name="recent_actions_count"
        )
    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Get Current Context & Working Memory ---
            async with conn.execute(
                "SELECT workflow_id, focal_memory_id, working_memory FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            ) as cursor:
                state_row = await cursor.fetchone()
                if not state_row:
                    raise ToolInputError(
                        f"Context {context_id} not found.", param_name="context_id"
                    )
                workflow_id = state_row["workflow_id"]
                previous_focal_id = state_row["focal_memory_id"]
                current_memory_ids = await MemoryUtils.deserialize(state_row["working_memory"]) or []

            if not current_memory_ids:
                logger.info(
                    f"Working memory for context {context_id} is empty. Cannot determine focus."
                )
                return {
                    "context_id": context_id,
                    "workflow_id": workflow_id,
                    "previous_focal_memory_id": previous_focal_id,
                    "new_focal_memory_id": None,
                    "focus_changed": previous_focal_id is not None,
                    "reason": "Working memory is empty.",
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # --- 2. Get Details for Working Memories ---
            working_memories_details = []
            placeholders = ", ".join(["?"] * len(current_memory_ids))
            query = f"""
                SELECT memory_id, action_id, memory_type, memory_level, importance, confidence,
                       created_at, last_accessed, access_count
                FROM memories WHERE memory_id IN ({placeholders})
            """
            async with conn.execute(query, current_memory_ids) as cursor:
                working_memories_details = [dict(row) for row in await cursor.fetchall()]

            # --- 3. Get Recent Action IDs ---
            recent_action_ids = []
            if recent_actions_count > 0:
                async with conn.execute(
                    "SELECT action_id FROM actions WHERE workflow_id = ? ORDER BY sequence_number DESC LIMIT ?",
                    (workflow_id, recent_actions_count),
                ) as cursor:
                    recent_action_ids = [row["action_id"] for row in await cursor.fetchall()]

            # --- 4. Score Memories & Find Best Candidate ---
            now_unix = int(time.time())
            best_candidate_id = None
            highest_score = -1.0

            for memory in working_memories_details:
                score = _calculate_focus_score(memory, recent_action_ids, now_unix)
                # logger.debug(f"Memory {memory['memory_id'][:8]} focus score: {score:.2f}") # Debug logging
                if score > highest_score:
                    highest_score = score
                    best_candidate_id = memory["memory_id"]

            # --- 5. Update Focus if Changed ---
            focus_changed = False
            reason = "Focus unchanged."
            if best_candidate_id and best_candidate_id != previous_focal_id:
                await conn.execute(
                    "UPDATE cognitive_states SET focal_memory_id = ?, last_active = ? WHERE state_id = ?",
                    (best_candidate_id, now_unix, context_id),
                )
                focus_changed = True
                reason = f"Memory {best_candidate_id[:8]}... has highest score ({highest_score:.2f}) based on relevance and recent activity."
                logger.info(
                    f"Auto-shifting focus for context {context_id} to memory {best_candidate_id}. Previous: {previous_focal_id}",
                    emoji_key="compass",
                )
                # Log the operation
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "auto_focus_shift",
                    best_candidate_id,
                    None,
                    {
                        "context_id": context_id,
                        "previous_focus": previous_focal_id,
                        "score": highest_score,
                    },
                )
                await conn.commit()  # Commit the change
            elif not best_candidate_id:
                reason = "No suitable memory found in working set to focus on."
                logger.info(
                    f"Auto-focus update for context {context_id}: No suitable candidate found."
                )
            else:
                logger.info(
                    f"Auto-focus update for context {context_id}: Focus remains on {previous_focal_id}."
                )

            processing_time = time.time() - start_time
            return {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "previous_focal_memory_id": previous_focal_id,
                "new_focal_memory_id": best_candidate_id,
                "focus_changed": focus_changed,
                "reason": reason,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error auto-updating focus for context {context_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to auto-update focus: {str(e)}") from e


# --- Tool: Promote Memory Level ---
@with_tool_metrics
@with_error_handling
async def promote_memory_level(
    memory_id: str,
    target_level: Optional[str] = None,  # Explicit target level (e.g., 'semantic')
    min_access_count_episodic: int = 5,  # Configurable thresholds
    min_confidence_episodic: float = 0.8,
    min_access_count_semantic: int = 10,
    min_confidence_semantic: float = 0.9,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Attempts to promote a memory to a higher cognitive level based on usage and confidence.

    Checks if a memory meets heuristic criteria for promotion (e.g., high access count,
    high confidence) to the next logical level (Episodic -> Semantic, Semantic -> Procedural).
    If criteria are met, the memory's level is updated.

    Args:
        memory_id: The ID of the memory to potentially promote.
        target_level: (Optional) Explicitly specify the level to promote TO. If provided,
                      checks if the memory meets criteria for *that specific level*.
                      If None, checks for promotion to the *next logical* level.
        min_access_count_episodic: (Optional) Min access count to promote Episodic->Semantic. Default 5.
        min_confidence_episodic: (Optional) Min confidence to promote Episodic->Semantic. Default 0.8.
        min_access_count_semantic: (Optional) Min access count to promote Semantic->Procedural. Default 10.
        min_confidence_semantic: (Optional) Min confidence to promote Semantic->Procedural. Default 0.9.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary indicating if promotion occurred and the result.
        {
            "memory_id": "memory-uuid",
            "promoted": true | false,
            "previous_level": "episodic",
            "new_level": "semantic" | None, # None if not promoted
            "reason": "Met criteria: access_count >= 5, confidence >= 0.8" | "Criteria not met." | "Already at highest/target level.",
            "success": true,
            "processing_time": 0.07
        }

    Raises:
        ToolInputError: If memory not found or target level is invalid.
        ToolError: If the database operation fails.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    target_level_enum = None
    if target_level:
        try:
            target_level_enum = MemoryLevel(target_level.lower())
        except ValueError as e:
            valid_levels = [ml.value for ml in MemoryLevel]
            raise ToolInputError(
                f"Invalid target_level. Use one of: {valid_levels}", param_name="target_level"
            ) from e

    start_time = time.time()

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Get Current Memory Details ---
            async with conn.execute(
                "SELECT workflow_id, memory_level, memory_type, access_count, confidence, importance FROM memories WHERE memory_id = ?",
                (memory_id,),
            ) as cursor:
                mem_row = await cursor.fetchone()
                if not mem_row:
                    raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

                current_level = MemoryLevel(mem_row["memory_level"])
                mem_type = MemoryType(mem_row["memory_type"])  # Get type for procedural check
                access_count = mem_row["access_count"] or 0
                confidence = mem_row["confidence"] or 0.0
                workflow_id = mem_row["workflow_id"]  # Needed for logging

            # --- 2. Determine Target Level and Check Criteria ---
            promoted = False
            new_level_enum = None
            reason = "Criteria not met or already at highest/target level."

            # Determine the *next* logical level if no explicit target is given
            potential_next_level = None
            if current_level == MemoryLevel.EPISODIC:
                potential_next_level = MemoryLevel.SEMANTIC
            elif current_level == MemoryLevel.SEMANTIC and mem_type in [
                MemoryType.PROCEDURE,
                MemoryType.SKILL,
            ]:
                # Only allow promotion to procedural if type is appropriate
                potential_next_level = MemoryLevel.PROCEDURAL

            # Use explicit target level if provided, otherwise use the potential next level
            level_to_check_for = target_level_enum or potential_next_level

            # Check if promotion is possible and criteria are met
            if (
                level_to_check_for and level_to_check_for.value > current_level.value
            ):  # Ensure target is actually higher
                criteria_met = False
                criteria_desc = ""

                if (
                    level_to_check_for == MemoryLevel.SEMANTIC
                    and current_level == MemoryLevel.EPISODIC
                ):
                    criteria_met = (
                        access_count >= min_access_count_episodic
                        and confidence >= min_confidence_episodic
                    )
                    criteria_desc = f"Met criteria for Semantic: access_count >= {min_access_count_episodic} ({access_count}), confidence >= {min_confidence_episodic} ({confidence:.2f})"
                elif (
                    level_to_check_for == MemoryLevel.PROCEDURAL
                    and current_level == MemoryLevel.SEMANTIC
                ):
                    # Add stricter checks for procedural: must be procedure/skill type AND meet usage/confidence
                    if mem_type in [MemoryType.PROCEDURE, MemoryType.SKILL]:
                        criteria_met = (
                            access_count >= min_access_count_semantic
                            and confidence >= min_confidence_semantic
                        )
                        criteria_desc = f"Met criteria for Procedural: type is '{mem_type.value}', access_count >= {min_access_count_semantic} ({access_count}), confidence >= {min_confidence_semantic} ({confidence:.2f})"
                    else:
                        criteria_desc = f"Criteria not met for Procedural: memory type '{mem_type.value}' is not procedure/skill."

                if criteria_met:
                    promoted = True
                    new_level_enum = level_to_check_for
                    reason = criteria_desc
                else:
                    reason = (
                        criteria_desc
                        if criteria_desc
                        else f"Criteria not met for promotion to {level_to_check_for.value}."
                    )

            elif level_to_check_for and level_to_check_for.value <= current_level.value:
                reason = (
                    f"Memory is already at or above the target level '{level_to_check_for.value}'."
                )
            elif not level_to_check_for:
                reason = f"Memory is already at the highest promotable level ({current_level.value}) or not eligible for promotion (type: {mem_type.value})."

            # --- 3. Update Memory if Promoted ---
            if promoted and new_level_enum:
                now_unix = int(time.time())
                await conn.execute(
                    "UPDATE memories SET memory_level = ?, updated_at = ? WHERE memory_id = ?",
                    (new_level_enum.value, now_unix, memory_id),
                )
                # Log the promotion
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "promote_level",
                    memory_id,
                    None,
                    {
                        "previous_level": current_level.value,
                        "new_level": new_level_enum.value,
                        "reason": reason,
                    },
                )
                await conn.commit()
                logger.info(
                    f"Promoted memory {memory_id} from {current_level.value} to {new_level_enum.value}",
                    emoji_key="arrow_up",
                )
            else:
                logger.info(f"Memory {memory_id} not promoted. Reason: {reason}")

            # --- 4. Return Result ---
            processing_time = time.time() - start_time
            return {
                "memory_id": memory_id,
                "promoted": promoted,
                "previous_level": current_level.value,
                "new_level": new_level_enum.value if promoted else None,
                "reason": reason,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error promoting memory {memory_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to promote memory: {str(e)}") from e


# --- 15. Memory Update ---
@with_tool_metrics
@with_error_handling
async def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    description: Optional[str] = None,
    reasoning: Optional[str] = None,
    tags: Optional[List[str]] = None,  # Replaces existing tags if provided
    ttl: Optional[int] = None,
    memory_level: Optional[str] = None,
    regenerate_embedding: bool = False,  # Explicit flag to trigger re-embedding
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Updates fields of an existing memory entry.

    Allows modification of content, importance, confidence, tags, etc.
    If content or description is updated and regenerate_embedding=True,
    the embedding will be recalculated.

    Args:
        memory_id: ID of the memory to update.
        content: (Optional) New content for the memory.
        importance: (Optional) New importance score (1.0-10.0).
        confidence: (Optional) New confidence score (0.0-1.0).
        description: (Optional) New description.
        reasoning: (Optional) New reasoning.
        tags: (Optional) List of new tags (replaces existing tags).
        ttl: (Optional) New time-to-live in seconds (0 for permanent).
        memory_level: (Optional) New memory level.
        regenerate_embedding: (Optional) Force regeneration of embedding if content or description changes. Default False.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the updated memory details (excluding full content unless changed).
        {
            "memory_id": "memory-uuid",
            "updated_fields": ["importance", "tags"],
            "embedding_regenerated": true | false,
            "updated_at_unix": 1678886400,
            "success": true,
            "processing_time": 0.15
        }

    Raises:
        ToolInputError: If memory not found or parameters are invalid.
        ToolError: If the database operation fails.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    start_time = time.time()

    # Parameter validations
    if importance is not None and not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0-10.0.", param_name="importance")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ToolInputError("Confidence must be 0.0-1.0.", param_name="confidence")
    final_tags_json = None
    if tags is not None:
        final_tags_json = json.dumps(list(set(str(t).lower() for t in tags)))
    if memory_level:
        try:
            MemoryLevel(memory_level.lower())
        except ValueError as e:
            raise ToolInputError("Invalid memory_level.", param_name="memory_level") from e

    update_clauses = []
    params = []
    updated_fields = []

    # Build dynamic SET clause
    if content is not None:
        update_clauses.append("content = ?")
        params.append(content)
        updated_fields.append("content")
    if importance is not None:
        update_clauses.append("importance = ?")
        params.append(importance)
        updated_fields.append("importance")
    if confidence is not None:
        update_clauses.append("confidence = ?")
        params.append(confidence)
        updated_fields.append("confidence")
    if description is not None:
        update_clauses.append("description = ?")
        params.append(description)
        updated_fields.append("description")
    if reasoning is not None:
        update_clauses.append("reasoning = ?")
        params.append(reasoning)
        updated_fields.append("reasoning")
    if final_tags_json is not None:
        update_clauses.append("tags = ?")
        params.append(final_tags_json)
        updated_fields.append("tags")
    if ttl is not None:
        update_clauses.append("ttl = ?")
        params.append(ttl)
        updated_fields.append("ttl")
    if memory_level:
        update_clauses.append("memory_level = ?")
        params.append(memory_level.lower())
        updated_fields.append("memory_level")

    if not update_clauses and not regenerate_embedding:
        raise ToolInputError("No fields provided for update and regenerate_embedding is False.", param_name="content")

    # Always update timestamp if other fields are changing
    embedding_regenerated = False # Initialize here
    new_embedding_db_id = None # Initialize here
    if update_clauses:
        now_unix = int(time.time())
        update_clauses.append("updated_at = ?")
        params.append(now_unix)
    else:
        # If only regenerating embedding, we don't need to update the timestamp via SQL
        now_unix = int(time.time()) # Still need timestamp for potential embedding
    params.append(memory_id)  # For the WHERE clause

    try:
        async with DBConnection(db_path) as conn:
            # Check memory exists and get current description/content if needed for embedding
            current_desc = None
            current_content = None
            needs_embedding_check = regenerate_embedding

            async with conn.execute(
                "SELECT workflow_id, description, content FROM memories WHERE memory_id = ?",
                (memory_id,),
            ) as cursor:
                mem_row = await cursor.fetchone()
                if not mem_row:
                    raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
                workflow_id = mem_row["workflow_id"]
                # Always fetch current content/desc if checking embedding, even if not updating them via SQL
                if needs_embedding_check:
                    current_desc = mem_row["description"]
                    current_content = mem_row["content"]

            if update_clauses:
                update_sql = f"UPDATE memories SET {', '.join(update_clauses)} WHERE memory_id = ?"
                await conn.execute(update_sql, params)

            # Regenerate embedding if requested
            if needs_embedding_check:
                # Determine the text to embed based on potential new values OR current values
                new_desc = description if "description" in updated_fields else current_desc
                new_content = content if "content" in updated_fields else current_content
                text_for_embedding = f"{new_desc}: {new_content}" if new_desc else new_content
                try:
                    new_embedding_db_id = await _store_embedding(
                        conn, memory_id, text_for_embedding
                    )
                    if new_embedding_db_id:
                        embedding_regenerated = True
                        logger.info(
                            f"Regenerated embedding for updated memory {memory_id}",
                            emoji_key="brain",
                        )
                    else:
                        logger.warning(f"Embedding regeneration failed for memory {memory_id}")
                except Exception as embed_err:
                    logger.error(
                        f"Error during embedding regeneration for memory {memory_id}: {embed_err}",
                        exc_info=True,
                    )

            # Log operation
            log_data = {"updated_fields": updated_fields} # updated_fields is still correct
            if embedding_regenerated:
                log_data["embedding_regenerated"] = True
            await MemoryUtils._log_memory_operation(
                conn, workflow_id, "update", memory_id, None, log_data
            )

            await conn.commit()

            if not update_clauses and embedding_regenerated:
                # If only embedding changed, the DB wasn't updated, use the time embedding happened
                # This assumes _store_embedding happens around now_unix
                pass # now_unix is already defined
            elif not update_clauses and not embedding_regenerated:
                # No change occurred, maybe fetch the existing updated_at? Or return None?
                # Let's return the time of the attempt.
                pass # now_unix is already defined
            # If update_clauses existed, now_unix was set correctly earlier.

            processing_time = time.time() - start_time
            logger.info(
                f"Updated memory {memory_id}. Fields: {', '.join(updated_fields) or 'None (Embedding only)'}.",
                emoji_key="pencil2",
            )
            return {
                "memory_id": memory_id,
                "updated_fields": updated_fields,
                "embedding_regenerated": embedding_regenerated,
                "updated_at_unix": now_unix, # Use the timestamp reflecting when the operation occurred
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error updating memory {memory_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to update memory: {str(e)}") from e


# ======================================================
# Linked Memories Retrieval
# ======================================================


@with_tool_metrics
@with_error_handling
async def get_linked_memories(
    memory_id: str,
    direction: str = "both",  # "outgoing", "incoming", or "both"
    link_type: Optional[str] = None,  # Optional filter by link type
    limit: int = 10,
    include_memory_details: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves memories linked to/from the specified memory.

    Fetches all memories linked to or from a given memory, optionally filtered by link type.
    Can include basic or detailed information about the linked memories.

    Args:
        memory_id: ID of the memory to get links for
        direction: Which links to include - "outgoing" (memory_id is the source),
                  "incoming" (memory_id is the target), or "both" (default)
        link_type: Optional filter for specific link types (e.g., "related", "supports")
        limit: Maximum number of links to return per direction (default 10)
        include_memory_details: Whether to include full details of the linked memories
        db_path: Path to the SQLite database file

    Returns:
        Dictionary containing the linked memories organized by direction:
        {
            "memory_id": "memory-uuid",
            "links": {
                "outgoing": [
                    {
                        "link_id": "link-uuid",
                        "source_memory_id": "memory-uuid",
                        "target_memory_id": "linked-memory-uuid",
                        "link_type": "related",
                        "strength": 0.85,
                        "description": "Auto-link based on similarity",
                        "created_at_unix": 1678886400,
                        "target_memory": { # Only if include_memory_details=True
                            "memory_id": "linked-memory-uuid",
                            "description": "Memory description",
                            "memory_type": "observation",
                            ... other memory fields ...
                        }
                    },
                    ... more outgoing links ...
                ],
                "incoming": [
                    {
                        "link_id": "link-uuid",
                        "source_memory_id": "other-memory-uuid",
                        "target_memory_id": "memory-uuid",
                        "link_type": "supports",
                        "strength": 0.7,
                        "description": "Supporting evidence",
                        "created_at_unix": 1678885400,
                        "source_memory": { # Only if include_memory_details=True
                            "memory_id": "other-memory-uuid",
                            "description": "Memory description",
                            "memory_type": "fact",
                            ... other memory fields ...
                        }
                    },
                    ... more incoming links ...
                ]
            },
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If memory_id is not provided or direction is invalid
        ToolError: If database operation fails
    """
    start_time = time.time()

    if not memory_id:
        raise ToolInputError("Memory ID is required", param_name="memory_id")

    valid_directions = ["outgoing", "incoming", "both"]
    direction = direction.lower()
    if direction not in valid_directions:
        raise ToolInputError(
            f"Direction must be one of: {', '.join(valid_directions)}", param_name="direction"
        )

    if link_type:
        try:
            LinkType(link_type.lower())  # Validate enum
        except ValueError as e:
            valid_types = [lt.value for lt in LinkType]
            raise ToolInputError(
                f"Invalid link_type. Must be one of: {', '.join(valid_types)}",
                param_name="link_type",
            ) from e

    # Initialize result structure
    result = {
        "memory_id": memory_id,
        "links": {"outgoing": [], "incoming": []},
        "success": True,
        "processing_time": 0.0,
    }

    try:
        async with DBConnection(db_path) as conn:
            # Check if memory exists
            async with conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(f"Memory {memory_id} not found", param_name="memory_id")

            # Process outgoing links (memory_id is the source)
            if direction in ["outgoing", "both"]:
                outgoing_query = """
                    SELECT ml.*, m.memory_type AS target_type, m.description AS target_description
                    FROM memory_links ml
                    JOIN memories m ON ml.target_memory_id = m.memory_id
                    WHERE ml.source_memory_id = ?
                """
                params = [memory_id]

                if link_type:
                    outgoing_query += " AND ml.link_type = ?"
                    params.append(link_type.lower())

                outgoing_query += " ORDER BY ml.created_at DESC LIMIT ?"
                params.append(limit)

                async with conn.execute(outgoing_query, params) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        link_data = dict(row)

                        # Add target memory details if requested
                        if include_memory_details:
                            target_memory_id = link_data["target_memory_id"]
                            async with conn.execute(
                                """
                                SELECT memory_id, memory_level, memory_type, importance, confidence, 
                                       description, created_at, updated_at, tags
                                FROM memories WHERE memory_id = ?
                                """,
                                (target_memory_id,),
                            ) as mem_cursor:
                                target_memory = await mem_cursor.fetchone()
                                if target_memory:
                                    mem_dict = dict(target_memory)
                                    # Process fields
                                    mem_dict["created_at_unix"] = mem_dict["created_at"]
                                    mem_dict["updated_at_unix"] = mem_dict["updated_at"]
                                    mem_dict["tags"] = await MemoryUtils.deserialize(
                                        mem_dict.get("tags")
                                    )
                                    link_data["target_memory"] = mem_dict

                        # Format link data
                        link_data["created_at_unix"] = link_data["created_at"]
                        result["links"]["outgoing"].append(link_data)

            # Process incoming links (memory_id is the target)
            if direction in ["incoming", "both"]:
                incoming_query = """
                    SELECT ml.*, m.memory_type AS source_type, m.description AS source_description
                    FROM memory_links ml
                    JOIN memories m ON ml.source_memory_id = m.memory_id
                    WHERE ml.target_memory_id = ?
                """
                params = [memory_id]

                if link_type:
                    incoming_query += " AND ml.link_type = ?"
                    params.append(link_type.lower())

                incoming_query += " ORDER BY ml.created_at DESC LIMIT ?"
                params.append(limit)

                async with conn.execute(incoming_query, params) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        link_data = dict(row)

                        # Add source memory details if requested
                        if include_memory_details:
                            source_memory_id = link_data["source_memory_id"]
                            async with conn.execute(
                                """
                                SELECT memory_id, memory_level, memory_type, importance, confidence, 
                                       description, created_at, updated_at, tags
                                FROM memories WHERE memory_id = ?
                                """,
                                (source_memory_id,),
                            ) as mem_cursor:
                                source_memory = await mem_cursor.fetchone()
                                if source_memory:
                                    mem_dict = dict(source_memory)
                                    # Process fields
                                    mem_dict["created_at_unix"] = mem_dict["created_at"]
                                    mem_dict["updated_at_unix"] = mem_dict["updated_at"]
                                    mem_dict["tags"] = await MemoryUtils.deserialize(
                                        mem_dict.get("tags")
                                    )
                                    link_data["source_memory"] = mem_dict

                        # Format link data
                        link_data["created_at_unix"] = link_data["created_at"]
                        result["links"]["incoming"].append(link_data)

            # Record access stats for the source memory
            await MemoryUtils._update_memory_access(conn, memory_id)
            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(result['links']['outgoing'])} outgoing and {len(result['links']['incoming'])} incoming links for memory {memory_id}",
                emoji_key="link",
            )

            result["processing_time"] = processing_time
            return result

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving linked memories: {e}", exc_info=True)
        raise ToolError(f"Failed to retrieve linked memories: {str(e)}") from e


# ======================================================
# Meta-Cognition Tools (Adapted from cognitive_memory)
# ======================================================


# --- Helper: Generate Consolidation Prompt (FULL INSTRUCTIONS) ---
def _generate_consolidation_prompt(memories: List[Dict], consolidation_type: str) -> str:
    """Generates a prompt for memory consolidation based on the type, with full instructions."""
    # Format memories as text (Limit input memories and content length for prompt size)
    memory_texts = []
    # Limit source memories included in prompt to avoid excessive length
    for i, memory in enumerate(memories[:20], 1):
        desc = memory.get("description") or ""
        # Limit content preview significantly to avoid overly long prompts
        content_preview = (memory.get("content", "") or "")[:300]
        mem_type = memory.get("memory_type", "N/A")
        importance = memory.get("importance", 5.0)
        confidence = memory.get("confidence", 1.0)
        created_ts = memory.get("created_at", 0)
        created_dt_str = (
            datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")
            if created_ts
            else "Unknown Date"
        )
        mem_id_short = memory.get("memory_id", "UNKNOWN")[:8]

        formatted = f"--- MEMORY #{i} (ID: {mem_id_short}..., Type: {mem_type}, Importance: {importance:.1f}, Confidence: {confidence:.1f}, Date: {created_dt_str}) ---\n"
        if desc:
            formatted += f"Description: {desc}\n"
        formatted += f"Content Preview: {content_preview}"
        # Indicate truncation
        if len(memory.get("content", "")) > 300:
            formatted += "...\n"
        else:
            formatted += "\n"
        memory_texts.append(formatted)

    memories_str = "\n".join(memory_texts)

    # Base prompt template
    base_prompt = f"""You are an advanced cognitive system processing and consolidating memories for an AI agent. Below are {len(memories)} memory items containing information, observations, and insights relevant to a task. Your goal is to perform a specific type of consolidation: '{consolidation_type}'.

Analyze the following memories carefully:

{memories_str}
--- END OF MEMORIES ---

"""

    # Add specific instructions based on consolidation type (FULL INSTRUCTIONS)
    if consolidation_type == "summary":
        base_prompt += """TASK: Create a comprehensive and coherent summary that synthesizes the key information and context from ALL the provided memories. Your summary should:
1.  Distill the most critical facts, findings, and core ideas presented across the memories.
2.  Organize the information logically, perhaps chronologically or thematically, creating a clear narrative flow.
3.  Highlight significant connections, relationships, or developments revealed by considering the memories together.
4.  Eliminate redundancy while preserving essential details and nuances.
5.  Be objective and accurately reflect the content of the source memories.
6.  Be well-structured and easy to understand for someone reviewing the workflow's progress.

Generate ONLY the summary content based on the provided memories.

CONSOLIDATED SUMMARY:"""

    elif consolidation_type == "insight":
        base_prompt += """TASK: Generate high-level insights by identifying significant patterns, implications, conclusions, or discrepancies emerging from the provided memories. Your insights should:
1.  Go beyond simple summarization to reveal non-obvious patterns, trends, or relationships connecting different memories.
2.  Draw meaningful conclusions or formulate hypotheses that are supported by the collective information but may not be explicit in any single memory.
3.  Explicitly highlight any contradictions, tensions, or unresolved issues found between memories.
4.  Identify the broader significance, potential impact, or actionable implications of the combined information.
5.  Be stated clearly and concisely, using cautious language where certainty is limited (e.g., "It appears that...", "This might suggest...").
6.  Focus on the most impactful and novel understandings gained from analyzing these memories together.

Generate ONLY the list of insights based on the provided memories.

CONSOLIDATED INSIGHTS:"""

    elif consolidation_type == "procedural":
        base_prompt += """TASK: Formulate a generalized procedure, method, or set of steps based on the actions, outcomes, and observations described in the memories. Your procedure should:
1.  Identify recurring sequences of actions or steps that appear to lead to successful or notable outcomes.
2.  Generalize from specific instances described in the memories to create a potentially reusable approach or workflow pattern.
3.  Clearly outline the sequence of steps involved in the procedure.
4.  Note important conditions, prerequisites, inputs, outputs, or constraints associated with the procedure or its steps.
5.  Highlight decision points, potential variations, or common failure points if identifiable from the memories.
6.  Be structured as a clear set of instructions or a logical flow that could guide similar future situations.

Generate ONLY the procedure based on the provided memories.

CONSOLIDATED PROCEDURE:"""

    elif consolidation_type == "question":
        base_prompt += """TASK: Identify the most important and actionable questions that arise from analyzing these memories. Your questions should:
1.  Target significant gaps in knowledge, understanding, or information revealed by the memories.
2.  Highlight areas of uncertainty, ambiguity, or contradiction that require further investigation or clarification.
3.  Focus on issues that are critical for achieving the implied or stated goals related to these memories.
4.  Be specific and well-defined enough to guide further research, analysis, or action.
5.  Be prioritized, starting with the most critical or foundational questions.
6.  Avoid questions that are already answered within the provided memory content.

Generate ONLY the list of questions based on the provided memories.

CONSOLIDATED QUESTIONS:"""

    # The final marker like "CONSOLIDATED SUMMARY:" is added by the TASK instruction itself.
    return base_prompt


# Helper for reflection prompt (similar structure to consolidation)
def _generate_reflection_prompt(
    workflow_name: str,
    workflow_desc: Optional[str],
    operations: List[Dict],
    memories: Dict[str, Dict],
    reflection_type: str,
) -> str:
    """Generates a prompt for reflective analysis based on the type, with detailed instructions."""

    # Format operations (limited for prompt size)
    op_texts = []
    for i, op_data in enumerate(operations[:30], 1):  # Limit input operations
        op_ts_unix = op_data.get("timestamp", 0)
        op_ts_str = (
            datetime.fromtimestamp(op_ts_unix).strftime("%Y-%m-%d %H:%M:%S")
            if op_ts_unix
            else "Unknown Time"
        )
        op_type = op_data.get("operation", "UNKNOWN").upper()
        mem_id = op_data.get("memory_id")
        action_id = op_data.get("action_id")  # Get action_id if present

        # Extract relevant details from operation_data if present
        op_details_dict = {}
        op_data_raw = op_data.get("operation_data")
        if op_data_raw:
            try:
                op_details_dict = json.loads(op_data_raw)
            except (json.JSONDecodeError, TypeError):
                op_details_dict = {"raw_data": str(op_data_raw)[:50]}  # Fallback

        # Build description string parts
        desc_parts = [f"OP #{i} ({op_ts_str})", f"Type: {op_type}"]
        if mem_id:
            mem_info = memories.get(mem_id) # Use .get() which returns None if key missing
            if mem_info:
                mem_desc_text = f"Mem({mem_id[:6]}..)"
                # Safely get description and type
                mem_desc = mem_info.get('description', 'N/A')
                mem_type_info = mem_info.get('memory_type')
                mem_desc_text += f" Desc: {mem_desc[:40] if mem_desc else 'N/A'}" # Handle None description
                if mem_type_info:
                    mem_desc_text += f" Type: {mem_type_info}"
                desc_parts.append(mem_desc_text)
            else:
                # Log specifically when this happens IN THE DEMO CONTEXT
                logger.warning(
                    f"Reflection prompt generator: Memory details not found for mem_id '{mem_id}' "
                    f"(referenced in operation log entry #{i}, type: {op_type}). "
                    f"This is unexpected in the demo."
                )
                desc_parts.append(f"Mem({mem_id[:6]}.. NOT FOUND)")

        if action_id:
            desc_parts.append(f"Action({action_id[:6]}..)")

        # Add details from operation_data, excluding verbose fields
        detail_items = []
        for k, v in op_details_dict.items():
            if k not in [
                "content",
                "description",
                "embedding",
                "prompt",
            ]:  # Exclude common large fields
                detail_items.append(f"{k}={str(v)[:30]}")
        if detail_items:
            desc_parts.append(f"Data({', '.join(detail_items)})")

        op_texts.append(" | ".join(desc_parts))

    operations_str = "\n".join(op_texts)

    # Base prompt template
    base_prompt = f"""You are an advanced meta-cognitive system analyzing an AI agent's workflow: "{workflow_name}".
Workflow Description: {workflow_desc or "N/A"}
Your task is to perform a '{reflection_type}' reflection based on the recent memory operations listed below (newest first). Analyze these operations to understand the agent's process, progress, and knowledge state.

RECENT OPERATIONS (Up to 30):
{operations_str}

"""

    # --- Add specific instructions based on reflection type (FULL INSTRUCTIONS) ---
    if reflection_type == "summary":
        base_prompt += """TASK: Create a reflective summary of this workflow's progress and current state. Your summary should:
1. Trace the key developments, insights, and significant actions derived from the provided memory operations.
2. Identify the primary focus areas suggested by recent operations and the nature of memories being created or accessed.
3. Summarize the overall arc of the agent's thinking, knowledge acquisition, and task execution during this period.
4. Organize the summary logically (e.g., chronologically by operation, thematically by task).
5. Include a concise assessment of the current state of understanding or progress relative to the workflow's implied or stated goal.

REFLECTIVE SUMMARY:"""

    elif reflection_type == "progress":
        base_prompt += """TASK: Analyze the progress the agent has made toward its goals and understanding within this workflow, based *only* on the provided operations. Your analysis should:
1. Infer the likely immediate goals or sub-tasks the agent was pursuing during these operations.
2. Assess what tangible progress was made toward these inferred goals (e.g., information gathered, artifacts created, decisions made, errors overcome).
3. Highlight key milestones evident in the operations (e.g., successful tool use, creation of important memories, focus shifts).
4. Note operations that suggest stalled progress, repeated actions, failures, or areas needing more work.
5. If possible, suggest observable indicators or metrics from future operations that would signify further progress.

PROGRESS ANALYSIS:"""

    elif reflection_type == "gaps":
        base_prompt += """TASK: Identify gaps in knowledge, reasoning, or process suggested by these operations that should be addressed for this workflow. Your analysis should:
1. Pinpoint potential missing information or unanswered questions implied by the operations (e.g., failed actions, repeated queries, lack of evidence for inferences).
2. Identify possible logical inconsistencies, contradictions between memory operations, or areas where reasoning seems weak based on the sequence of operations.
3. Note operations indicating low confidence (if available in op_data) or areas where actions were taken without sufficient preceding evidence or planning evident in the log.
4. Formulate specific, actionable questions arising directly from analyzing these operations.
5. Recommend concrete next steps (e.g., specific tool use, memory queries, reasoning steps) suggested by the operations log to address these gaps.

KNOWLEDGE GAPS ANALYSIS:"""

    elif reflection_type == "strengths":
        base_prompt += """TASK: Identify what went well during the sequence of operations and what valuable knowledge or effective strategies were likely employed. Your analysis should focus on patterns *within these operations*:
1. Highlight sequences of operations suggesting successful reasoning patterns, problem-solving approaches, or effective decision-making (e.g., planning followed by successful execution).
2. Note operations that created potentially valuable insights, facts, or summaries (look for 'create' operations with relevant memory types).
3. Identify potentially reusable patterns or methods suggested by successful tool use sequences or memory linking operations.
4. Recognize potentially effective use of information sources or successful navigation of constraints if implied by the operation data.
5. Suggest ways the patterns observed in these successful operations could be leveraged or reinforced moving forward.

STRENGTHS ANALYSIS:"""

    elif reflection_type == "plan":
        base_prompt += """TASK: Based *only* on the provided recent operations and the workflow's current implied state, create a strategic plan for the immediate next steps. Your plan should:
1. Identify the most logical next actions based on the last few operations (e.g., follow up on a question, use results from a tool, consolidate findings).
2. Prioritize information gathering, analysis, or tool use based on the workflow's apparent trajectory revealed in the operations log.
3. Suggest specific, concrete actions, potentially including tool names or memory operations, for the very next phase.
4. Include brief considerations for potential challenges or alternative paths if suggested by recent failures or uncertainty in the log.
5. Define what a successful outcome for the immediate next 1-3 steps would look like, based on the context provided by the operations.

STRATEGIC PLAN:"""

    # Note: The final marker like "REFLECTIVE SUMMARY:" is added by the calling function `generate_reflection`.
    return base_prompt


# --- Tool: Consolidate Memories ---
@with_tool_metrics
@with_error_handling
async def consolidate_memories(
    workflow_id: Optional[str] = None,
    target_memories: Optional[List[str]] = None,
    consolidation_type: str = "summary",
    query_filter: Optional[Dict[str, Any]] = None,
    max_source_memories: int = 20,
    prompt_override: Optional[str] = None,
    provider: Optional[str] = None, # Changed default to None
    model: Optional[str] = None,    # Changed default to None
    store_result: bool = True,
    store_as_level: str = MemoryLevel.SEMANTIC.value,
    store_as_type: Optional[str] = None,
    max_tokens: int = 1000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Consolidates multiple memories using an LLM to generate summaries, insights, etc."""
    start_time = time.time()
    valid_types = ["summary", "insight", "procedural", "question"]
    if consolidation_type not in valid_types:
        raise ToolInputError(
            f"consolidation_type must be one of: {valid_types}", param_name="consolidation_type"
        )

    source_memories_list = []
    source_memory_ids = []
    effective_workflow_id = workflow_id

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Select Source Memories (Full Logic with Corrected Validation) ---
            if target_memories:
                if not isinstance(target_memories, list) or len(target_memories) < 2:
                    raise ToolInputError(
                        "target_memories must be a list containing at least 2 memory IDs.",
                        param_name="target_memories",
                    )

                # Fetch specified memories and their workflow IDs
                placeholders = ", ".join(["?"] * len(target_memories))
                # Fetch workflow_id along with other columns
                query = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
                params = list(target_memories) # Use list directly

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                found_memories_details = {r["memory_id"]: dict(r) for r in rows}
                found_mem_ids = set(found_memories_details.keys())
                requested_ids_set = set(target_memories)

                # --- START CORRECTED VALIDATION ---
                mismatched_workflow_ids = set()
                not_found_ids = requested_ids_set - found_mem_ids

                # Determine the effective workflow ID. If not provided, infer from the first found memory.
                # If provided, use it for validation.
                if not effective_workflow_id and found_mem_ids:
                    first_found_id = next(iter(found_mem_ids))
                    effective_workflow_id = found_memories_details[first_found_id].get("workflow_id")
                    if not effective_workflow_id:
                        # This case is highly unlikely if data integrity is maintained,
                        # but handle defensively.
                        raise ToolError(f"Memory {first_found_id} exists but lacks a workflow ID.")
                    logger.debug(f"Inferred effective_workflow_id: {effective_workflow_id} from target memories.")
                elif not effective_workflow_id and not found_mem_ids:
                    raise ToolInputError(
                        "Workflow ID must be provided if target_memories are specified but none are found.",
                        param_name="workflow_id"
                        )

                # Now validate all found memories against the effective_workflow_id
                for mem_id, mem_data in found_memories_details.items():
                    if mem_data.get("workflow_id") != effective_workflow_id:
                        mismatched_workflow_ids.add(mem_id)

                problematic_ids = not_found_ids.union(mismatched_workflow_ids)
                # --- END CORRECTED VALIDATION ---

                # Use the combined problematic_ids set for error reporting
                if problematic_ids:
                    # Format error message using the actual problematic IDs
                    missing_ids_str = ", ".join(list(problematic_ids)[:5]) + ("..." if len(problematic_ids) > 5 else "")
                    err_msg = f"Target memories issue. The following requested IDs were not found or do not belong to the effective workflow '{effective_workflow_id}': {missing_ids_str}"
                    raise ToolInputError(err_msg, param_name="target_memories")

                # If validation passes, proceed with the found memories that match the workflow
                source_memories_list = [
                    mem_data for mem_id, mem_data in found_memories_details.items()
                    if mem_id not in mismatched_workflow_ids # Only include those matching the workflow
                ]
                source_memory_ids = [mem["memory_id"] for mem in source_memories_list] # IDs that are valid and match workflow

            elif query_filter:
                # Build filter query dynamically
                filter_where = ["1=1"]
                filter_params = []
                if effective_workflow_id:
                    filter_where.append("workflow_id = ?")
                    filter_params.append(effective_workflow_id)
                for key, value in query_filter.items():
                    # Ensure key is a valid column name before adding
                    valid_filter_keys = {"memory_level", "memory_type", "source", "min_importance", "min_confidence"}
                    if key not in valid_filter_keys:
                         logger.warning(f"Ignoring unsupported filter key: {key}")
                         continue

                    if key in ["memory_level", "memory_type", "source"] and value:
                        filter_where.append(f"{key} = ?")
                        filter_params.append(value)
                    elif key == "min_importance" and value is not None:
                        filter_where.append("importance >= ?")
                        filter_params.append(float(value))
                    elif key == "min_confidence" and value is not None:
                        filter_where.append("confidence >= ?")
                        filter_params.append(float(value))

                # Add TTL check
                now_unix = int(time.time())
                filter_where.append("(ttl = 0 OR created_at + ttl > ?)")
                filter_params.append(now_unix)

                query = f"SELECT * FROM memories WHERE {' AND '.join(filter_where)} ORDER BY importance DESC, created_at DESC LIMIT ?"
                filter_params.append(max_source_memories)

                async with conn.execute(query, filter_params) as cursor:
                    source_memories_list = [dict(row) for row in await cursor.fetchall()]
                    source_memory_ids = [m["memory_id"] for m in source_memories_list]
                    if not effective_workflow_id and source_memories_list:
                        effective_workflow_id = source_memories_list[0].get("workflow_id")
            else:
                # Default: Get recent, important memories from the specified workflow
                if not effective_workflow_id:
                    raise ToolInputError(
                        "workflow_id is required if not using target_memories or query_filter.",
                        param_name="workflow_id",
                    )
                query = "SELECT * FROM memories WHERE workflow_id = ? AND (ttl = 0 OR created_at + ttl > ?) ORDER BY importance DESC, created_at DESC LIMIT ?"
                now_unix = int(time.time())
                async with conn.execute(
                    query, [effective_workflow_id, now_unix, max_source_memories]
                ) as cursor:
                    source_memories_list = [dict(row) for row in await cursor.fetchall()]
                    source_memory_ids = [m["memory_id"] for m in source_memories_list]

            if len(source_memories_list) < 2:
                # Log the criteria used when insufficient memories are found
                criteria_desc = ""
                if target_memories:
                     criteria_desc = f"target_memories={target_memories}"
                elif query_filter:
                     criteria_desc = f"query_filter={query_filter}"
                elif effective_workflow_id:
                     criteria_desc = f"workflow_id={effective_workflow_id} (default recent/important)"

                logger.warning(f"Insufficient source memories found ({len(source_memories_list)}) for consolidation based on criteria: {criteria_desc}.")
                raise ToolError(
                    f"Insufficient source memories found ({len(source_memories_list)}) for consolidation based on criteria."
                )

            if not effective_workflow_id:
                # This should ideally be caught earlier, but added as a final safeguard
                raise ToolError("Could not determine a workflow ID for consolidation.")

            # --- 2. Generate Consolidation Prompt (using the full helper) ---
            prompt = prompt_override or _generate_consolidation_prompt(
                source_memories_list, consolidation_type
            )

            # --- 3. Call LLM via Gateway (Dynamic Provider/Model) ---
            consolidated_content = ""
            provider_to_use = provider or config.default_provider or LLMGatewayProvider.OPENAI.value # Fallback chain
            try:
                provider_instance = await get_provider(provider_to_use)
                if not provider_instance:
                     raise ToolError(f"Failed to initialize provider '{provider_to_use}'.")

                # Use passed model, or provider's default
                model_to_use = model or provider_instance.get_default_model()
                if not model_to_use:
                     logger.warning(f"Provider '{provider_to_use}' has no default model configured. LLM call might fail.")
                     # Allow attempting the call without a model, provider might handle it

                logger.info(f"Consolidating memories using LLM: {provider_to_use}/{model_to_use or 'provider_default'}...")

                llm_result = await provider_instance.generate_completion(
                    prompt=prompt, model=model_to_use, max_tokens=max_tokens, temperature=0.6
                )
                consolidated_content = llm_result.text.strip()

                if not consolidated_content:
                    logger.warning(
                        "LLM returned empty content for consolidation. Cannot store result."
                    )
                    consolidated_content = ""
                else:
                    logger.debug(
                        f"LLM consolidation successful. Content length: {len(consolidated_content)}"
                    )
            except Exception as llm_err:
                logger.error(f"LLM call failed during consolidation: {llm_err}", exc_info=True)
                raise ToolError(f"Consolidation failed due to LLM error: {llm_err}") from llm_err

            # --- 4. Store Result ---
            stored_memory_id = None
            if store_result and consolidated_content:
                # Determine memory type for the result
                result_type_val = store_as_type or {
                    "summary": MemoryType.SUMMARY.value,
                    "insight": MemoryType.INSIGHT.value,
                    "procedural": MemoryType.PROCEDURE.value,
                    "question": MemoryType.QUESTION.value,
                }.get(consolidation_type, MemoryType.INSIGHT.value) # Default to insight
                try:
                    result_type = MemoryType(result_type_val.lower())
                except ValueError:
                    logger.warning(f"Invalid store_as_type '{result_type_val}', defaulting to insight.")
                    result_type = MemoryType.INSIGHT

                # Determine memory level for the result
                try:
                    result_level = MemoryLevel(store_as_level.lower())
                except ValueError:
                    logger.warning(f"Invalid store_as_level '{store_as_level}', defaulting to semantic.")
                    result_level = MemoryLevel.SEMANTIC

                result_desc = (
                    f"Consolidated {consolidation_type} from {len(source_memory_ids)} memories."
                )
                result_tags = ["consolidated", consolidation_type, result_type.value, result_level.value]
                result_context = {"source_memories": source_memory_ids, "consolidation_type": consolidation_type}

                # --- Calculate derived importance and confidence ---
                derived_importance = 5.0  # Default
                derived_confidence = 0.75  # Default
                if source_memories_list:
                    # Filter out potential None values before calculating max/sum
                    source_importances = [m.get("importance", 5.0) for m in source_memories_list if m.get("importance") is not None]
                    source_confidences = [m.get("confidence", 0.5) for m in source_memories_list if m.get("confidence") is not None]

                    if source_importances:
                         # Importance: Max source importance + small boost, capped at 10
                         derived_importance = min(max(source_importances) + 0.5, 10.0)

                    if source_confidences:
                         # Confidence: Average source confidence, capped at 1
                         derived_confidence = min(sum(source_confidences) / len(source_confidences), 1.0)
                         # Add a slight penalty based on number of sources
                         derived_confidence = max(
                             0.1,
                             derived_confidence
                             * (1.0 - min(0.2, (len(source_memories_list) - 1) * 0.02)),
                         )
                    else: # Handle case where all source confidences were None
                         derived_confidence = 0.5 # Fallback confidence

                logger.debug(
                    f"Derived Importance: {derived_importance:.2f}, Confidence: {derived_confidence:.2f}"
                )
                # --- End calculation ---

                try:
                    # Use store_memory tool function with derived values
                    # Pass the connection to store_memory if it's implemented to accept it, otherwise rely on its internal connection management
                    store_result_dict = await store_memory(
                        workflow_id=effective_workflow_id,
                        content=consolidated_content,
                        memory_type=result_type.value,
                        memory_level=result_level.value,
                        importance=round(derived_importance, 2),
                        confidence=round(derived_confidence, 3),
                        description=result_desc,
                        source=f"consolidation_{consolidation_type}",
                        tags=result_tags,
                        context_data=result_context,
                        generate_embedding=True, # Usually good to embed consolidations
                        suggest_links=True, # Also suggest links for the new memory
                        db_path=db_path,
                        # conn=conn # Pass connection if store_memory supports it for transactions
                    )
                    stored_memory_id = store_result_dict.get("memory_id")
                except Exception as store_err:
                    logger.error(
                        f"Failed to store consolidated memory result: {store_err}", exc_info=True
                    )
                    # Continue without storing if it fails, but log the error

            # --- 5. Link Result to Sources ---
            if stored_memory_id:
                link_tasks = []
                for source_id in source_memory_ids:
                    # Use create_memory_link tool function
                    # Again, pass connection if supported, otherwise rely on its management
                    link_task = create_memory_link(
                        source_memory_id=stored_memory_id,
                        target_memory_id=source_id,
                        link_type=LinkType.GENERALIZES.value, # Consolidated memory generalizes sources
                        description=f"Source for consolidated {consolidation_type}",
                        db_path=db_path,
                        # conn=conn # Pass connection if supported
                    )
                    link_tasks.append(link_task)
                # Use asyncio.gather with return_exceptions=True to handle potential errors
                link_results = await asyncio.gather(*link_tasks, return_exceptions=True)
                failed_links = [res for res in link_results if isinstance(res, Exception)]
                if failed_links:
                    logger.warning(
                        f"Failed to create {len(failed_links)} links from consolidated memory {stored_memory_id} to sources. Errors: {[str(e) for e in failed_links]}"
                    )

            # --- 6. Log Consolidation Operation ---
            log_data = {
                "consolidation_type": consolidation_type,
                "source_count": len(source_memory_ids),
                "llm_provider": provider_to_use, # Log the actual provider used
                "llm_model": model_to_use or "provider_default", # Log the actual model used
                "stored": bool(stored_memory_id),
                "stored_memory_id": stored_memory_id,
            }
            await MemoryUtils._log_memory_operation(
                conn, effective_workflow_id, "consolidate", None, None, log_data
            )

            # Commit the logging operation (and potentially link creation if done within this transaction)
            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Consolidated {len(source_memory_ids)} memories ({consolidation_type}). Stored as: {stored_memory_id or 'Not Stored'}",
                emoji_key="sparkles",
                time=processing_time,
            )
            return {
                "consolidated_content": consolidated_content or "Consolidation failed or produced no content.",
                "consolidation_type": consolidation_type,
                "source_memory_ids": source_memory_ids,
                "workflow_id": effective_workflow_id,
                "stored_memory_id": stored_memory_id,
                "success": True, # Success of the operation itself, even if LLM failed or storage failed
                "processing_time": processing_time,
            }

    except (ToolInputError, ToolError) as e: # Catch specific handled errors
        # Log these at warning level as they are often user/input related
        logger.warning(f"Consolidation failed due to input/tool error: {e}")
        raise # Re-raise to be handled by the @with_error_handling decorator
    except Exception as e:
        # Log unexpected errors at error level
        logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)
        # Wrap in ToolError for consistent error response structure
        raise ToolError(f"Unexpected error during memory consolidation: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def generate_reflection(
    workflow_id: str,
    reflection_type: str = "summary",  # summary, progress, gaps, strengths, plan
    recent_ops_limit: int = 30,
    provider: str = LLMGatewayProvider.OPENAI.value,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Generates a reflective analysis of a workflow using an LLM.

    Args:
        workflow_id: ID of the workflow to reflect on.
        reflection_type: Type of reflection ('summary', 'progress', 'gaps', 'strengths', 'plan').
        recent_ops_limit: (Optional) Number of recent operations to analyze. Default 30.
        provider: (Optional) LLM provider name.
        model: (Optional) Specific LLM model name.
        max_tokens: (Optional) Max tokens for LLM response. Default 1000.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the reflection details.
    """
    start_time = time.time()
    valid_types = ["summary", "progress", "gaps", "strengths", "plan"]
    if reflection_type not in valid_types:
        raise ToolInputError("Invalid reflection_type.", param_name="reflection_type")

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Fetch Workflow Info & Recent Operations ---
            async with conn.execute(
                "SELECT title, description FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                wf_row = await cursor.fetchone()
                if not wf_row:
                    raise ToolInputError(
                        f"Workflow {workflow_id} not found.", param_name="workflow_id"
                    )
                workflow_name = wf_row["title"]
                workflow_desc = wf_row["description"]

            operations = []
            async with conn.execute(
                "SELECT * FROM memory_operations WHERE workflow_id = ? ORDER BY timestamp DESC LIMIT ?",
                (workflow_id, recent_ops_limit),
            ) as cursor:
                operations = [dict(row) for row in await cursor.fetchall()]

            if not operations:
                raise ToolError("No operations found for reflection.")

            # --- 2. Fetch Details of Referenced Memories ---
            mem_ids = {op["memory_id"] for op in operations if op.get("memory_id")}
            memories_details = {}
            if mem_ids:
                placeholders = ",".join("?" * len(mem_ids))
                async with conn.execute(
                    f"SELECT memory_id, description FROM memories WHERE memory_id IN ({placeholders})",
                    list(mem_ids),
                ) as cursor:
                    async for row in cursor:
                        memories_details[row["memory_id"]] = dict(row)

            # --- 3. Generate Reflection Prompt ---
            prompt = _generate_reflection_prompt(
                workflow_name, workflow_desc, operations, memories_details, reflection_type
            )

            # --- 4. Call LLM via Gateway ---
            try:
                provider_instance = await get_provider(provider)
                llm_result = await provider_instance.generate_completion(
                    prompt=prompt, model=model, max_tokens=max_tokens, temperature=0.7
                )
                reflection_content = llm_result.text.strip()
                if not reflection_content:
                    raise ToolError("LLM returned empty reflection.")
            except Exception as llm_err:
                logger.error(f"LLM call failed during reflection: {llm_err}", exc_info=True)
                raise ToolError(f"Reflection failed due to LLM error: {llm_err}") from llm_err

            # --- 5. Store Reflection ---
            reflection_id = MemoryUtils.generate_id()
            now_unix = int(time.time())
            # Extract title from content (simple approach)
            title = (
                reflection_content.split("\n", 1)[0].strip("# ")[:100]
                or f"{reflection_type.capitalize()} Reflection"
            )
            referenced_memory_ids = list(mem_ids)  # Store IDs of memories involved in operations

            await conn.execute(
                """INSERT INTO reflections
                    (reflection_id, workflow_id, title, content, reflection_type, created_at, referenced_memories)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    reflection_id,
                    workflow_id,
                    title,
                    reflection_content,
                    reflection_type,
                    now_unix,
                    json.dumps(referenced_memory_ids),
                ),
            )

            # Log operation
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "reflect",
                None,
                None,
                {
                    "reflection_id": reflection_id,
                    "reflection_type": reflection_type,
                    "ops_analyzed": len(operations),
                },
            )

            await conn.commit()

            processing_time = time.time() - start_time
            logger.info(
                f"Generated reflection '{title}' ({reflection_id}) for workflow {workflow_id}",
                emoji_key="mirror",
            )
            return {
                "reflection_id": reflection_id,
                "reflection_type": reflection_type,
                "title": title,
                "content": reflection_content,  # Return full content
                "workflow_id": workflow_id,
                "operations_analyzed": len(operations),
                "success": True,
                "processing_time": processing_time,
            }

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Failed to generate reflection: {str(e)}", exc_info=True)
        raise ToolError(f"Failed to generate reflection: {str(e)}") from e


# ======================================================
# Text Summarization (using LLM)
# ======================================================


@with_tool_metrics
@with_error_handling
async def summarize_text(
    text_to_summarize: str,
    target_tokens: int = 500,
    prompt_template: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    workflow_id: Optional[str] = None,
    record_summary: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Summarizes text content using an LLM to generate a concise summary.

    Uses the configured LLM provider to generate a summary of the provided text,
    optimizing for the requested token length. Optionally stores the summary
    as a memory in the specified workflow.

    Args:
        text_to_summarize: Text content to summarize
        target_tokens: Approximate desired length of summary (default 500)
        prompt_template: Optional custom prompt template for summarization
        provider: LLM provider to use for summarization (default "openai")
        model: Specific model to use, or None for provider default
        workflow_id: Optional workflow ID to store the summary in
        record_summary: Whether to store the summary as a memory
        db_path: Path to the SQLite database file

    Returns:
        Dictionary containing the generated summary:
        {
            "summary": "Concise summary text...",
            "original_length": 2500,  # Approximate character count
            "summary_length": 350,    # Approximate character count
            "stored_memory_id": "memory-uuid" | None,  # Only if record_summary=True
            "success": true,
            "processing_time": 0.123
        }

    Raises:
        ToolInputError: If text_to_summarize is empty or provider/model is invalid
        ToolError: If summarization fails or database operation fails
    """
    start_time = time.time()

    if not text_to_summarize:
        raise ToolInputError("Text to summarize cannot be empty", param_name="text_to_summarize")

    if record_summary and not workflow_id:
        raise ToolInputError(
            "Workflow ID is required when record_summary=True", param_name="workflow_id"
        )

    # Ensure target_tokens is reasonable
    target_tokens = max(50, min(2000, target_tokens))

    # Use default models for common providers if none specified
    default_models = {"openai": "gpt-4.1-mini", "anthropic": "claude-3-5-haiku-20241022"}
    model_to_use = model or default_models.get(provider)

    # Default prompt template if none provided
    if not prompt_template:
        prompt_template = """
You are an expert summarizer. Your task is to create a concise and accurate summary of the following text.
The summary should be approximately {target_tokens} tokens long.
Focus on the main points, key information, and essential details.
Maintain the tone and factual accuracy of the original text.

TEXT TO SUMMARIZE:
{text_to_summarize}

CONCISE SUMMARY:
"""

    try:
        # Get provider instance from ultimate
        provider_instance = await get_provider(provider)
        if not provider_instance:
            raise ToolError(f"Failed to initialize provider '{provider}'. Check configuration.")

        # Prepare prompt by filling in the template
        prompt = prompt_template.format(
            text_to_summarize=text_to_summarize, target_tokens=target_tokens
        )

        # Generate summary using LLM provider
        generation_result = await provider_instance.generate_completion(
            prompt=prompt,
            model=model_to_use,
            max_tokens=target_tokens + 100,  # Add some buffer for prompt tokens
            temperature=0.3,  # Lower temperature for more deterministic summaries
        )

        summary_text = generation_result.text.strip()
        if not summary_text:
            raise ToolError("LLM returned empty summary.")

        # Optional: Store summary as a memory
        stored_memory_id = None
        if record_summary and workflow_id:
            # Validate workflow exists
            async with DBConnection(db_path) as conn:
                async with conn.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Workflow {workflow_id} not found", param_name="workflow_id"
                        )

                # Create new memory entry for the summary
                memory_id = MemoryUtils.generate_id()
                now_unix = int(time.time())

                description = f"Summary of text ({len(text_to_summarize)} chars)"
                tags = json.dumps(["summary", "automated", "text_summary"])

                await conn.execute(
                    """
                    INSERT INTO memories (memory_id, workflow_id, content, memory_level, memory_type, 
                                        importance, confidence, description, source, tags, 
                                        created_at, updated_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        workflow_id,
                        summary_text,
                        MemoryLevel.SEMANTIC.value,
                        MemoryType.SUMMARY.value,
                        6.0,
                        0.85,
                        description,
                        "summarize_text",
                        tags,
                        now_unix,
                        now_unix,
                        0,
                    ),
                )

                # Log the operation
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "create_summary",
                    memory_id,
                    None,
                    {
                        "original_length": len(text_to_summarize),
                        "summary_length": len(summary_text),
                    },
                )

                await conn.commit()
                stored_memory_id = memory_id

        processing_time = time.time() - start_time
        logger.info(
            f"Generated summary of {len(text_to_summarize)} chars text to {len(summary_text)} chars",
            emoji_key="scissors",
            time=processing_time,
        )

        return {
            "summary": summary_text,
            "original_length": len(text_to_summarize),
            "summary_length": len(summary_text),
            "stored_memory_id": stored_memory_id,
            "success": True,
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error summarizing text: {e}", exc_info=True)
        raise ToolError(f"Failed to summarize text: {str(e)}") from e


# --- 17. Maintenance (Adapted from cognitive_memory) ---
@with_tool_metrics
@with_error_handling
async def delete_expired_memories(db_path: str = agent_memory_config.db_path) -> Dict[str, Any]:
    """Deletes memories that have exceeded their time-to-live (TTL)."""
    start_time = time.time()
    deleted_count = 0
    workflows_affected = set()

    try:
        async with DBConnection(db_path) as conn:
            now_unix = int(time.time())
            # Find expired memory IDs and their workflows
            expired_memories = []
            async with conn.execute(
                "SELECT memory_id, workflow_id FROM memories WHERE ttl > 0 AND created_at + ttl < ?",
                (now_unix,),
            ) as cursor:
                expired_memories = await cursor.fetchall()

            if not expired_memories:
                logger.info("No expired memories found to delete.")
                return {
                    "deleted_count": 0,
                    "workflows_affected": [],
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            expired_ids = [row["memory_id"] for row in expired_memories]
            workflows_affected = {row["workflow_id"] for row in expired_memories}
            deleted_count = len(expired_ids)

            # Delete in batches to avoid issues with too many placeholders
            batch_size = 500
            for i in range(0, deleted_count, batch_size):
                batch_ids = expired_ids[i : i + batch_size]
                placeholders = ", ".join(["?"] * len(batch_ids))
                # Delete from memories table (FK constraints handle related embeddings/links)
                await conn.execute(
                    f"DELETE FROM memories WHERE memory_id IN ({placeholders})", batch_ids
                )
                logger.debug(f"Deleted batch of {len(batch_ids)} expired memories.")

            # Log the operation for each affected workflow
            for wf_id in workflows_affected:
                await MemoryUtils._log_memory_operation(
                    conn,
                    wf_id,
                    "expire_batch",
                    None,
                    None,
                    {
                        "expired_count_in_workflow": sum(
                            1 for mid, wid in expired_memories if wid == wf_id
                        )
                    },
                )

            await conn.commit()

            processing_time = time.time() - start_time
            logger.success(
                f"Deleted {deleted_count} expired memories across {len(workflows_affected)} workflows.",
                emoji_key="wastebasket",
                time=processing_time,
            )
            return {
                "deleted_count": deleted_count,
                "workflows_affected": list(workflows_affected),
                "success": True,
                "processing_time": processing_time,
            }

    except Exception as e:
        logger.error(f"Failed to delete expired memories: {str(e)}", exc_info=True)
        raise ToolError(f"Failed to delete expired memories: {str(e)}") from e


# --- 18. Statistics (Adapted from cognitive_memory) ---
@with_tool_metrics
@with_error_handling
async def compute_memory_statistics(
    workflow_id: Optional[str] = None,  # Optional: If None, compute global stats
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Computes statistics about memories, optionally filtered by workflow."""
    start_time = time.time()
    stats: Dict[str, Any] = {"scope": workflow_id or "global"}

    try:
        async with DBConnection(db_path) as conn:
            # Base WHERE clause and params
            where_clause = "WHERE workflow_id = ?" if workflow_id else ""
            params = [workflow_id] if workflow_id else []

            # Total Memories
            async with conn.execute(
                f"SELECT COUNT(*) FROM memories {where_clause}", params
            ) as cursor:
                stats["total_memories"] = (await cursor.fetchone())[0]

            if stats["total_memories"] == 0:
                stats.update({"success": True, "processing_time": time.time() - start_time})
                logger.info(f"No memories found for statistics in scope: {stats['scope']}")
                return stats

            # By Level
            async with conn.execute(
                f"SELECT memory_level, COUNT(*) FROM memories {where_clause} GROUP BY memory_level",
                params,
            ) as cursor:
                stats["by_level"] = {row["memory_level"]: row[1] for row in await cursor.fetchall()}
            # By Type
            async with conn.execute(
                f"SELECT memory_type, COUNT(*) FROM memories {where_clause} GROUP BY memory_type",
                params,
            ) as cursor:
                stats["by_type"] = {row["memory_type"]: row[1] for row in await cursor.fetchall()}

            # Confidence & Importance Aggregates
            async with conn.execute(
                f"SELECT AVG(confidence), AVG(importance) FROM memories {where_clause}", params
            ) as cursor:
                row = await cursor.fetchone()
                stats["confidence_avg"] = round(row[0], 3) if row[0] is not None else None
                stats["importance_avg"] = round(row[1], 2) if row[1] is not None else None

            # Temporal Stats
            async with conn.execute(
                f"SELECT MAX(created_at), MIN(created_at) FROM memories {where_clause}", params
            ) as cursor:
                row = await cursor.fetchone()
                stats["newest_memory_unix"] = row[0]
                stats["oldest_memory_unix"] = row[1]

            # Link Stats
            link_where = "WHERE m.workflow_id = ?" if workflow_id else ""
            link_params = params  # Reuse params
            async with conn.execute(
                f"SELECT COUNT(*) FROM memory_links ml JOIN memories m ON ml.source_memory_id = m.memory_id {link_where}",
                link_params,
            ) as cursor:
                stats["total_links"] = (await cursor.fetchone())[0]
            async with conn.execute(
                f"SELECT ml.link_type, COUNT(*) FROM memory_links ml JOIN memories m ON ml.source_memory_id = m.memory_id {link_where} GROUP BY ml.link_type",
                link_params,
            ) as cursor:
                stats["links_by_type"] = {
                    row["link_type"]: row[1] for row in await cursor.fetchall()
                }

            # Tag Stats (Top 5)
            tag_where = (
                "WHERE wt.workflow_id = ?" if workflow_id else ""
            )  # Filter by workflow if needed
            tag_params = params  # Reuse params
            async with conn.execute(
                f"""SELECT t.name, COUNT(wt.workflow_id) as count
                                         FROM tags t JOIN workflow_tags wt ON t.tag_id = wt.tag_id {tag_where}
                                         GROUP BY t.tag_id ORDER BY count DESC LIMIT 5""",
                tag_params,
            ) as cursor:
                stats["top_workflow_tags"] = {
                    row["name"]: row["count"] for row in await cursor.fetchall()
                }

            # Workflow Stats (if global)
            if not workflow_id:
                async with conn.execute(
                    "SELECT status, COUNT(*) FROM workflows GROUP BY status"
                ) as cursor:
                    stats["workflows_by_status"] = {
                        row["status"]: row[1] for row in await cursor.fetchall()
                    }

            stats["success"] = True
            stats["processing_time"] = time.time() - start_time
            logger.info(
                f"Computed memory statistics for scope: {stats['scope']}", emoji_key="bar_chart"
            )
            return stats

    except Exception as e:
        logger.error(f"Failed to compute statistics: {str(e)}", exc_info=True)
        raise ToolError(f"Failed to compute statistics: {str(e)}") from e


def _mermaid_escape(text: str) -> str:
    """Escapes characters problematic for Mermaid node labels."""
    if not isinstance(text, str):
        text = str(text)
    # Replace quotes first, then other potentially problematic characters
    text = text.replace('"', "#quot;")
    text = text.replace("(", "#40;")
    text = text.replace(")", "#41;")
    text = text.replace("[", "#91;")
    text = text.replace("]", "#93;")
    text = text.replace("{", "#123;")
    text = text.replace("}", "#125;")
    text = text.replace(":", "#58;")
    text = text.replace(";", "#59;")
    text = text.replace("<", "#lt;")
    text = text.replace(">", "#gt;")
    # Replace newline with <br> for multiline labels if needed, or just space
    text = text.replace("\n", "<br>")
    return text


async def _generate_mermaid_diagram(workflow: Dict[str, Any]) -> str:
    """Generates a detailed Mermaid flowchart representation of the workflow."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            # Generate a unique fallback for missing IDs to avoid collisions
            return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
        # Replace hyphens which are problematic in unquoted Mermaid node IDs
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "flowchart TD"]  # Top-Down flowchart

    # --- Workflow Node ---
    wf_node_id = sanitize_mermaid_id(workflow.get("workflow_id"), "W")  # Use sanitized full ID
    wf_title = _mermaid_escape(workflow.get("title", "Workflow"))
    wf_status_class = f":::{workflow.get('status', 'active')}"  # Style based on status
    diagram.append(f'    {wf_node_id}("{wf_title}"){wf_status_class}')
    diagram.append("")  # Spacer

    # --- Action Nodes & Links ---
    action_nodes = {}  # Map action_id to mermaid_node_id
    parent_links = {}  # Map child_action_id to parent_action_id
    sequential_links = {}  # Map sequence_number to action_id for sequential linking if no parent

    actions = sorted(workflow.get("actions", []), key=lambda a: a.get("sequence_number", 0))

    for i, action in enumerate(actions):
        action_id = action.get("action_id")
        if not action_id:
            continue  # Skip actions somehow missing an ID

        node_id = sanitize_mermaid_id(action_id, "A")  # Use sanitized full ID
        action_nodes[action_id] = node_id
        sequence_number = action.get("sequence_number", i)  # Use sequence number if available

        # Label: Include type, title, and potentially tool name
        action_type = action.get("action_type", "Action").capitalize()
        action_title = _mermaid_escape(action.get("title", action_type))
        label = f"<b>{action_type} #{sequence_number}</b><br/>{action_title}"
        if action.get("tool_name"):
            label += f"<br/><i>Tool: {_mermaid_escape(action['tool_name'])}</i>"

        # Node shape/style based on status
        status = action.get("status", ActionStatus.PLANNED.value)
        node_style = f":::{status}"  # Use status directly for class name

        # Node Definition
        diagram.append(f'    {node_id}["{label}"]{node_style}')

        # Store parent relationship
        parent_action_id = action.get("parent_action_id")
        if parent_action_id:
            parent_links[action_id] = parent_action_id
        else:
            sequential_links[sequence_number] = action_id

    diagram.append("")  # Spacer

    # Draw Links: Parent/Child first, then sequential for roots
    linked_actions = set()
    # Parent->Child links
    for child_id, parent_id in parent_links.items():
        if child_id in action_nodes and parent_id in action_nodes:
            child_node = action_nodes[child_id]
            parent_node = action_nodes[parent_id]
            diagram.append(f"    {parent_node} --> {child_node}")
            linked_actions.add(child_id)  # Mark child as linked

    # Sequential links for actions without explicit parents
    last_sequential_node = wf_node_id  # Start sequence from workflow node
    sorted_sequences = sorted(sequential_links.keys())
    for seq_num in sorted_sequences:
        action_id = sequential_links[seq_num]
        if action_id in action_nodes:  # Ensure action node exists
            node_id = action_nodes[action_id]
            diagram.append(f"    {last_sequential_node} --> {node_id}")
            last_sequential_node = node_id  # Chain sequential actions
            linked_actions.add(action_id)  # Mark root as linked

    # Link any remaining unlinked actions (e.g., if parents were missing/invalid) sequentially
    for action in actions:
        action_id = action.get("action_id")
        if action_id and action_id not in linked_actions and action_id in action_nodes:
            node_id = action_nodes[action_id]
            # Link from workflow if no other link established
            diagram.append(f"    {wf_node_id} -.-> {node_id} :::orphanLink")
            logger.debug(f"Linking orphan action {action_id} to workflow.")

    diagram.append("")  # Spacer

    # --- Artifact Nodes & Links ---
    artifacts = workflow.get("artifacts", [])
    if artifacts:
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id")
            if not artifact_id:
                continue  # Skip artifacts missing ID

            node_id = sanitize_mermaid_id(artifact_id, "F")  # Use sanitized full ID
            artifact_name = _mermaid_escape(artifact.get("name", "Artifact"))
            artifact_type = _mermaid_escape(artifact.get("artifact_type", "file"))
            label = f"📄<br/><b>{artifact_name}</b><br/>({artifact_type})"

            # Node shape/style based on type/output status
            node_shape_start, node_shape_end = "[(", ")]"  # Default: capsule for artifacts
            node_style = ":::artifact"
            if artifact.get("is_output"):
                node_style = ":::artifact_output"  # Style final outputs differently

            diagram.append(f'    {node_id}{node_shape_start}"{label}"{node_shape_end}{node_style}')

            # Link from creating action or workflow
            creator_action_id = artifact.get("action_id")
            if creator_action_id and creator_action_id in action_nodes:
                creator_node = action_nodes[creator_action_id]
                diagram.append(f"    {creator_node} -- Creates --> {node_id}")
            else:
                # Link artifact to workflow if no specific action created it
                diagram.append(f"    {wf_node_id} -.-> {node_id}")

    # --- Class Definitions (Full Set) ---
    diagram.append("\n    %% Stylesheets")
    diagram.append("    classDef workflow fill:#e7f0fd,stroke:#0056b3,stroke-width:2px,color:#000")
    # Action Statuses
    diagram.append(
        "    classDef completed fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724"
    )
    diagram.append("    classDef failed fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24")
    diagram.append(
        "    classDef skipped fill:#e2e3e5,stroke:#383d41,stroke-width:1px,color:#383d41"
    )
    diagram.append(
        "    classDef in_progress fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404"
    )
    diagram.append(
        "    classDef planned fill:#fefefe,stroke:#6c757d,stroke-width:1px,color:#343a40,stroke-dasharray: 3 3"
    )
    # Artifacts
    diagram.append("    classDef artifact fill:#fdfae7,stroke:#b3a160,stroke-width:1px,color:#333")
    diagram.append(
        "    classDef artifact_output fill:#e7fdf4,stroke:#2e855d,stroke-width:2px,color:#000"
    )
    diagram.append("    classDef orphanLink stroke:#ccc,stroke-dasharray: 2 2")

    diagram.append("```")
    return "\n".join(diagram)


async def _generate_thought_chain_mermaid(thought_chain: Dict[str, Any]) -> str:
    """Generates a detailed Mermaid flowchart of a thought chain."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "graph TD"]  # Top-Down graph

    # --- Header Node ---
    chain_node_id = sanitize_mermaid_id(
        thought_chain.get("thought_chain_id"), "TC"
    )  # Use sanitized full ID
    chain_title = _mermaid_escape(thought_chain.get("title", "Thought Chain"))
    diagram.append(f'    {chain_node_id}("{chain_title}"):::header')
    diagram.append("")

    # --- Thought Nodes ---
    thoughts = sorted(thought_chain.get("thoughts", []), key=lambda t: t.get("sequence_number", 0))
    thought_nodes = {}  # Map thought_id to mermaid_node_id
    parent_links = {}  # Map child_thought_id to parent_thought_id

    if thoughts:
        for thought in thoughts:
            thought_id = thought.get("thought_id")
            if not thought_id:
                continue

            node_id = sanitize_mermaid_id(thought_id, "T")  # Use sanitized full ID
            thought_nodes[thought_id] = node_id
            thought_type = thought.get("thought_type", "thought").lower()

            # Node shape and style based on thought type
            shapes = {
                "goal": ("([", "])"),
                "question": ("{{", "}}"),
                "decision": ("[/", "\\]"),
                "summary": ("[(", ")]"),
                "constraint": ("[[", "]]"),
                "hypothesis": ("( ", " )"),
            }
            node_shape_start, node_shape_end = shapes.get(
                thought_type, ("[", "]")
            )  # Default rectangle
            node_style = f":::type{thought_type}"

            # Label content
            content = _mermaid_escape(thought.get("content", "..."))
            label = f"<b>{thought_type.capitalize()} #{thought.get('sequence_number')}</b><br/>{content}"

            diagram.append(f'    {node_id}{node_shape_start}"{label}"{node_shape_end}{node_style}')

            # Store parent relationship
            parent_id = thought.get("parent_thought_id")
            if parent_id:
                parent_links[thought_id] = parent_id

    diagram.append("")

    # --- Draw Links ---
    linked_thoughts = set()
    # Parent -> Child links
    for child_id, parent_id in parent_links.items():
        if child_id in thought_nodes and parent_id in thought_nodes:
            child_node = thought_nodes[child_id]
            parent_node = thought_nodes[parent_id]
            diagram.append(f"    {parent_node} --> {child_node}")
            linked_thoughts.add(child_id)

    # Link root thoughts (no parent or parent not found) sequentially from the header
    last_root_node = chain_node_id  # Use sanitized chain node ID
    for thought in thoughts:
        thought_id = thought.get("thought_id")
        if thought_id and thought_id not in linked_thoughts and thought_id in thought_nodes:
            # Check if its parent exists in the fetched thoughts; if not, treat as root for linking
            parent_id = parent_links.get(thought_id)
            if not parent_id or parent_id not in thought_nodes:
                node_id = thought_nodes[thought_id]
                diagram.append(f"    {last_root_node} --> {node_id}")
                last_root_node = node_id  # Chain subsequent roots
                linked_thoughts.add(thought_id)

    # --- External Links (Actions/Artifacts/Memories) ---
    if thoughts:
        diagram.append("")
        for thought in thoughts:
            thought_id = thought.get("thought_id")
            if not thought_id or thought_id not in thought_nodes:
                continue  # Skip if thought or its node wasn't created
            node_id = thought_nodes[thought_id]

            # Link to relevant action
            rel_action_id = thought.get("relevant_action_id")
            if rel_action_id:
                ext_node_id = sanitize_mermaid_id(rel_action_id, "ExtA")  # Sanitize external ID
                diagram.append(f'    {ext_node_id}["Action: {rel_action_id[:8]}..."]:::action')
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

            # Link to relevant artifact
            rel_artifact_id = thought.get("relevant_artifact_id")
            if rel_artifact_id:
                ext_node_id = sanitize_mermaid_id(rel_artifact_id, "ExtF")  # Sanitize external ID
                diagram.append(
                    f'    {ext_node_id}[("Artifact: {rel_artifact_id[:8]}...")]:::artifact'
                )
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

            # Link to relevant memory
            rel_memory_id = thought.get("relevant_memory_id")
            if rel_memory_id:
                ext_node_id = sanitize_mermaid_id(rel_memory_id, "ExtM")  # Sanitize external ID
                diagram.append(f'    {ext_node_id}("Memory: {rel_memory_id[:8]}..."):::memory')
                diagram.append(f"    {node_id} -.-> {ext_node_id}")

    # --- Class Definitions (Full Set) ---
    diagram.append("\n    %% Stylesheets")
    diagram.append(
        "    classDef header fill:#666,stroke:#333,color:#fff,stroke-width:2px,font-weight:bold;"
    )
    # Thought Types
    diagram.append("    classDef typegoal fill:#d4edda,stroke:#155724,color:#155724;")
    diagram.append("    classDef typequestion fill:#cce5ff,stroke:#004085,color:#004085;")
    diagram.append("    classDef typehypothesis fill:#e2e3e5,stroke:#383d41,color:#383d41;")
    diagram.append("    classDef typeinference fill:#fff3cd,stroke:#856404,color:#856404;")
    diagram.append("    classDef typeevidence fill:#d1ecf1,stroke:#0c5460,color:#0c5460;")
    diagram.append("    classDef typeconstraint fill:#f8d7da,stroke:#721c24,color:#721c24;")
    diagram.append("    classDef typeplan fill:#d6d8f8,stroke:#3f4d9a,color:#3f4d9a;")
    diagram.append(
        "    classDef typedecision fill:#ffe6f5,stroke:#97114c,color:#97114c,font-weight:bold;"
    )
    diagram.append("    classDef typereflection fill:#f5f5f5,stroke:#5a5a5a,color:#5a5a5a;")
    diagram.append("    classDef typecritique fill:#feeed8,stroke:#a34e00,color:#a34e00;")
    diagram.append("    classDef typesummary fill:#cfe2ff,stroke:#0a3492,color:#0a3492;")
    # External Links
    diagram.append(
        "    classDef action fill:#f9f2f4,stroke:#c7254e,color:#c7254e,stroke-dasharray: 5 5;"
    )
    diagram.append(
        "    classDef artifact fill:#f3f6f9,stroke:#367fa9,color:#367fa9,stroke-dasharray: 5 5;"
    )
    diagram.append("    classDef memory fill:#f0f0f0,stroke:#777,color:#333,stroke-dasharray: 2 2;")

    diagram.append("```")
    return "\n".join(diagram)


# --- 19. Reporting (Corrected Port from agent_memory) ---
@with_tool_metrics
@with_error_handling
async def generate_workflow_report(
    workflow_id: str,
    report_format: str = "markdown",  # markdown, html, json, mermaid
    include_details: bool = True,
    include_thoughts: bool = True,
    include_artifacts: bool = True,
    style: Optional[str] = "professional",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Generates a comprehensive report for a specific workflow in various formats and styles.

    Creates a human-readable summary or detailed log of a workflow's progress, actions,
    artifacts, and reasoning. For HTML reports, includes CSS for code syntax highlighting
    if the `pygments` library is installed.

    Args:
        workflow_id: The ID of the workflow to report on.
        report_format: (Optional) Output format: 'markdown', 'html', 'json', or 'mermaid'. Default 'markdown'.
        include_details: (Optional) Whether to include detailed sections like reasoning, arguments, results. Default True.
        include_thoughts: (Optional) Whether to include thought chain details. Default True.
        include_artifacts: (Optional) Whether to include artifact details. Default True.
        style: (Optional) Reporting style for Markdown/HTML: 'professional', 'concise',
               'narrative', or 'technical'. Default 'professional'. Ignored for JSON/Mermaid.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        A dictionary containing the generated report and metadata.
        {
            "workflow_id": "workflow-uuid",
            "title": "Workflow Title",
            "report": "Generated report content...",
            "format": "markdown",
            "style_used": "professional", # Included for clarity
            "generated_at": "iso-timestampZ",
            "success": true,
            "processing_time": 0.45
        }

    Raises:
        ToolInputError: If workflow not found or parameters are invalid.
        ToolError: If report generation fails.
    """
    # --- Input Validation ---
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    valid_formats = ["markdown", "html", "json", "mermaid"]
    report_format_lower = report_format.lower()
    if report_format_lower not in valid_formats:
        raise ToolInputError(
            f"Invalid format '{report_format}'. Must be one of: {valid_formats}",
            param_name="report_format",
        )

    valid_styles = ["professional", "concise", "narrative", "technical"]
    style_lower = (style or "professional").lower()  # Default to professional if None
    if style_lower not in valid_styles:
        raise ToolInputError(
            f"Invalid style '{style}'. Must be one of: {valid_styles}", param_name="style"
        )

    start_time = time.time()

    try:
        # --- Fetch Workflow Data ---
        # Fetch all potentially needed data, filtering happens during report generation
        workflow_data = await get_workflow_details(
            workflow_id=workflow_id,
            include_actions=True,  # Fetch actions for all report types/styles
            include_artifacts=include_artifacts,  # Fetch if requested
            include_thoughts=include_thoughts,  # Fetch if requested
            include_memories=False,  # Keep reports focused on tracked items for now
            db_path=db_path,
        )
        # get_workflow_details should raise ToolInputError if workflow not found
        if not workflow_data.get("success"):  # Check just in case error handling fails
            raise ToolError(
                f"Failed to retrieve workflow details for report generation (ID: {workflow_id})."
            )

        # --- Generate Report Content ---
        report_content = None
        markdown_report_content = None  # To store markdown for HTML conversion

        if report_format_lower == "markdown" or report_format_lower == "html":
            # Select the appropriate Markdown generation helper based on style
            if style_lower == "concise":
                markdown_report_content = await _generate_concise_report(
                    workflow_data, include_details
                )
            elif style_lower == "narrative":
                markdown_report_content = await _generate_narrative_report(
                    workflow_data, include_details
                )
            elif style_lower == "technical":
                markdown_report_content = await _generate_technical_report(
                    workflow_data, include_details
                )
            else:  # Default to professional
                markdown_report_content = await _generate_professional_report(
                    workflow_data, include_details
                )

            if report_format_lower == "markdown":
                report_content = markdown_report_content
            else:  # HTML format
                try:
                    # Convert markdown to HTML body content
                    html_body = markdown.markdown(
                        markdown_report_content, extensions=["tables", "fenced_code", "codehilite"]
                    )

                    pygments_css = ""
                    try:
                        # Use the default pygments CSS style
                        formatter = HtmlFormatter(style="default")
                        pygments_css = f"<style>{formatter.get_style_defs('.codehilite')}</style>"
                    except Exception as css_err:
                        logger.warning(f"Failed to generate Pygments CSS: {css_err}")
                        pygments_css = "<!-- Pygments CSS generation failed -->"

                    # Construct the full HTML document
                    report_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Workflow Report: {workflow_data.get("title", "Untitled")}</title>
    {pygments_css}
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        pre code {{ display: block; padding: 10px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; }}
        /* Basic codehilite styles if pygments fails */
        .codehilite pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
                except Exception as md_err:
                    logger.error(f"Markdown to HTML conversion failed: {md_err}", exc_info=True)
                    raise ToolError(f"Failed to convert report to HTML: {md_err}") from md_err

        elif report_format_lower == "json":
            # Return the structured data directly, cleaned up
            clean_data = {
                k: v for k, v in workflow_data.items() if k not in ["success", "processing_time"]
            }  # Remove tool metadata
            try:
                report_content = json.dumps(clean_data, indent=2, ensure_ascii=False)
            except Exception as json_err:
                logger.error(f"JSON serialization failed for report: {json_err}", exc_info=True)
                raise ToolError(
                    f"Failed to serialize workflow data to JSON: {json_err}"
                ) from json_err

        elif report_format_lower == "mermaid":
            # Generate the Mermaid diagram string
            report_content = await _generate_mermaid_diagram(workflow_data)

        # Final check if content generation failed unexpectedly
        if report_content is None:
            raise ToolError(
                f"Report content generation failed unexpectedly for format '{report_format_lower}' and style '{style_lower}'."
            )

        # --- Prepare Result ---
        result = {
            "workflow_id": workflow_id,
            "title": workflow_data.get("title", "Workflow Report"),
            "report": report_content,
            "format": report_format_lower,
            "style_used": style_lower
            if report_format_lower in ["markdown", "html"]
            else None,  # Indicate style used
            "generated_at": to_iso_z(
                datetime.now(timezone.utc).timestamp()
            ),  # Use helper for consistency
            "success": True,
            "processing_time": time.time() - start_time,
        }
        logger.info(
            f"Generated {report_format_lower} report (style: {style_lower if report_format_lower in ['markdown', 'html'] else 'N/A'}) for workflow {workflow_id}",
            emoji_key="newspaper",
        )
        return result

    except (ToolInputError, ToolError):
        raise  # Re-raise specific handled errors
    except Exception as e:
        logger.error(f"Unexpected error generating report for {workflow_id}: {e}", exc_info=True)
        raise ToolError(
            f"Failed to generate workflow report due to an unexpected error: {str(e)}"
        ) from e


# --- 20. Visualization ---
@with_tool_metrics
@with_error_handling
async def visualize_reasoning_chain(
    thought_chain_id: str,
    output_format: str = "mermaid",  # mermaid, json
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Generates a visualization of a specific thought chain."""
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")
    valid_formats = ["mermaid", "json"]
    if output_format.lower() not in valid_formats:
        raise ToolInputError(
            f"Invalid format. Use one of: {valid_formats}", param_name="output_format"
        )
    output_format = output_format.lower()
    start_time = time.time()

    try:
        # Fetch the thought chain data
        thought_chain_data = await get_thought_chain(thought_chain_id, db_path=db_path)
        if not thought_chain_data.get("success"):
            raise ToolError(
                f"Failed to retrieve thought chain {thought_chain_id} for visualization."
            )

        visualization_content = None
        # Generate visualization content
        if output_format == "mermaid":
            visualization_content = await _generate_thought_chain_mermaid(thought_chain_data)
        elif output_format == "json":
            # Create hierarchical JSON structure
            structured_chain = {
                k: v for k, v in thought_chain_data.items() if k not in ["success", "thoughts"]
            }
            child_map = defaultdict(list)
            all_thoughts = thought_chain_data.get("thoughts", [])
            for thought in all_thoughts:
                child_map[thought.get("parent_thought_id")].append(thought)

            def build_tree(thought_list):
                tree = []
                for thought in thought_list:
                    node = dict(thought)
                    children = child_map.get(thought["thought_id"])
                    if children:
                        node["children"] = build_tree(children)
                    tree.append(node)
                return tree

            structured_chain["thought_tree"] = build_tree(
                child_map.get(None, [])
            )  # Start with roots (None parent)
            visualization_content = json.dumps(structured_chain, indent=2)

        if visualization_content is None:
            raise ToolError(
                f"Failed to generate visualization content for format '{output_format}'."
            )

        result = {
            "thought_chain_id": thought_chain_id,
            "title": thought_chain_data.get("title", "Thought Chain"),
            "visualization": visualization_content,
            "format": output_format,
            "success": True,
            "processing_time": time.time() - start_time,
        }
        logger.info(
            f"Generated {output_format} visualization for thought chain {thought_chain_id}",
            emoji_key="projector",
        )
        return result

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error visualizing thought chain {thought_chain_id}: {e}", exc_info=True)
        raise ToolError(f"Failed to visualize thought chain: {str(e)}") from e



async def _generate_professional_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a professional-style report with formal structure and comprehensive details."""
    report_lines = [f"# Workflow Report: {workflow.get('title', 'Untitled Workflow')}"]

    # --- Executive Summary ---
    report_lines.append("\n## Executive Summary\n")
    report_lines.append(f"**Status:** {workflow.get('status', 'N/A').capitalize()}")
    if workflow.get("goal"):
        report_lines.append(f"**Goal:** {workflow['goal']}")
    if workflow.get("description"):
        report_lines.append(f"\n{workflow['description']}")

    # Use safe_format_timestamp with the correct keys
    report_lines.append(f"\n**Created:** {safe_format_timestamp(workflow.get('created_at'))}")
    report_lines.append(f"**Last Updated:** {safe_format_timestamp(workflow.get('updated_at'))}")
    if workflow.get('completed_at'):
        report_lines.append(f"**Completed:** {safe_format_timestamp(workflow.get('completed_at'))}")
    if workflow.get("tags"):
        report_lines.append(f"**Tags:** {', '.join(workflow['tags'])}")

    # --- Progress Overview ---
    actions = workflow.get("actions", [])
    if actions:
        total_actions = len(actions)
        completed_actions_count = sum(
            1 for a in actions if a.get("status") == ActionStatus.COMPLETED.value
        )
        completion_percentage = (
            int((completed_actions_count / total_actions) * 100) if total_actions > 0 else 0
        )
        report_lines.append("\n## Progress Overview\n")
        report_lines.append(
            f"Overall completion: **{completion_percentage}%** ({completed_actions_count}/{total_actions} actions completed)"
        )
        bar_filled = "#" * (completion_percentage // 5)
        bar_empty = " " * (20 - (completion_percentage // 5))
        report_lines.append(f"\n```\n[{bar_filled}{bar_empty}] {completion_percentage}%\n```")

    # --- Key Actions and Steps ---
    if actions and include_details:
        report_lines.append("\n## Key Actions and Steps\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float('inf')))
        for i, action in enumerate(sorted_actions):
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "in_progress": "⏳",
                "planned": "🗓️",
            }.get(action.get("status"), "❓")
            title = action.get("title", action.get("action_type", "Action")).strip()
            report_lines.append(f"### {i + 1}. {status_emoji} {title}\n")
            report_lines.append(f"**Action ID:** `{action.get('action_id')}`")
            report_lines.append(f"**Type:** {action.get('action_type', 'N/A').capitalize()}")
            report_lines.append(f"**Status:** {action.get('status', 'N/A').capitalize()}")
            report_lines.append(f"**Started:** {safe_format_timestamp(action.get('started_at'))}")
            if action.get("completed_at"):
                report_lines.append(f"**Completed:** {safe_format_timestamp(action['completed_at'])}")

            if action.get("reasoning"):
                report_lines.append(f"\n**Reasoning:**\n```\n{action['reasoning']}\n```")
            if action.get("tool_name"):
                report_lines.append(f"\n**Tool Used:** `{action['tool_name']}`")
                # tool_args might already be deserialized by get_workflow_details
                tool_args = action.get("tool_args")
                if tool_args:
                    try:
                        # Attempt to format as JSON if it's dict/list
                        if isinstance(tool_args, (dict, list)):
                            args_str = json.dumps(tool_args, indent=2)
                            lang = "json"
                        else:
                            args_str = str(tool_args)
                            lang = ""
                    except Exception: # Catch potential errors during dump
                        args_str = str(tool_args)
                        lang = ""
                    report_lines.append(f"**Arguments:**\n```{lang}\n{args_str}\n```")

                # tool_result might already be deserialized by get_workflow_details
                tool_result = action.get("tool_result")
                if tool_result is not None: # Check for None explicitly
                    result_repr = tool_result
                    try:
                         # Attempt to format as JSON if it's dict/list
                        if isinstance(result_repr, (dict, list)):
                            result_str = json.dumps(result_repr, indent=2)
                            lang = "json"
                        else:
                            result_str = str(result_repr)
                            lang = ""
                    except Exception: # Catch potential errors during dump
                        result_str = str(result_repr)
                        lang = ""

                    if len(result_str) > 500:
                        result_str = result_str[:497] + "..."
                    report_lines.append(f"**Result Preview:**\n```{lang}\n{result_str}\n```")

            if action.get("tags"):
                report_lines.append(f"**Tags:** {', '.join(action['tags'])}")
            report_lines.append("\n---")  # Separator

    # --- Key Findings & Insights (from Thoughts) ---
    thought_chains = workflow.get("thought_chains", [])
    if thought_chains and include_details:
        report_lines.append("\n## Key Findings & Insights (from Reasoning)\n")
        for i, chain in enumerate(thought_chains):
            report_lines.append(f"### Reasoning Chain {i + 1}: {chain.get('title', 'Untitled')}\n")
             # Ensure sorting key exists or provide default
            thoughts = sorted(chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float('inf')))
            if not thoughts:
                report_lines.append("_No thoughts recorded in this chain._")
            else:
                for thought in thoughts:
                    is_key_thought = thought.get("thought_type") in [
                        "goal",
                        "decision",
                        "summary",
                        "hypothesis",
                        "inference",
                        "reflection",
                        "critique",
                    ]
                    prefix = "**" if is_key_thought else ""
                    suffix = "**" if is_key_thought else ""
                    type_label = thought.get("thought_type", "Thought").capitalize()
                    # Use safe_format_timestamp for thought timestamps
                    thought_time = safe_format_timestamp(thought.get("created_at"))
                    report_lines.append(
                        f"- {prefix}{type_label}{suffix} ({thought_time}): {thought.get('content', '')}"
                    )
                    links = []
                    if thought.get("relevant_action_id"):
                        links.append(f"Action `{thought['relevant_action_id'][:8]}`")
                    if thought.get("relevant_artifact_id"):
                        links.append(f"Artifact `{thought['relevant_artifact_id'][:8]}`")
                    if thought.get("relevant_memory_id"):
                        links.append(f"Memory `{thought['relevant_memory_id'][:8]}`")
                    if links:
                        report_lines.append(f"  *Related to:* {', '.join(links)}")
            report_lines.append("")

    # --- Artifacts & Outputs ---
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Artifacts & Outputs\n")
        report_lines.append(
            "| Name | Type | Description | Path/Preview | Created | Tags | Output? |"
        )
        report_lines.append(
            "| ---- | ---- | ----------- | ------------ | ------- | ---- | ------- |"
        )
        for artifact in artifacts:
            name = artifact.get("name", "N/A")
            atype = artifact.get("artifact_type", "N/A")
            desc = (artifact.get("description", "") or "")[:50]
            path_or_preview = artifact.get("path", "") or (
                artifact.get("content_preview", "") or ""
            )
            path_or_preview = (
                f"`{path_or_preview}`" if artifact.get("path") else path_or_preview[:60]
            )
            # Use safe_format_timestamp for artifact timestamps
            created_time = safe_format_timestamp(artifact.get("created_at"))
            tags = ", ".join(artifact.get("tags", []))
            is_output = "Yes" if artifact.get("is_output") else "No"
            report_lines.append(
                f"| {name} | {atype} | {desc} | {path_or_preview} | {created_time} | {tags} | {is_output} |"
            )

    # --- Conclusion / Next Steps ---
    report_lines.append("\n## Conclusion & Next Steps\n")
    status = workflow.get("status", "N/A")
    if status == WorkflowStatus.COMPLETED.value:
        report_lines.append("Workflow marked as **Completed**.")
    elif status == WorkflowStatus.FAILED.value:
        report_lines.append("Workflow marked as **Failed**.")
    elif status == WorkflowStatus.ABANDONED.value:
        report_lines.append("Workflow marked as **Abandoned**.")
    elif status == WorkflowStatus.PAUSED.value:
        report_lines.append("Workflow is currently **Paused**.")
    else:  # Active
        report_lines.append("Workflow is **Active**. Potential next steps include:")
        last_action = (
            sorted(actions, key=lambda a: a.get("sequence_number", float('inf')))[-1] if actions else None
        )
        if last_action and last_action.get("status") == ActionStatus.IN_PROGRESS.value:
            report_lines.append(f"- Completing action: '{last_action.get('title', 'Last Action')}'")
        elif last_action:
            report_lines.append(
                f"- Planning the next action after '{last_action.get('title', 'Last Action')}'"
            )
        else:
            report_lines.append("- Defining the initial actions for the workflow goal.")

    # Footer
    report_lines.append(
        "\n---\n*Report generated on " + safe_format_timestamp(datetime.now(timezone.utc).timestamp()) + "*"
    )
    return "\n".join(report_lines)


async def _generate_concise_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a concise report focusing on key information."""
    report_lines = [
        f"# {workflow.get('title', 'Untitled Workflow')} (`{workflow.get('workflow_id', '')[:8]}`)"
    ]
    report_lines.append(f"**Status:** {workflow.get('status', 'N/A').capitalize()}")
    if workflow.get("goal"):
        report_lines.append(f"**Goal:** {workflow.get('goal', '')[:100]}...")

    actions = workflow.get("actions", [])
    if actions:
        total = len(actions)
        completed = sum(1 for a in actions if a.get("status") == ActionStatus.COMPLETED.value)
        perc = int((completed / total) * 100) if total > 0 else 0
        report_lines.append(f"**Progress:** {perc}% ({completed}/{total} actions)")

    # Recent/Current Actions
    if actions:
        report_lines.append("\n**Recent Activity:**")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float('inf')), reverse=True)
        for action in sorted_actions[:3]:  # Show top 3 recent
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "in_progress": "⏳",
                "planned": "🗓️",
            }.get(action.get("status"), "❓")
            report_lines.append(f"- {status_emoji} {action.get('title', 'Action')[:50]}")

    # Outputs
    artifacts = workflow.get("artifacts", [])
    outputs = [a for a in artifacts if a.get("is_output")]
    if outputs:
        report_lines.append("\n**Outputs:**")
        for output in outputs[:5]:  # Limit outputs listed
            report_lines.append(
                f"- {output.get('name', 'N/A')} (`{output.get('artifact_type', 'N/A')}`)"
            )

    return "\n".join(report_lines)


async def _generate_narrative_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a narrative-style report as a story."""
    report_lines = [f"# The Journey of: {workflow.get('title', 'Untitled Workflow')}"]

    # Introduction
    report_lines.append("\n## Our Quest Begins\n")
    # Use safe_format_timestamp with the correct key
    start_time = safe_format_timestamp(workflow.get("created_at"))
    if workflow.get("goal"):
        report_lines.append(
            f"We embarked on a mission around {start_time}: **{workflow['goal']}**."
        )
    else:
        report_lines.append(
            f"Our story started on {start_time}, aiming to understand or create '{workflow.get('title', 'something interesting')}'"
        )
    if workflow.get("description"):
        report_lines.append(f"> {workflow['description']}\n")

    # The Path
    actions = workflow.get("actions", [])
    if actions:
        report_lines.append("## The Path Unfolds\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float('inf')))
        for action in sorted_actions:
            title = action.get("title", action.get("action_type", "A step"))
            # Use safe_format_timestamp for action start time
            start_time_action = safe_format_timestamp(action.get("started_at"))
            if action.get("status") == ActionStatus.COMPLETED.value:
                report_lines.append(
                    f"Then, around {start_time_action}, we successfully **{title}**."
                )
                if include_details and action.get("reasoning"):
                    report_lines.append(f"  *Our reasoning was: {action['reasoning'][:150]}...*")
            elif action.get("status") == ActionStatus.FAILED.value:
                report_lines.append(
                    f"Around {start_time_action}, we encountered trouble when trying to **{title}**."
                )
            elif action.get("status") == ActionStatus.IN_PROGRESS.value:
                report_lines.append(
                    f"Starting around {start_time_action}, we are working on **{title}**."
                )
            # Add other statuses if needed (skipped, planned)
            elif action.get("status") == ActionStatus.SKIPPED.value:
                 report_lines.append(f"Around {start_time_action}, we decided to skip the step: **{title}**.")
            elif action.get("status") == ActionStatus.PLANNED.value:
                 report_lines.append(f"The plan included the step: **{title}** (not yet started).")
            report_lines.append("") # Add spacing between actions

    # Discoveries
    thoughts = [
        t for chain in workflow.get("thought_chains", []) for t in chain.get("thoughts", [])
    ]
    key_thoughts = [
        t
        for t in thoughts
        if t.get("thought_type") in ["decision", "insight", "hypothesis", "summary", "reflection"]
    ]
    if key_thoughts and include_details:
        report_lines.append("## Moments of Clarity\n")
        # Ensure sorting key exists or provide default
        sorted_thoughts = sorted(key_thoughts, key=lambda t: t.get("sequence_number", float('inf')))
        for thought in sorted_thoughts[:7]:
            # Use safe_format_timestamp for thought timestamp
            thought_time = safe_format_timestamp(thought.get("created_at"))
            report_lines.append(
                f"- Around {thought_time}, a key **{thought.get('thought_type')}** emerged: *{thought.get('content', '')[:150]}...*"
            )

    # Treasures
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Treasures Found\n")
        outputs = [a for a in artifacts if a.get("is_output")]
        other_artifacts = [a for a in artifacts if not a.get("is_output")]
        # Ensure sorting key exists or provide default, sort by creation time
        outputs.sort(key=lambda a: a.get("created_at", 0))
        other_artifacts.sort(key=lambda a: a.get("created_at", 0))
        # Combine lists for display, respecting limits
        display_artifacts = outputs[:3] + other_artifacts[: max(0, 5 - len(outputs))]
        display_artifacts.sort(key=lambda a: a.get("created_at", 0)) # Sort combined list

        for artifact in display_artifacts:
            marker = "🏆 Final Result:" if artifact.get("is_output") else "📌 Item Created:"
            # Use safe_format_timestamp for artifact timestamp
            artifact_time = safe_format_timestamp(artifact.get("created_at"))
            report_lines.append(
                f"- {marker} Around {artifact_time}, **{artifact.get('name')}** ({artifact.get('artifact_type')}) was produced."
            )
            if artifact.get("description"):
                report_lines.append(f"  *{artifact['description'][:100]}...*")

    # Current Status/Ending
    status = workflow.get("status", "active")
    report_lines.append(
        f"\n## {"Journey's End" if status == 'completed' else 'The Story So Far...'}\n"
    )
    if status == WorkflowStatus.COMPLETED.value:
        report_lines.append("Our quest is complete! We achieved our objectives.")
    elif status == WorkflowStatus.FAILED.value:
        report_lines.append(
            "Alas, this chapter ends here, marked by challenges we could not overcome."
        )
    elif status == WorkflowStatus.ABANDONED.value:
        report_lines.append("We chose to leave this path, perhaps to return another day.")
    elif status == WorkflowStatus.PAUSED.value:
        report_lines.append("We pause here, taking stock before continuing the adventure.")
    else:
        report_lines.append("The journey continues...")

    # Footer timestamp formatting
    report_lines.append("\n---\n*Narrative recorded on " + safe_format_timestamp(datetime.now(timezone.utc).timestamp()) + "*")
    return "\n".join(report_lines)


async def _generate_technical_report(workflow: Dict[str, Any], include_details: bool) -> str:
    """Generates a technical report with data-oriented structure."""
    report_lines = [f"# Technical Report: {workflow.get('title', 'Untitled Workflow')}"]

    # --- Metadata ---
    report_lines.append("\n## Workflow Metadata\n```yaml")
    report_lines.append(f"workflow_id: {workflow.get('workflow_id')}")
    report_lines.append(f"title: {workflow.get('title')}")
    report_lines.append(f"status: {workflow.get('status')}")
    report_lines.append(f"goal: {workflow.get('goal') or 'N/A'}")
    # Use safe_format_timestamp for workflow timestamps
    report_lines.append(f"created_at: {safe_format_timestamp(workflow.get('created_at'))}")
    report_lines.append(f"updated_at: {safe_format_timestamp(workflow.get('updated_at'))}")
    if workflow.get("completed_at"):
        report_lines.append(f"completed_at: {safe_format_timestamp(workflow.get('completed_at'))}")
    if workflow.get("tags"):
        report_lines.append(f"tags: {workflow['tags']}")
    report_lines.append("```")

    # --- Metrics ---
    actions = workflow.get("actions", [])
    if actions:
        report_lines.append("\n## Execution Metrics\n")
        total = len(actions)
        counts = defaultdict(int)
        for a in actions:
            counts[a.get("status", "unknown")] += 1
        report_lines.append("**Action Status Counts:**")
        for status, count in counts.items():
            report_lines.append(f"- {status.capitalize()}: {count} ({int(count / total * 100)}%)")
        type_counts = defaultdict(int)
        for a in actions:
            type_counts[a.get("action_type", "unknown")] += 1
        report_lines.append("\n**Action Type Counts:**")
        for atype, count in type_counts.items():
            report_lines.append(f"- {atype.capitalize()}: {count} ({int(count / total * 100)}%)")

    # --- Action Log ---
    if actions and include_details:
        report_lines.append("\n## Action Log\n")
        # Ensure sorting key exists or provide default
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float('inf')))
        for action in sorted_actions:
            report_lines.append(f"### Action Sequence: {action.get('sequence_number')}\n```yaml")
            report_lines.append(f"action_id: {action.get('action_id')}")
            report_lines.append(f"title: {action.get('title')}")
            report_lines.append(f"type: {action.get('action_type')}")
            report_lines.append(f"status: {action.get('status')}")
            # Use safe_format_timestamp for action timestamps
            report_lines.append(f"started_at: {safe_format_timestamp(action.get('started_at'))}")
            if action.get("completed_at"):
                report_lines.append(f"completed_at: {safe_format_timestamp(action.get('completed_at'))}")
            if action.get("tool_name"):
                report_lines.append(f"tool_name: {action['tool_name']}")
            # Use the already deserialized data if present
            tool_args_repr = str(action.get('tool_args', 'N/A'))
            tool_result_repr = str(action.get('tool_result', 'N/A'))
            report_lines.append(f"tool_args_preview: {tool_args_repr[:100]}...")
            report_lines.append(f"tool_result_preview: {tool_result_repr[:100]}...")
            report_lines.append("```")
            if action.get("reasoning"):
                report_lines.append(f"**Reasoning:**\n```\n{action['reasoning']}\n```")

    # --- Artifacts ---
    artifacts = workflow.get("artifacts", [])
    if artifacts and include_details:
        report_lines.append("\n## Artifacts\n```json")
        artifact_list_repr = []
        for artifact in artifacts:
            repr_dict = {
                k: artifact.get(k)
                for k in [
                    "artifact_id",
                    "name",
                    "artifact_type",
                    "description",
                    "path",
                    "is_output",
                    "tags",
                    "created_at",
                ]
            }
            # Format timestamp safely
            if "created_at" in repr_dict:
                repr_dict["created_at"] = safe_format_timestamp(repr_dict["created_at"])
            artifact_list_repr.append(repr_dict)
        # Use default=str for safe JSON dumping
        report_lines.append(json.dumps(artifact_list_repr, indent=2, default=str))
        report_lines.append("```")

    # --- Thoughts ---
    thought_chains = workflow.get("thought_chains", [])
    if thought_chains and include_details:
        report_lines.append("\n## Thought Chains\n")
        for chain in thought_chains:
            report_lines.append(
                f"### Chain: {chain.get('title')} (`{chain.get('thought_chain_id')}`)\n```json"
            )
            # Ensure sorting key exists or provide default
            thoughts = sorted(chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float('inf')))
            formatted_thoughts = []
            for thought in thoughts:
                fmt_thought = dict(thought)
                # Format timestamp safely
                if fmt_thought.get("created_at"):
                    fmt_thought["created_at"] = safe_format_timestamp(fmt_thought["created_at"])
                formatted_thoughts.append(fmt_thought)
            # Use default=str for safe JSON dumping
            report_lines.append(json.dumps(formatted_thoughts, indent=2, default=str))
            report_lines.append("```")

    return "\n".join(report_lines)


async def _generate_memory_network_mermaid(
    memories: List[Dict], links: List[Dict], center_memory_id: Optional[str] = None
) -> str:
    """Helper function to generate Mermaid graph syntax for a memory network."""

    def sanitize_mermaid_id(uuid_str: Optional[str], prefix: str) -> str:
        """Creates a valid Mermaid node ID from a UUID, handling None."""
        if not uuid_str:
            # Generate a unique fallback for missing IDs to avoid collisions
            # Ensure MemoryUtils is available if needed
            # return f"{prefix}_MISSING_{MemoryUtils.generate_id().replace('-', '_')}"
             return f"{prefix}_MISSING_{str(uuid.uuid4()).replace('-', '_')}" # Use uuid directly
        # Replace hyphens which are problematic in unquoted Mermaid node IDs
        sanitized = uuid_str.replace("-", "_")
        return f"{prefix}_{sanitized}"

    diagram = ["```mermaid", "graph TD"]  # Top-Down graph direction

    # Node Definitions
    diagram.append("\n    %% Memory Nodes")
    memory_id_to_node_id = {}  # Map full memory ID to sanitized Mermaid node ID
    for memory in memories:
        mem_id = memory.get("memory_id")
        if not mem_id:
            continue

        node_id = sanitize_mermaid_id(mem_id, "M")  # Use sanitized full ID
        memory_id_to_node_id[mem_id] = node_id  # Store mapping

        # Label content: Type, Description (truncated), Importance
        mem_type = memory.get("memory_type", "memory").capitalize()
        # Ensure _mermaid_escape is available
        desc = _mermaid_escape(memory.get("description", mem_id))  # Use full ID if no desc
        if len(desc) > 40:
            desc = desc[:37] + "..."
        importance = memory.get("importance", 5.0)
        label = f"<b>{mem_type}</b><br/>{desc}<br/><i>(I: {importance:.1f})</i>"

        # Node shape/style based on level (e.g., Semantic=rectangle, Episodic=rounded, Procedural=subroutine)
        # Ensure MemoryLevel enum is available
        level = memory.get("memory_level", MemoryLevel.EPISODIC.value)
        shape_start, shape_end = "[", "]"  # Default rectangle (Semantic)
        if level == MemoryLevel.EPISODIC.value:
            shape_start, shape_end = "(", ")"  # Rounded rectangle
        elif level == MemoryLevel.PROCEDURAL.value:
            shape_start, shape_end = "[[", "]]"  # Subroutine shape
        elif level == MemoryLevel.WORKING.value:
            shape_start, shape_end = "([", "])"  # Capsule shape for working memory?

        # Style based on level + highlight center node
        node_style = f":::level{level}"
        if mem_id == center_memory_id:
            node_style += " :::centerNode"  # Add specific style for center

        diagram.append(f'    {node_id}{shape_start}"{label}"{shape_end}{node_style}')

    # Edge Definitions
    diagram.append("\n    %% Memory Links")
    for link in links:
        source_mem_id = link.get("source_memory_id")
        target_mem_id = link.get("target_memory_id")
        link_type = link.get("link_type", "related")

        # Only draw links where both source and target are in the visualized node set
        if source_mem_id in memory_id_to_node_id and target_mem_id in memory_id_to_node_id:
            source_node = memory_id_to_node_id[source_mem_id]
            target_node = memory_id_to_node_id[target_mem_id]
            # Add link type as label, style based on strength? (Keep simple for now)
            diagram.append(f"    {source_node} -- {link_type} --> {target_node}")

    # Class Definitions for Styling
    diagram.append("\n    %% Stylesheets")
    diagram.append(
        "    classDef levelworking fill:#e3f2fd,stroke:#2196f3,color:#1e88e5,stroke-width:1px;"
    )  # Light blue
    diagram.append(
        "    classDef levelepisodic fill:#e8f5e9,stroke:#4caf50,color:#388e3c,stroke-width:1px;"
    )  # Light green
    diagram.append(
        "    classDef levelsemantic fill:#fffde7,stroke:#ffc107,color:#ffa000,stroke-width:1px;"
    )  # Light yellow
    diagram.append(
        "    classDef levelprocedural fill:#fce4ec,stroke:#e91e63,color:#c2185b,stroke-width:1px;"
    )  # Light pink
    diagram.append(
        "    classDef centerNode stroke-width:3px,stroke:#0d47a1,font-weight:bold;"
    )  # Darker blue border for center

    diagram.append("```")
    return "\n".join(diagram)


@with_tool_metrics
@with_error_handling
async def visualize_memory_network(
    workflow_id: Optional[str] = None,
    center_memory_id: Optional[str] = None,
    depth: int = 1,  # How many steps away from the center node to include
    max_nodes: int = 30,  # Max total memory nodes to display
    memory_level: Optional[str] = None,  # Filter nodes by level
    memory_type: Optional[str] = None,  # Filter nodes by type
    output_format: str = "mermaid",  # Only mermaid for now
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Generates a visualization of the memory network for a workflow or around a specific memory.

    Creates a Mermaid graph showing memories as nodes and links between them as edges.
    Can be focused around a central memory or show the most relevant memories in a workflow.

    Args:
        workflow_id: (Optional) ID of the workflow to visualize. Required if center_memory_id is not provided.
        center_memory_id: (Optional) ID of the memory to center the visualization around.
        depth: (Optional) If center_memory_id is provided, how many link steps away to include (e.g., 1=immediate neighbors). Default 1.
        max_nodes: (Optional) Maximum total number of memory nodes to include in the graph. Default 30.
        memory_level: (Optional) Filter included memories by level (applied after neighbor selection if centered).
        memory_type: (Optional) Filter included memories by type (applied after neighbor selection if centered).
        output_format: (Optional) Currently only supports 'mermaid'. Default 'mermaid'.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        Dictionary containing the Mermaid visualization string.
        {
            "workflow_id": "workflow-uuid" | None,
            "center_memory_id": "center-uuid" | None,
            "visualization": "```mermaid\ngraph TD\n...",
            "node_count": 25,
            "link_count": 35,
            "format": "mermaid",
            "success": true,
            "processing_time": 0.25
        }

    Raises:
        ToolInputError: If required parameters are missing, invalid, or entities not found.
        ToolError: If database or visualization generation fails.
    """
    if not workflow_id and not center_memory_id:
        raise ToolInputError(
            "Either workflow_id or center_memory_id must be provided.", param_name="workflow_id"
        )
    if output_format.lower() != "mermaid":
        raise ToolInputError(
            "Currently only 'mermaid' format is supported.", param_name="output_format"
        )
    if depth < 0:
        raise ToolInputError("Depth cannot be negative.", param_name="depth")
    if max_nodes <= 0:
        raise ToolInputError("Max nodes must be positive.", param_name="max_nodes")

    start_time = time.time()
    selected_memory_ids = set()  # Keep track of memory IDs to include
    memories_data = {}  # Store fetched memory details {mem_id: data}
    links_data = []  # Store fetched links between selected memories

    try:
        async with DBConnection(db_path) as conn:
            # --- 1. Fetch Initial Set of Memory IDs ---
            target_workflow_id = workflow_id  # Use workflow_id if provided

            if center_memory_id:
                # Fetch center memory to get its workflow_id if not provided
                async with conn.execute(
                    "SELECT workflow_id FROM memories WHERE memory_id = ?", (center_memory_id,)
                ) as cursor:
                    center_row = await cursor.fetchone()
                    if not center_row:
                        raise ToolInputError(
                            f"Center memory {center_memory_id} not found.",
                            param_name="center_memory_id",
                        )
                    if not target_workflow_id:
                        target_workflow_id = center_row["workflow_id"]
                    elif target_workflow_id != center_row["workflow_id"]:
                        raise ToolInputError(
                            f"Center memory {center_memory_id} does not belong to workflow {target_workflow_id}.",
                            param_name="center_memory_id",
                        )

                # BFS-like approach to get neighbors up to depth
                queue = {center_memory_id}
                visited = set()
                for current_depth in range(
                    depth + 1
                ):  # Iterate depth + 1 times (depth 0 is the center node itself)
                    if len(selected_memory_ids) >= max_nodes:
                        break  # Stop if max nodes reached

                    next_queue = set()
                    ids_to_query = list(queue - visited)  # Process only new nodes at this level
                    if not ids_to_query:
                        break  # No new nodes to explore

                    visited.update(ids_to_query)

                    # Add nodes found at this level to the final selection
                    for node_id in ids_to_query:
                        if len(selected_memory_ids) < max_nodes:
                            selected_memory_ids.add(node_id)
                        else:
                            break  # Hit max nodes limit

                    if (
                        current_depth < depth and len(selected_memory_ids) < max_nodes
                    ):  # Don't fetch neighbors on the last level or if max nodes hit
                        # Fetch direct neighbors (both incoming and outgoing) for the next level
                        placeholders = ", ".join(["?"] * len(ids_to_query))
                        neighbor_query = f"""
                            SELECT target_memory_id as neighbor_id FROM memory_links WHERE source_memory_id IN ({placeholders})
                            UNION
                            SELECT source_memory_id as neighbor_id FROM memory_links WHERE target_memory_id IN ({placeholders})
                        """
                        async with conn.execute(neighbor_query, ids_to_query * 2) as cursor:
                            async for row in cursor:
                                if row["neighbor_id"] not in visited:
                                    next_queue.add(row["neighbor_id"])

                    queue = next_queue  # Move to the next level

            else:  # No center node, get top memories based on filters/sorting
                if not target_workflow_id:  # Should have been caught earlier, but safeguard
                    raise ToolInputError(
                        "Workflow ID is required when not specifying a center memory.",
                        param_name="workflow_id",
                    )

                filter_where = ["workflow_id = ?"]
                filter_params: List[Any] = [target_workflow_id]
                if memory_level:
                    filter_where.append("memory_level = ?")
                    filter_params.append(memory_level.lower())
                if memory_type:
                    filter_where.append("memory_type = ?")
                    filter_params.append(memory_type.lower())

                # Add TTL check
                now_unix = int(time.time())
                filter_where.append("(ttl = 0 OR created_at + ttl > ?)")
                filter_params.append(now_unix)

                where_sql = " AND ".join(filter_where)
                # Fetch most important/relevant first
                query = f"""
                    SELECT memory_id
                    FROM memories
                    WHERE {where_sql}
                    ORDER BY compute_memory_relevance(importance, confidence, created_at, access_count, last_accessed) DESC
                    LIMIT ?
                """
                filter_params.append(max_nodes)
                async with conn.execute(query, filter_params) as cursor:
                    selected_memory_ids = {row["memory_id"] for row in await cursor.fetchall()}

            if not selected_memory_ids:
                logger.info("No memories selected for visualization based on criteria.")
                return {  # Return empty graph structure
                    "workflow_id": target_workflow_id,
                    "center_memory_id": center_memory_id,
                    "visualization": "```mermaid\ngraph TD\n    NoNodes[No memories found]\n```",
                    "node_count": 0,
                    "link_count": 0,
                    "format": "mermaid",
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # --- 2. Fetch Details for Selected Memories ---
            # Apply level/type filters *now* if a center_memory_id was used
            # (We needed neighbors first, now filter the selected set)
            fetch_ids = list(selected_memory_ids)
            placeholders = ", ".join(["?"] * len(fetch_ids))
            details_query = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
            details_params = fetch_ids

            final_selected_ids = set()  # Rebuild the set after filtering

            async with conn.execute(details_query, details_params) as cursor:
                async for row in cursor:
                    mem_data = dict(row)
                    # Apply post-filters if centering was used
                    if center_memory_id:
                        passes_filter = True
                        if memory_level and mem_data.get("memory_level") != memory_level.lower():
                            passes_filter = False
                        if memory_type and mem_data.get("memory_type") != memory_type.lower():
                            passes_filter = False

                        # Keep the center node even if it doesn't match filters
                        if passes_filter or mem_data["memory_id"] == center_memory_id:
                            memories_data[mem_data["memory_id"]] = mem_data
                            final_selected_ids.add(mem_data["memory_id"])
                    else:  # If not centering, filters were applied in the initial query
                        memories_data[mem_data["memory_id"]] = mem_data
                        final_selected_ids.add(mem_data["memory_id"])

            # Ensure center node is included if specified and exists, even if filters applied
            if center_memory_id and center_memory_id not in final_selected_ids:
                if center_memory_id in memories_data:  # Check if fetched but filtered out
                    final_selected_ids.add(center_memory_id)
                # else: it wasn't fetched initially, likely doesn't exist or error occurred

            # Check again if filtering removed all nodes
            if not final_selected_ids:
                logger.info("No memories remained after applying filters for visualization.")
                # Return empty graph
                return {
                    "workflow_id": target_workflow_id,
                    "center_memory_id": center_memory_id,
                    "visualization": "```mermaid\ngraph TD\n    NoNodes[No memories match filters]\n```",
                    "node_count": 0,
                    "link_count": 0,
                    "format": "mermaid",
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            # --- 3. Fetch Links BETWEEN Selected Memories ---
            final_ids_list = list(final_selected_ids)
            placeholders = ", ".join(["?"] * len(final_ids_list))
            links_query = f"""
                SELECT * FROM memory_links
                WHERE source_memory_id IN ({placeholders}) AND target_memory_id IN ({placeholders})
            """
            async with conn.execute(links_query, final_ids_list * 2) as cursor:
                links_data = [dict(row) for row in await cursor.fetchall()]

            # --- 4. Generate Mermaid Diagram ---
            # Use the final filtered memories_data values
            mermaid_string = await _generate_memory_network_mermaid(
                list(memories_data.values()), links_data, center_memory_id
            )

            # --- 5. Return Result ---
            processing_time = time.time() - start_time
            node_count = len(memories_data)
            link_count = len(links_data)
            logger.info(
                f"Generated memory network visualization ({node_count} nodes, {link_count} links) for workflow {target_workflow_id}",
                emoji_key="network",
            )
            return {
                "workflow_id": target_workflow_id,
                "center_memory_id": center_memory_id,
                "visualization": mermaid_string,
                "node_count": node_count,
                "link_count": link_count,
                "format": "mermaid",
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(f"Error visualizing memory network: {e}", exc_info=True)
        raise ToolError(f"Failed to visualize memory network: {str(e)}") from e


# ======================================================
# Exports
# ======================================================

__all__ = [
    # Initialization
    "initialize_memory_system",
    # Workflow
    "create_workflow",
    "update_workflow_status",
    "list_workflows",
    "get_workflow_details",
    # Actions
    "record_action_start",
    "record_action_completion",
    "get_recent_actions",
    "get_action_details",
    # Action Dependency Tools
    "add_action_dependency",
    "get_action_dependencies",
    # Artifacts
    "record_artifact",
    "get_artifacts",
    "get_artifact_by_id",
    # Thoughts
    "record_thought",
    "create_thought_chain",
    "get_thought_chain",
    # Core Memory
    "store_memory",
    "get_memory_by_id",
    "create_memory_link",
    "search_semantic_memories",
    "query_memories",
    "hybrid_search_memories",
    "update_memory",
    "get_linked_memories",
    # Context & State
    "get_working_memory",
    "focus_memory",
    "optimize_working_memory",
    "save_cognitive_state",
    "load_cognitive_state",
    "get_workflow_context",
    # Automated Cognitive Management
    "auto_update_focus",
    "promote_memory_level",
    # Meta-Cognition & Maintenance
    "consolidate_memories",
    "generate_reflection",
    "summarize_text",
    "delete_expired_memories",
    "compute_memory_statistics",
    "compute_memory_statistics",
    # Reporting & Visualization
    "generate_workflow_report",
    "visualize_reasoning_chain",
    "visualize_memory_network",
]


# Example Usage (for testing)
async def _example():
    db = "test_unified_memory.db"
    if os.path.exists(db):
        os.remove(db)

    init_result = await initialize_memory_system(db_path=db)
    print("Init Result:", init_result)

    wf_result = await create_workflow(
        title="Test Analysis Workflow",
        goal="Analyze test data",
        tags=["testing", "example"],
        db_path=db,
    )
    wf_id = wf_result["workflow_id"]
    print("\nWorkflow Created:", wf_result)

    thought1 = await record_thought(
        workflow_id=wf_id, content="Need to load the data first.", thought_type="plan", db_path=db
    )
    print("\nThought Recorded:", thought1)

    action1_start = await record_action_start(
        workflow_id=wf_id,
        action_type="tool_use",
        reasoning="Load data from file.",
        tool_name="load_data",
        tool_args={"file": "data.csv"},
        title="Load Data",
        tags=["io"],
        related_thought_id=thought1["thought_id"],
        db_path=db,
    )
    action1_id = action1_start["action_id"]
    print("\nAction Started:", action1_start)

    # Simulate tool execution
    await asyncio.sleep(0.1)
    tool_output = {"rows_loaded": 100, "columns": ["A", "B"]}

    action1_end = await record_action_completion(
        action_id=action1_id,
        tool_result=tool_output,
        summary="Data loaded successfully.",
        db_path=db,
    )
    print("\nAction Completed:", action1_end)

    artifact1 = await record_artifact(
        workflow_id=wf_id,
        action_id=action1_id,
        name="Loaded Data Sample",
        artifact_type="json",
        content=json.dumps(tool_output),
        description="Sample of loaded data structure",
        tags=["data"],
        db_path=db,
    )
    print("\nArtifact Recorded:", artifact1)

    mem1 = await store_memory(
        workflow_id=wf_id,
        content="Column A seems to be numerical.",
        memory_type="observation",
        importance=6.0,
        action_id=action1_id,
        db_path=db,
    )
    print("\nMemory Stored:", mem1)
    mem2 = await store_memory(
        workflow_id=wf_id,
        content="Column B looks categorical.",
        memory_type="observation",
        importance=6.0,
        action_id=action1_id,
        db_path=db,
    )
    print("\nMemory Stored:", mem2)

    link1 = await create_memory_link(
        source_memory_id=mem1["memory_id"],
        target_memory_id=mem2["memory_id"],
        link_type="related",
        db_path=db,
    )
    print("\nMemory Link Created:", link1)

    mem_get = await get_memory_by_id(memory_id=mem1["memory_id"], include_links=True, db_path=db)
    print("\nGet Memory By ID:", mem_get)

    # Close the connection on app shutdown
    await DBConnection.close_connection()


# if __name__ == "__main__":
#     asyncio.run(_example())
