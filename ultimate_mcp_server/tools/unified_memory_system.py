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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
from ultimate_mcp_server.utils.text import count_tokens

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
    logger.critical(
        f"CRITICAL: Failed to load configuration for unified_memory_system: {config_e}",
        exc_info=True,
    )
    # Provide fallback defaults if config fails, allowing *some* functionality maybe?
    # Or raise the error immediately. Raising is probably safer.
    raise RuntimeError(
        f"Failed to initialize configuration for unified_memory_system: {config_e}"
    ) from config_e

# --- UMS Tool Default Constants (can be overridden by agent via fetch_limits/show_limits) ---
# These should mirror or be inspired by the agent's CONTEXT_*_FETCH_LIMIT and CONTEXT_*_SHOW_LIMIT
UMS_DEFAULT_FETCH_LIMIT_RECENT_ACTIONS = 10
UMS_DEFAULT_FETCH_LIMIT_IMPORTANT_MEMORIES = 7
UMS_DEFAULT_FETCH_LIMIT_KEY_THOUGHTS = 7
UMS_DEFAULT_FETCH_LIMIT_PROACTIVE = 5
UMS_DEFAULT_FETCH_LIMIT_PROCEDURAL = 3
UMS_DEFAULT_FETCH_LIMIT_LINKS = 5
UMS_DEFAULT_FETCH_LIMIT_GOAL_DEPTH = 3  # Max parent goals to fetch details for
UMS_DEFAULT_SHOW_LIMIT_WORKING_MEMORY = 10
UMS_DEFAULT_SHOW_LIMIT_GOAL_STACK = 5
UMS_PKG_DEFAULT_FETCH_RECENT_ACTIONS = 10
UMS_PKG_DEFAULT_FETCH_IMPORTANT_MEMORIES = 7
UMS_PKG_DEFAULT_FETCH_KEY_THOUGHTS = 7
UMS_PKG_DEFAULT_FETCH_PROACTIVE = 5
UMS_PKG_DEFAULT_FETCH_PROCEDURAL = 3
UMS_PKG_DEFAULT_FETCH_LINKS = 5
# Show limits are mainly for compression decisions within this tool, or if it were to do truncation.
UMS_PKG_DEFAULT_SHOW_LINKS_SUMMARY = 3

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
    USER_GUIDANCE = "user_guidance"
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
    USER_INPUT = "user_input"
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


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ABANDONED = "abandoned"


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
    """CREATE TABLE IF NOT EXISTS ums_internal_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
    );""",
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
    # ───────────────── goals ───────────────────
    """CREATE TABLE IF NOT EXISTS goals (
        goal_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL
                        REFERENCES workflows(workflow_id)
                        ON DELETE CASCADE,
        parent_goal_id TEXT
                        REFERENCES goals(goal_id)
                        ON DELETE SET NULL, -- Or ON DELETE CASCADE if sub-goals should be removed with parent
        title TEXT, -- Optional brief title, description is main content
        description TEXT NOT NULL,
        status TEXT NOT NULL, -- e.g., 'active', 'completed', 'failed', 'paused', 'abandoned'
        priority INTEGER DEFAULT 3, -- e.g., 1 (high) to 5 (low)
        reasoning TEXT, -- Why this goal exists or is important
        acceptance_criteria TEXT, -- JSON list of strings for criteria
        metadata TEXT, -- JSON for other structured data
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        completed_at INTEGER, -- Timestamp when goal reached a terminal status
        sequence_number INTEGER -- For ordering sibling goals under a parent (optional but good)
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
    "CREATE INDEX IF NOT EXISTS idx_goals_workflow_id ON goals(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_goals_parent_goal_id ON goals(parent_goal_id);",
    "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);",
    "CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority);",
    "CREATE INDEX IF NOT EXISTS idx_goals_sequence_number ON goals(parent_goal_id, sequence_number);",
    "CREATE INDEX IF NOT EXISTS idx_ums_internal_metadata_key ON ums_internal_metadata(key);",
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
    """CREATE TRIGGER IF NOT EXISTS memories_after_update_delete AFTER UPDATE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content, description, reasoning, tags, workflow_id, memory_id)
        VALUES ('delete', old.rowid, old.content, old.description, old.reasoning, old.tags, old.workflow_id, old.memory_id);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS memories_after_update_insert AFTER UPDATE ON memories BEGIN
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
    global SCHEMA_STATEMENTS
    _instance: Optional[aiosqlite.Connection] = None
    _lock = asyncio.Lock()
    _db_path_used: Optional[str] = None
    _init_lock_timeout = 15.0

    def __init__(self, db_path: str = agent_memory_config.db_path):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _initialize_instance(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path, timeout=agent_memory_config.connection_timeout)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA temp_store=MEMORY;")
        await conn.execute("PRAGMA cache_size=-32000;")
        await conn.execute("PRAGMA mmap_size=2147483647;")
        await conn.execute("PRAGMA busy_timeout=30000;")
        await conn.create_function("json_contains", 2, _json_contains, deterministic=True)
        await conn.create_function("json_contains_any", 2, _json_contains_any, deterministic=True)
        await conn.create_function("json_contains_all", 2, _json_contains_all, deterministic=True)
        await conn.create_function(
            "compute_memory_relevance", 5, _compute_memory_relevance, deterministic=True
        )
        await conn.execute("BEGIN IMMEDIATE;")
        for stmt in SCHEMA_STATEMENTS:
            u = stmt.strip().upper()
            if u.startswith("PRAGMA") and not (u.startswith("PRAGMA FOREIGN_KEYS") and "ON" in u):
                continue
            await conn.execute(stmt)
        await conn.commit()
        DBConnection._db_path_used = self.db_path
        return conn

    async def __aenter__(self) -> aiosqlite.Connection:
        if DBConnection._instance and self.db_path != DBConnection._db_path_used:
            raise RuntimeError(f"singleton already bound to {DBConnection._db_path_used}")
        if DBConnection._instance:
            await DBConnection._instance.execute("PRAGMA foreign_keys=ON;")
            return DBConnection._instance
        async with asyncio.timeout(self._init_lock_timeout):
            async with DBConnection._lock:
                if DBConnection._instance is None:
                    DBConnection._instance = await self._initialize_instance()
        await DBConnection._instance.execute("PRAGMA foreign_keys=ON;")
        return DBConnection._instance

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    async def _ckpt(cls, conn: aiosqlite.Connection, mode: str) -> Tuple[int, int, int]:
        async with conn.execute(f"PRAGMA wal_checkpoint({mode});") as c:
            return await c.fetchone()

    @classmethod
    async def force_wal_checkpoint(
        cls,
        modes: Sequence[str] = ("PASSIVE",),  # Default to PASSIVE for less disruption
        retries: int = 5,
        wait: float = 0.2,
        log_if_unable: bool = True,  # Control logging if it fails
    ):
        async with cls._lock:  # Acquire lock for the whole checkpointing attempt
            if cls._instance is None:
                if log_if_unable:
                    logger.warning("force_wal_checkpoint: No active connection, cannot checkpoint.")
                return  # Cannot checkpoint if no connection

            conn = cls._instance

            # Critical: Do not attempt checkpoint if the shared connection is in an active transaction
            # started by some other part of the code that hasn't committed yet.
            if conn.in_transaction:
                if log_if_unable:
                    logger.warning(
                        "force_wal_checkpoint: Shared connection is in an active transaction. "
                        "Checkpoint attempt deferred. This might indicate a transaction leak or "
                        "checkpoint being called at an inappropriate time."
                    )
                # Do NOT raise ToolError here, as it might interrupt the ongoing transaction.
                # Let the caller decide if this is critical. Often, a deferred checkpoint is okay.
                return

            # If we are here, conn is not in a transaction *at this moment by this check*.
            # The PRAGMA wal_checkpoint itself will try to acquire necessary locks.

            for mode in modes:
                for attempt in range(retries):
                    try:
                        # The PRAGMA wal_checkpoint itself doesn't need to be wrapped in BEGIN/COMMIT
                        # as it operates on the WAL file system level.
                        async with conn.execute(f"PRAGMA wal_checkpoint({mode});") as c:
                            result = await c.fetchone()
                            if result:
                                busy, log_frames, checkpointed_frames = result
                                logger.debug(
                                    f"WAL Checkpoint ({mode}, attempt {attempt + 1}): busy={busy}, log={log_frames}, ckpt={checkpointed_frames}"
                                )
                                if busy == 0 and log_frames == 0:
                                    logger.debug(f"WAL successfully checkpointed with mode {mode}.")
                                    return  # Success
                            else:
                                logger.warning(
                                    f"WAL Checkpoint ({mode}, attempt {attempt + 1}): No result from PRAGMA."
                                )

                        # If not returned yet, means busy or log_frames > 0
                        if attempt < retries - 1:  # Don't sleep on the last attempt
                            await asyncio.sleep(
                                wait * (attempt + 1)
                            )  # Exponential backoff for sleep

                    except aiosqlite.OperationalError as oe:
                        # This can happen if the DB is locked or busy in a way that prevents checkpoint
                        logger.warning(
                            f"WAL Checkpoint ({mode}, attempt {attempt + 1}) failed with OperationalError: {oe}. Retrying..."
                        )
                        if attempt < retries - 1:
                            await asyncio.sleep(wait * (attempt + 1))
                        elif mode == modes[-1] and attempt == retries - 1:  # Last mode, last retry
                            if log_if_unable:
                                logger.error(
                                    f"force_wal_checkpoint: Unable to flush WAL after all retries due to OperationalError: {oe}"
                                )
                            raise ToolError(f"unable to flush WAL due to: {oe}") from oe
                    except Exception as e:
                        logger.error(
                            f"force_wal_checkpoint: Unexpected error during checkpoint ({mode}, attempt {attempt + 1}): {e}",
                            exc_info=True,
                        )
                        if mode == modes[-1] and attempt == retries - 1:  # Last mode, last retry
                            if log_if_unable:
                                logger.error(
                                    f"force_wal_checkpoint: Unable to flush WAL after all retries due to unexpected error: {e}"
                                )
                            raise ToolError(
                                f"unable to flush WAL due to unexpected error: {e}"
                            ) from e
                # If loop for modes finishes and we haven't returned, it means all modes failed for this attempt

            # If we've gone through all modes and retries and haven't returned, it failed.
            if log_if_unable:
                logger.error(
                    "force_wal_checkpoint: Unable to flush WAL after all modes and retries."
                )
            raise ToolError("unable to flush WAL")

    @classmethod
    async def close_connection(cls):
        if not cls._instance:
            return
        async with cls._lock:
            conn, cls._instance, cls._db_path_used = cls._instance, None, None
            await cls.force_wal_checkpoint()
            await conn.close()

    @contextlib.asynccontextmanager
    async def transaction(self):
        conn = await self.__aenter__()
        if conn.in_transaction:
            try:
                yield conn
            except:
                raise
        else:
            try:
                await conn.execute("BEGIN DEFERRED;")
                yield conn
                await conn.commit()
            except:
                if conn.in_transaction:
                    await conn.rollback()
                raise


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

    decayed_importance = max(
        0, importance * (1.0 - agent_memory_config.memory_decay_rate * age_hours)
    )
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
            if abs(ts_value) > 2**40:  # Arbitrary large number check
                logger.warning(
                    f"Numeric timestamp {ts_value} seems out of range, returning as string."
                )
                return str(ts_value)
            return to_iso_z(ts_value)
        except (OverflowError, OSError, ValueError, TypeError) as e:
            logger.warning(f"Failed to convert numeric timestamp {ts_value} to ISO: {e}")
            return str(ts_value)  # Fallback to string representation of number
    elif isinstance(ts_value, str):
        # Try to parse and reformat to ensure consistency, but return original if parsing fails
        try:
            # Attempt parsing, assuming it might already be close to ISO
            dt_obj = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            # Reformat to our standard Z format
            return to_iso_z(dt_obj.timestamp())
        except ValueError:
            # If parsing fails, return the original string but log a warning
            logger.debug(
                f"Timestamp value '{ts_value}' is a string but not valid ISO format. Returning as is."
            )
            return ts_value
    elif ts_value is None:
        return None
    else:
        logger.warning(
            f"Unexpected timestamp type {type(ts_value)}, value: {ts_value}. Returning string representation."
        )
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
            print(
                "CRITICAL WARNING: agent_memory_config not loaded in serialize, using default max_len"
            )  # Use print as logger might not be ready
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
        conn: aiosqlite.Connection,  # Accept the connection object
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
            op_data_json = (
                await MemoryUtils.serialize(operation_data) if operation_data is not None else None
            )

            # Use the PASSED connection object 'conn'
            await conn.execute(
                """
                INSERT INTO memory_operations
                (operation_log_id, workflow_id, memory_id, action_id, operation, operation_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (op_id, workflow_id, memory_id, action_id, operation, op_data_json, timestamp_unix),
            )
        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to log memory operation '{operation}' using provided conn: {e}",
                exc_info=True,
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
    """(Agent Internal) Initializes/verifies UMS DB & embeddings. Not for direct LLM call."""
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
    """
    Creates a new UMS workflow, ensuring all related database operations are atomic.
    Also creates a primary thought chain and an initial goal thought if a goal is provided.

    Args:
        title: The title of the workflow.
        description: (Optional) A detailed description of the workflow.
        goal: (Optional) The primary goal or objective of this workflow.
        tags: (Optional) A list of tags to associate with the workflow.
        metadata: (Optional) A dictionary for storing additional structured data.
        parent_workflow_id: (Optional) The ID of a parent workflow, if this is a sub-workflow.
        db_path: (Optional) Path to the SQLite database file.

    Returns:
        A dictionary containing details of the created workflow, including its ID,
        primary thought chain ID, and success status.

    Raises:
        ToolInputError: If input parameters are invalid (e.g., missing title, non-existent parent).
        ToolError: If any database operation fails unexpectedly.
    """
    if not title or not isinstance(title, str):
        raise ToolInputError("Workflow title must be a non-empty string.", param_name="title")

    workflow_id = MemoryUtils.generate_id()
    now_unix = int(time.time())  # Use Unix timestamp for all DB operations
    chain_id = MemoryUtils.generate_id()  # Pre-generate for return value consistency

    # Get a DBConnection manager instance
    db_manager = DBConnection(db_path)

    try:
        # Start an explicit transaction for all database modifications
        async with db_manager.transaction() as conn:
            # 1. Validate parent_workflow_id if provided
            if parent_workflow_id:
                async with conn.execute(
                    "SELECT 1 FROM workflows WHERE workflow_id = ?", (parent_workflow_id,)
                ) as cursor:
                    if not await cursor.fetchone():
                        raise ToolInputError(
                            f"Parent workflow ID '{parent_workflow_id}' not found.",
                            param_name="parent_workflow_id",
                        )
                # Cursor is automatically closed here by 'async with'

            # 2. Serialize metadata
            metadata_json = await MemoryUtils.serialize(metadata)

            # 3. Insert the main workflow record
            await conn.execute(
                """
                INSERT INTO workflows (
                    workflow_id, title, description, goal, status,
                    created_at, updated_at, parent_workflow_id, metadata, last_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    title,
                    description,
                    goal,
                    WorkflowStatus.ACTIVE.value,
                    now_unix,  # Use Unix timestamp
                    now_unix,  # Use Unix timestamp
                    parent_workflow_id,
                    metadata_json,
                    now_unix,  # Use Unix timestamp for last_active
                ),
            )

            # 4. Process tags for the workflow (uses the transactional 'conn')
            await MemoryUtils.process_tags(conn, workflow_id, tags or [], "workflow")

            # 5. Create the primary thought chain for this workflow
            await conn.execute(
                """
                INSERT INTO thought_chains (
                    thought_chain_id, workflow_id, title, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    chain_id,
                    workflow_id,
                    f"Main reasoning for: {title}",
                    now_unix,
                ),  # Use Unix timestamp
            )

            # 6. If a goal string is provided, record it as the first thought in the primary chain
            if goal:
                thought_id = MemoryUtils.generate_id()
                # get_next_sequence_number uses the transactional 'conn'
                sequence_number = await MemoryUtils.get_next_sequence_number(
                    conn, chain_id, "thoughts", "thought_chain_id"
                )
                await conn.execute(
                    """
                    INSERT INTO thoughts (
                        thought_id, thought_chain_id, thought_type, content,
                        sequence_number, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thought_id,
                        chain_id,
                        ThoughtType.GOAL.value,
                        goal,
                        sequence_number,
                        now_unix,  # Use Unix timestamp
                    ),
                )
            # If all operations within the 'async with db_manager.transaction() as conn:' block succeed,
            # the transaction will be committed automatically upon exiting the block.
            # If any exception occurs, it will be rolled back.

        try:
            await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
        except ToolError as te_wal:
            logger.warning(
                f"Passive WAL checkpoint failed after creating workflow '{title}': {te_wal}. Main operation succeeded."
            )

            logger.info(
                f"Successfully created workflow '{title}' (ID: {_fmt_id(workflow_id)}) and primary thought chain (ID: {_fmt_id(chain_id)})."
            )

        # Prepare the return dictionary using ISO formatted timestamps for external consumers
        return {
            "workflow_id": workflow_id,
            "title": title,
            "description": description,
            "goal": goal,
            "status": WorkflowStatus.ACTIVE.value,
            "created_at": to_iso_z(now_unix),  # Format for output
            "updated_at": to_iso_z(now_unix),  # Format for output
            "tags": tags or [],
            "primary_thought_chain_id": chain_id,
            "success": True,
        }

    except ToolInputError:
        # Re-raise ToolInputError to be handled by the decorator
        raise
    except Exception as e:
        # Log unexpected errors and wrap them in ToolError
        logger.error(f"Unexpected error creating workflow '{title}': {e}", exc_info=True)
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
    """Changes a workflow's status (e.g., active, completed, failed).
    Args: workflow_id, status, completion_message, update_tags."""
    try:
        status_enum = WorkflowStatus(status.lower())
    except ValueError as e:
        valid_statuses = [s.value for s in WorkflowStatus]
        raise ToolInputError(
            f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}",
            param_name="status",
        ) from e

    now_unix = int(time.time())
    db_manager = DBConnection(db_path)  # Get DB manager instance

    async with db_manager.transaction() as conn:  # Use transaction context manager
        # Check existence first within the transaction
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

        update_params: List[Any] = [
            status_enum.value,
            now_unix,
            now_unix,
        ]  # For status, updated_at, last_active
        set_clauses_parts = ["status = ?", "updated_at = ?", "last_active = ?"]

        if status_enum in [
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.ABANDONED,
        ]:
            set_clauses_parts.append("completed_at = ?")
            update_params.append(now_unix)

        set_clauses_sql = ", ".join(set_clauses_parts)
        update_params.append(workflow_id)  # For WHERE clause

        await conn.execute(
            f"UPDATE workflows SET {set_clauses_sql} WHERE workflow_id = ?", update_params
        )

        # Add completion message as thought (within the same transaction)
        if completion_message:
            primary_thought_chain_id = None
            async with conn.execute(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                (workflow_id,),
            ) as tc_cursor:
                tc_row = await tc_cursor.fetchone()
                if tc_row:
                    primary_thought_chain_id = tc_row["thought_chain_id"]

            if primary_thought_chain_id:
                seq_no = await MemoryUtils.get_next_sequence_number(
                    conn,
                    primary_thought_chain_id,
                    "thoughts",
                    "thought_chain_id",  # Pass conn
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
                        primary_thought_chain_id,
                        thought_type,
                        completion_message,
                        seq_no,
                        now_unix,
                    ),
                )
                logger.debug(f"Recorded completion thought {thought_id} for workflow {workflow_id}")
            else:
                logger.warning(
                    f"Could not find primary thought chain for workflow {workflow_id} to add completion message."
                )

        # Process additional tags (within the same transaction)
        # Assuming MemoryUtils.process_tags accepts and uses 'conn'
        await MemoryUtils.process_tags(conn, workflow_id, update_tags or [], "workflow")

        # Transaction will commit here if no exceptions
        # No explicit conn.commit() needed

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after updating workflow '{workflow_id}' status: {te_wal}. Main operation succeeded."
        )

    # Prepare the result dictionary
    result = {
        "workflow_id": workflow_id,
        "status": status_enum.value,
        "updated_at": to_iso_z(now_unix),
        "success": True,
    }

    if status_enum in [
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.ABANDONED,
    ]:
        result["completed_at"] = to_iso_z(now_unix)

    logger.info(
        f"Updated workflow {workflow_id} status to '{status_enum.value}'",
        emoji_key="arrows_counterclockwise",
    )
    return result


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
    """Logs the beginning of an agent action or tool use, with reasoning. Creates linked memory.
    Args: workflow_id, action_type, reasoning, tool_name, tool_args, title, parent_action_id, tags, related_thought_id.
    Returns: action_id, linked_memory_id."""
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
    start_time_perf = time.perf_counter()  # For precise processing time

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # --- Existence Checks (Workflow, Parent Action, Related Thought) ---
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            wf_exists = await cursor.fetchone()
            if not wf_exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

        if parent_action_id:
            async with conn.execute(
                "SELECT 1 FROM actions WHERE action_id = ? AND workflow_id = ?",
                (parent_action_id, workflow_id),
            ) as cursor:
                parent_exists = await cursor.fetchone()
                if not parent_exists:
                    raise ToolInputError(
                        f"Parent action '{parent_action_id}' not found or does not belong to workflow '{workflow_id}'.",
                        param_name="parent_action_id",
                    )

        if related_thought_id:
            async with conn.execute(
                """SELECT 1 FROM thoughts t
                   JOIN thought_chains tc ON t.thought_chain_id = tc.thought_chain_id
                   WHERE t.thought_id = ? AND tc.workflow_id = ?""",
                (related_thought_id, workflow_id),
            ) as cursor:
                thought_exists = await cursor.fetchone()
                if not thought_exists:
                    raise ToolInputError(
                        f"Related thought '{related_thought_id}' not found or does not belong to workflow '{workflow_id}'.",
                        param_name="related_thought_id",
                    )

        # --- Determine Action Title & Sequence Number ---
        sequence_number = await MemoryUtils.get_next_sequence_number(
            conn,
            workflow_id,
            "actions",
            "workflow_id",  # Pass conn
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
        await MemoryUtils.process_tags(conn, action_id, tags or [], "action")  # Pass conn

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
        mem_tags_list = ["action_start", action_type_enum.value] + (tags or [])
        mem_tags_json = json.dumps(list(set(mem_tags_list)))  # Ensure unique tags

        await conn.execute(
            """
             INSERT INTO memories (memory_id, workflow_id, action_id, content, memory_level, memory_type,
             importance, confidence, tags, created_at, updated_at, access_count, last_accessed)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL) 
             """,  # Added last_accessed as NULL
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
                0,  # access_count starts at 0
            ),
        )
        await MemoryUtils._log_memory_operation(  # Pass conn
            conn, workflow_id, "create_from_action_start", memory_id, action_id
        )

        # --- Update Workflow Timestamp ---
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )
    # Transaction commits automatically here if no exceptions

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after starting action '{action_id}': {te_wal}. Main operation succeeded."
        )

    # --- Prepare Result ---
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
        "processing_time": time.perf_counter() - start_time_perf,
    }

    logger.info(
        f"Started action '{_fmt_id(auto_title)}' ({_fmt_id(action_id)}) in workflow {_fmt_id(workflow_id)}",
        emoji_key="fast_forward",
        duration=result["processing_time"],
    )
    return result


@with_tool_metrics
@with_error_handling
async def record_action_completion(
    action_id: str,
    status: str = "completed",  # Keep default as string for flexibility from LLM
    tool_result: Optional[Any] = None,
    summary: Optional[str] = None,
    conclusion_thought: Optional[str] = None,
    conclusion_thought_type: str = "inference",  # Keep default as string
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Logs the end of an agent action, its result, and status. Updates linked memory.
    Args: action_id, status, tool_result, summary, conclusion_thought, conclusion_thought_type.
    Returns: action_id, conclusion_thought_id."""
    start_time_perf = time.perf_counter()
    # --- Validate Status ---
    try:
        # status comes in as string, convert to enum
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
    conclusion_thought_id_val: Optional[str] = None  # Initialize

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # --- 1. Verify Action and Get Workflow ID ---
        async with conn.execute(
            "SELECT workflow_id, status FROM actions WHERE action_id = ?", (action_id,)
        ) as cursor:
            action_row = await cursor.fetchone()
            if not action_row:
                raise ToolInputError(f"Action not found: {action_id}", param_name="action_id")

        workflow_id = action_row["workflow_id"]
        current_status_from_db = action_row["status"]
        if current_status_from_db not in [
            ActionStatus.IN_PROGRESS.value,
            ActionStatus.PLANNED.value,
        ]:
            logger.warning(
                f"Action {action_id} already has terminal status '{current_status_from_db}'. Allowing update anyway."
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
            (status_enum.value, now_unix, tool_result_json, action_id),
        )

        # --- 3. Update Workflow Timestamp ---
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # --- 4. Add Conclusion Thought (if provided) ---
        if conclusion_thought and thought_type_enum:
            primary_chain_id: Optional[str] = None
            async with conn.execute(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                (workflow_id,),
            ) as cursor:
                chain_row = await cursor.fetchone()
                if chain_row:
                    primary_chain_id = chain_row["thought_chain_id"]

            if primary_chain_id:
                seq_no = await MemoryUtils.get_next_sequence_number(
                    conn,
                    primary_chain_id,
                    "thoughts",
                    "thought_chain_id",  # Pass conn
                )
                conclusion_thought_id_val = MemoryUtils.generate_id()
                await conn.execute(
                    """
                    INSERT INTO thoughts
                        (thought_id, thought_chain_id, thought_type, content, sequence_number, created_at, relevant_action_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conclusion_thought_id_val,
                        primary_chain_id,
                        thought_type_enum.value,
                        conclusion_thought,
                        seq_no,
                        now_unix,
                        action_id,
                    ),
                )
                logger.debug(
                    f"Recorded conclusion thought {_fmt_id(conclusion_thought_id_val)} for action {_fmt_id(action_id)}"
                )
            else:
                logger.warning(
                    f"Could not find primary thought chain for workflow {_fmt_id(workflow_id)} to add conclusion thought."
                )

        # --- 5. Update Linked Episodic Memory ---
        memory_id_to_update: Optional[str] = None
        original_content: Optional[str] = None
        async with conn.execute(
            "SELECT memory_id, content FROM memories WHERE action_id = ? AND memory_type = ?",
            (action_id, MemoryType.ACTION_LOG.value),
        ) as cursor:
            memory_row = await cursor.fetchone()
            if memory_row:
                memory_id_to_update = memory_row["memory_id"]
                original_content = memory_row["content"]

        if memory_id_to_update and original_content is not None:
            update_parts = [f"Completed ({status_enum.value})."]
            if summary:
                update_parts.append(f"Summary: {summary}")

            result_preview = ""
            if tool_result is not None:
                if isinstance(tool_result, dict):
                    result_preview = f"Result: [Dict with {len(tool_result)} keys]"
                elif isinstance(tool_result, list):
                    result_preview = f"Result: [List with {len(tool_result)} items]"
                elif isinstance(tool_result, str) and len(tool_result) > 100:
                    result_preview = f"Result: {tool_result[:97]}..."
                elif isinstance(tool_result, str):
                    result_preview = f"Result: {tool_result}"
                elif tool_result is True:  # Explicitly check for boolean True
                    result_preview = "Result: Success (True)"
                elif tool_result is False:  # Explicitly check for boolean False
                    result_preview = "Result: Failure (False)"
                else:
                    result_preview = f"Result: [Object type {type(tool_result).__name__}]"
            if result_preview:
                update_parts.append(result_preview)

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
                    updated_at = ?,
                    last_accessed = ? 
                WHERE memory_id = ?
                """,  # Added last_accessed update
                (new_content, importance_mult, now_unix, now_unix, memory_id_to_update),
            )
            await MemoryUtils._log_memory_operation(  # Pass conn
                conn,
                workflow_id,
                "update_from_action_completion",
                memory_id_to_update,
                action_id,
                {"status": status_enum.value, "summary_added": bool(summary)},
            )
            logger.debug(
                f"Updated linked memory {_fmt_id(memory_id_to_update)} for completed action {_fmt_id(action_id)}"
            )
        elif action_row:  # action_row confirmed action_id exists
            logger.warning(
                f"Could not find corresponding action_log memory for completed action {_fmt_id(action_id)} to update."
            )
    # Transaction commits automatically here

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after completing action '{action_id}': {te_wal}. Main operation succeeded."
        )

    # --- Prepare Result ---
    result = {
        "action_id": action_id,
        "workflow_id": workflow_id,
        "status": status_enum.value,
        "completed_at": to_iso_z(now_unix),
        "conclusion_thought_id": conclusion_thought_id_val,
        "success": True,
        "processing_time": time.perf_counter() - start_time_perf,
    }

    logger.info(
        f"Completed action {_fmt_id(action_id)} with status {status_enum.value}",
        emoji_key="white_check_mark",
        duration=result["processing_time"],
    )
    return result


@with_tool_metrics
@with_error_handling
async def get_action_details(
    action_id: Optional[str] = None,
    action_ids: Optional[List[str]] = None,
    include_dependencies: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves detailed information about one or more specific actions.
    Args: action_id or action_ids, include_dependencies.
    Returns: List of action objects."""
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

            for action_data in actions_result:
                if action_data.get("started_at"):
                    action_data["started_at"] = safe_format_timestamp(action_data["started_at"])
                if action_data.get("completed_at"):
                    action_data["completed_at"] = safe_format_timestamp(action_data["completed_at"])
                # Format timestamps in dependencies if include_dependencies=True
                if "dependencies" in action_data:
                    for dep_list_key in ["depends_on", "dependent_actions"]:
                        for dep_action in action_data["dependencies"].get(dep_list_key, []):
                            if dep_action.get("started_at"):
                                dep_action["started_at"] = safe_format_timestamp(
                                    dep_action["started_at"]
                                )
                            if dep_action.get("completed_at"):
                                dep_action["completed_at"] = safe_format_timestamp(
                                    dep_action["completed_at"]
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
    provider: Optional[str] = None,  # Allow None to use default logic
    model: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal Utility) LLM-summarizes a given text block.
    Agent should use 'get_rich_context_package' for its main operational context.
    Args: text_to_summarize, target_tokens, context_type, workflow_id.
    Returns: summary text."""
    start_time = time.time()

    if not text_to_summarize:
        raise ToolInputError("Text to summarize cannot be empty", param_name="text_to_summarize")

    # Select appropriate prompt template based on context type (existing logic assumed correct)
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
    else:  # Generic template
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
        config = get_config()  # Assuming get_config() is available and works
        # Use provided 'provider', else config.default_provider, else a hardcoded fallback
        provider_to_use_str = (
            provider or config.default_provider or LLMGatewayProvider.ANTHROPIC.value
        )  # Ensure LLMGatewayProvider is imported

        provider_instance = await get_provider(
            provider_to_use_str
        )  # Assuming get_provider is available
        if not provider_instance:
            raise ToolError(f"Failed to initialize provider '{provider_to_use_str}'.")

        model_to_use = model or provider_instance.get_default_model()
        if not model_to_use:  # Fallback if provider has no default
            logger.warning(
                f"Provider '{provider_to_use_str}' has no default model. Attempting with a generic fallback (this might fail)."
            )
            # Define a generic fallback or raise error if no model can be determined
            # For example, if provider_to_use_str is 'openai', a common fallback is 'gpt-3.5-turbo'
            # This part depends heavily on how your get_default_model and providers are set up.
            # If get_default_model can return None, you need a robust fallback here.
            if provider_to_use_str == LLMGatewayProvider.OPENAI.value:
                model_to_use = "gpt-3.5-turbo"  # Example fallback
            elif provider_to_use_str == LLMGatewayProvider.ANTHROPIC.value:
                model_to_use = "claude-3-haiku-20240307"  # Example fallback
            else:
                raise ToolError(
                    f"No model specified and provider '{provider_to_use_str}' has no default model configured, and no generic fallback for it."
                )

        # Generate summary
        generation_result = await provider_instance.generate_completion(
            prompt=prompt_template.format(
                text_to_summarize=text_to_summarize, target_tokens=target_tokens
            ),
            model=model_to_use,
            max_tokens=target_tokens + 150,  # Increased buffer slightly
            temperature=0.2,
        )

        summary_text = generation_result.text.strip()
        if not summary_text:
            # It's better to return success=False or an empty summary than raise ToolError here,
            # as an empty summary from LLM is a valid (though unhelpful) outcome.
            # The caller can decide how to handle an empty summary.
            logger.warning(f"LLM returned empty context summary for context_type '{context_type}'.")
            summary_text = ""  # Ensure it's an empty string, not None

        original_length = max(1, len(text_to_summarize))  # Avoid division by zero
        compression_ratio = len(summary_text) / original_length if original_length > 0 else 0.0

        # Log the operation if workflow_id provided, using a transaction
        if workflow_id:
            db_manager = DBConnection(db_path)
            async with db_manager.transaction() as conn:
                await MemoryUtils._log_memory_operation(  # Ensure this uses 'conn'
                    conn,
                    workflow_id,
                    "compress_context",
                    None,  # memory_id
                    None,  # action_id
                    {
                        "context_type": context_type,
                        "original_length": len(text_to_summarize),
                        "summary_length": len(summary_text),
                        "compression_ratio": compression_ratio,
                        "provider": provider_to_use_str,
                        "model": model_to_use,
                    },
                )
            # No need for explicit WAL checkpoint here unless this operation is very infrequent
            # and other operations are also infrequent. PASSIVE is fine if done.

        processing_time = time.time() - start_time
        logger.info(
            f"Compressed {context_type} context: {len(text_to_summarize)} -> {len(summary_text)} chars (Ratio: {compression_ratio:.2f}, LLM: {provider_to_use_str}/{model_to_use})",
            emoji_key="compression",
            time=processing_time,
        )

        return {
            "summary": summary_text,
            "context_type": context_type,
            "compression_ratio": compression_ratio,
            "success": True,  # Indicates the summarization process completed
            "processing_time": processing_time,
        }

    except ToolInputError:
        raise
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
    """Defines a dependency: action A requires action B to complete first.
    Args: source_action_id (dependent), target_action_id (prerequisite), dependency_type.
    Returns: dependency_id."""
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
    dependency_id: Optional[int] = None  # lastrowid returns int

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # --- Validate Actions & Workflow Consistency ---
        source_workflow_id: Optional[str] = None
        target_workflow_id: Optional[str] = None

        async with conn.execute(
            "SELECT workflow_id FROM actions WHERE action_id = ?", (source_action_id,)
        ) as cursor:
            source_row = await cursor.fetchone()
            if not source_row:
                raise ToolInputError(
                    f"Source action {source_action_id} not found.", param_name="source_action_id"
                )
            source_workflow_id = source_row["workflow_id"]

        async with conn.execute(
            "SELECT workflow_id FROM actions WHERE action_id = ?", (target_action_id,)
        ) as cursor:
            target_row = await cursor.fetchone()
            if not target_row:
                raise ToolInputError(
                    f"Target action {target_action_id} not found.", param_name="target_action_id"
                )
            target_workflow_id = target_row["workflow_id"]

        if source_workflow_id != target_workflow_id:
            raise ToolInputError(
                f"Source action ({_fmt_id(source_action_id)}) and target action ({_fmt_id(target_action_id)}) belong to different workflows.",
                param_name="target_action_id",
            )
        workflow_id = source_workflow_id  # Both actions are in this workflow

        # --- Insert Dependency (Ignoring duplicates) ---
        # Use an explicit cursor variable to check rowcount and get lastrowid
        insert_cursor = await conn.execute(
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

        if insert_cursor.rowcount > 0:
            dependency_id = insert_cursor.lastrowid  # This is an int for autoincrement PK
            logger.debug(f"Inserted new dependency row with ID: {dependency_id}")
        else:
            # If IGNORE occurred, fetch the existing dependency_id
            async with conn.execute(  # Use a new cursor for this SELECT
                "SELECT dependency_id FROM dependencies WHERE source_action_id = ? AND target_action_id = ? AND dependency_type = ?",
                (source_action_id, target_action_id, dependency_type),
            ) as existing_cursor:
                existing_row = await existing_cursor.fetchone()
                if existing_row:
                    dependency_id = existing_row["dependency_id"]  # This is an int
                    logger.debug(
                        f"Dependency already existed. Retrieved existing ID: {dependency_id}"
                    )
                else:
                    # This state is unusual: IGNORE happened but SELECT found nothing.
                    # Could be a race condition if not for the transaction, or very rapid delete.
                    # Within a transaction, this implies an issue or the UNIQUE constraint wasn't what triggered IGNORE.
                    logger.warning(
                        f"Dependency insert was ignored for ({_fmt_id(source_action_id)}, {_fmt_id(target_action_id)}, {dependency_type}), "
                        "but could not retrieve existing row. This might indicate a data consistency issue if the UNIQUE constraint was expected to be hit."
                    )
        await insert_cursor.close()  # Close the insert_cursor

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
            "db_dependency_id": dependency_id,  # This can be None if the fallback lookup failed
        }
        # Ensure _log_memory_operation uses the passed 'conn'
        await MemoryUtils._log_memory_operation(
            conn, workflow_id, "add_dependency", None, source_action_id, log_data
        )
        # Transaction will commit here if no exceptions were raised

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after adding dependency for action '{source_action_id}': {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    logger.info(
        f"Added/Confirmed dependency ({dependency_type}) from {_fmt_id(source_action_id)} to {_fmt_id(target_action_id)} (ID: {dependency_id or 'N/A'}).",
        emoji_key="link",
    )

    return {
        "source_action_id": source_action_id,
        "target_action_id": target_action_id,
        "dependency_type": dependency_type,
        "dependency_id": dependency_id,  # Can be None if not newly inserted and lookup failed
        "created_at": to_iso_z(now_unix),  # Return ISO formatted time
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def get_action_dependencies(
    action_id: str,
    direction: str = "downstream",  # "downstream" (depends on this) or "upstream" (this depends on)
    dependency_type: Optional[str] = None,
    include_details: bool = False,  # Whether to fetch full action details
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Lists actions that an action depends on (upstream) or that depend on it (downstream).
    Args: action_id, direction, dependency_type, include_details.
    Returns: List of related action summaries/details."""
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
    """Logs a created file, data, code, or other output as an artifact. Creates linked memory.
    Args: workflow_id, name, artifact_type, action_id, description, path, content, metadata, is_output, tags.
    Returns: artifact_id, linked_memory_id."""
    start_time = time.time()
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
    db_manager = DBConnection(db_path)  # Get DB manager instance

    linked_memory_id = MemoryUtils.generate_id()  # Pre-generate for return value consistency
    final_db_content = None  # Initialize

    async with db_manager.transaction() as conn:  # Use transaction context manager
        # --- Existence Checks (within transaction) ---
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

        if action_id:
            async with conn.execute(
                "SELECT 1 FROM actions WHERE action_id = ? AND workflow_id = ?",
                (action_id, workflow_id),
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Action {action_id} not found or does not belong to workflow {workflow_id}",
                        param_name="action_id",
                    )

        # --- Prepare Data (within transaction if it involves DB reads, but serialize doesn't) ---
        metadata_json = await MemoryUtils.serialize(
            metadata
        )  # Safe to call outside transaction if it doesn't query DB

        # Content truncation logic
        max_len = agent_memory_config.max_text_length
        if content:
            content_bytes = content.encode("utf-8")
            if len(content_bytes) > max_len:
                logger.warning(
                    f"Artifact content for '{name}' exceeds max length ({max_len} bytes). Storing truncated version in DB."
                )
                truncated_bytes = content_bytes[:max_len]
                final_db_content = truncated_bytes.decode("utf-8", errors="replace")
                # Refined truncation for UTF-8 characters
                if final_db_content.endswith("\ufffd") and max_len > 0:
                    # Try decoding one byte less to avoid partial char if possible
                    # This is a heuristic
                    for i in range(1, min(4, max_len)):  # Check up to 3 bytes back
                        try_shorter_bytes = content_bytes[: max_len - i]
                        try_shorter_str = try_shorter_bytes.decode(
                            "utf-8", errors="strict"
                        )  # Use strict to check validity
                        final_db_content = try_shorter_str  # If strict decode works, use it
                        break  # Found a valid shorter string
                    else:  # If loop finished without break, stick with original replace
                        pass
                final_db_content += "..."  # Add ellipsis after truncation
            else:
                final_db_content = content

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
                final_db_content,
                metadata_json,
                now_unix,
                is_output,
            ),
        )
        logger.debug(f"Inserted artifact record {artifact_id}")

        # --- Process Tags (assuming MemoryUtils.process_tags uses passed conn) ---
        artifact_tags_list = tags or []
        await MemoryUtils.process_tags(conn, artifact_id, artifact_tags_list, "artifact")
        logger.debug(f"Processed {len(artifact_tags_list)} tags for artifact {artifact_id}")

        # --- Update Workflow Timestamp ---
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # --- Create Linked Episodic Memory about the Artifact Creation (within transaction) ---
        memory_content = f"Artifact '{name}' (type: {artifact_type_enum.value}) was created"
        if action_id:
            memory_content += f" during action '{action_id[:8]}...'"
        if description:
            memory_content += f". Description: {description[:100]}..."
        if path:
            memory_content += f". Located at: {path}"
        elif final_db_content and final_db_content != content:
            memory_content += ". Content stored (truncated)."
        elif content:
            memory_content += ". Content stored directly."
        if is_output:
            memory_content += ". Marked as a final workflow output."

        mem_tags_set = set(["artifact_creation", artifact_type_enum.value] + artifact_tags_list)
        mem_importance = 6.0 if is_output else 5.0

        await conn.execute(
            """
             INSERT INTO memories (memory_id, workflow_id, action_id, artifact_id, content, memory_level, memory_type,
             importance, confidence, tags, created_at, updated_at, access_count)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             """,
            (
                linked_memory_id,
                workflow_id,
                action_id,
                artifact_id,
                memory_content,
                MemoryLevel.EPISODIC.value,
                MemoryType.ARTIFACT_CREATION.value,
                mem_importance,
                1.0,
                json.dumps(list(mem_tags_set)),
                now_unix,
                now_unix,
                0,
            ),
        )
        logger.debug(f"Inserted linked memory record {linked_memory_id} for artifact {artifact_id}")

        # --- Log Operations (assuming MemoryUtils._log_memory_operation uses passed conn) ---
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
            linked_memory_id,
            action_id,
            {"artifact_id": artifact_id},
        )
        # Transaction will commit here if no exceptions

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after recording artifact '{name}': {te_wal}. Main operation succeeded."
        )

    result = {
        "artifact_id": artifact_id,
        "workflow_id": workflow_id,
        "name": name,
        "artifact_type": artifact_type_enum.value,
        "path": path,
        "content_stored_in_db": bool(
            final_db_content
        ),  # Reflects if content was stored (even if truncated)
        "content_truncated_in_db": bool(
            final_db_content and content and final_db_content != content
        ),
        "created_at": to_iso_z(now_unix),
        "is_output": is_output,
        "tags": artifact_tags_list,
        "linked_memory_id": linked_memory_id,
        "success": True,
        "processing_time": time.time() - start_time,
    }

    logger.info(
        f"Recorded artifact '{name}' ({artifact_id}) and linked memory {linked_memory_id} in workflow {workflow_id}",
        emoji_key="package",
    )
    return result


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
    conn: Optional[Any] = None,  # Keep type hint
) -> Dict[str, Any]:
    """Saves a single reasoning step, goal, decision, or reflection to a thought chain. Optionally links to memory.
    Args: workflow_id, content, thought_type, thought_chain_id, parent_thought_id, relevant_action/artifact/memory_id.
    Returns: thought_id, linked_memory_id (if created)."""
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
    # This will be set by _perform_db_operations if a memory is created
    # Must be declared here to be in scope for the final return.
    final_linked_memory_id: Optional[str] = None

    # Inner helper function remains the same as it correctly uses the passed 'db_conn'
    async def _perform_db_operations(db_conn: aiosqlite.Connection):
        nonlocal final_linked_memory_id  # Allow modification of outer scope variable

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
        # ... (rest of existence checks for relevant_action_id, relevant_artifact_id, relevant_memory_id using db_conn)
        if relevant_action_id:
            async with db_conn.execute(
                "SELECT 1 FROM actions WHERE action_id = ?", (relevant_action_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Relevant action not found: {relevant_action_id}",
                        param_name="relevant_action_id",
                    )
        if relevant_artifact_id:
            async with db_conn.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (relevant_artifact_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Relevant artifact not found: {relevant_artifact_id}",
                        param_name="relevant_artifact_id",
                    )
        if relevant_memory_id:
            async with db_conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?", (relevant_memory_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Relevant memory not found: {relevant_memory_id}",
                        param_name="relevant_memory_id",
                    )

        target_thought_chain_id = thought_chain_id
        if not target_thought_chain_id:
            async with db_conn.execute(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at ASC LIMIT 1",
                (workflow_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    # This is the critical part: if 'conn' (external transaction) is passed, we cannot create a new chain here
                    # because that would require a separate commit or handling within the external transaction's scope.
                    # The original logic `if conn: raise ToolError(...)` was correct.
                    if (
                        conn
                    ):  # Check if an *external* transaction connection was passed to record_thought
                        raise ToolError(
                            f"Primary thought chain for workflow {workflow_id} not found, and cannot auto-create within an existing external transaction."
                        )
                    else:  # No external conn, we are managing our own transaction, so we can create it.
                        target_thought_chain_id = MemoryUtils.generate_id()
                        logger.info(
                            f"No existing thought chain found for workflow {workflow_id}, creating default with ID {target_thought_chain_id}."
                        )
                        await db_conn.execute(
                            "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, ?, ?)",
                            (target_thought_chain_id, workflow_id, "Main reasoning", now_unix),
                        )
                else:
                    target_thought_chain_id = row["thought_chain_id"]
        else:
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

        sequence_number = await MemoryUtils.get_next_sequence_number(
            db_conn, target_thought_chain_id, "thoughts", "thought_chain_id"
        )

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

        await db_conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        important_thought_types = [
            ThoughtType.GOAL.value,
            ThoughtType.DECISION.value,
            ThoughtType.SUMMARY.value,
            ThoughtType.REFLECTION.value,
            ThoughtType.HYPOTHESIS.value,
            ThoughtType.INSIGHT.value,
        ]
        if thought_type_enum.value in important_thought_types:
            final_linked_memory_id = MemoryUtils.generate_id()  # Set the outer scope variable
            mem_content = (
                f"Thought [{sequence_number}] ({thought_type_enum.value.capitalize()}): {content}"
            )
            mem_tags = ["reasoning", thought_type_enum.value]
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
                    final_linked_memory_id,
                    workflow_id,
                    thought_id,
                    mem_content,
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.REASONING_STEP.value,
                    mem_importance,
                    1.0,
                    json.dumps(mem_tags),
                    now_unix,
                    now_unix,
                    0,
                ),
            )
            await MemoryUtils._log_memory_operation(  # Uses the passed db_conn
                db_conn,
                workflow_id,
                "create_from_thought",
                final_linked_memory_id,
                None,
                {"thought_id": thought_id},
            )
        return target_thought_chain_id, sequence_number

    target_thought_chain_id_res: Optional[str] = None
    sequence_number_res: Optional[int] = None
    manage_transaction_locally = conn is None  # True if no external connection was passed

    try:
        if manage_transaction_locally:
            db_manager = DBConnection(db_path)
            async with db_manager.transaction() as local_conn:
                target_thought_chain_id_res, sequence_number_res = await _perform_db_operations(
                    local_conn
                )
            # WAL checkpoint after local transaction commits
            try:
                await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
            except ToolError as te_wal:
                logger.warning(
                    f"Passive WAL checkpoint failed after recording thought for chain {target_thought_chain_id_res}: {te_wal}. Main operation succeeded."
                )
            logger.debug(
                f"Executed record_thought with internal transaction and WAL checkpoint for chain {target_thought_chain_id_res}"
            )
        else:  # Use the provided external connection
            # Type check for safety, though Pylance might complain if conn type is just 'Any' in signature
            if not isinstance(conn, aiosqlite.Connection):
                raise ToolError(
                    "Invalid database connection object passed to record_thought."
                )  # Should not happen with correct usage
            target_thought_chain_id_res, sequence_number_res = await _perform_db_operations(conn)
            # NO COMMIT and NO WAL CHECKPOINT here - handled by the outer transaction manager
            logger.debug(
                f"Executed record_thought within provided transaction for chain {target_thought_chain_id_res}"
            )

        result = {
            "thought_id": thought_id,
            "thought_chain_id": target_thought_chain_id_res,
            "thought_type": thought_type_enum.value,
            "content": content,
            "sequence_number": sequence_number_res,
            "created_at": to_iso_z(now_unix),
            "linked_memory_id": final_linked_memory_id,
            "success": True,
        }
        logger.info(
            f"Recorded thought ({thought_type_enum.value}) in workflow {workflow_id}",
            emoji_key="brain",
        )
        return result

    except ToolInputError:
        logger.warning(
            f"Input error recording thought: {traceback.format_exc(limit=0)}", exc_info=False
        )
        raise
    except ToolError as te:
        logger.error(f"Tool error recording thought: {te}", exc_info=True)
        raise te
    except Exception as e:
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
    ttl: Optional[int] = None,  # Use config default if None
    context_data: Optional[Dict[str, Any]] = None,
    generate_embedding: bool = True,
    suggest_links: bool = True,
    link_suggestion_threshold: float = agent_memory_config.similarity_threshold,
    max_suggested_links: int = 3,
    action_id: Optional[str] = None,
    thought_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Adds a new piece of knowledge, fact, or observation to UMS memory.
    Args: workflow_id, content, memory_type, memory_level, importance, confidence, description, reasoning, source, tags, ttl, context_data, generate_embedding, suggest_links, action_id, thought_id, artifact_id.
    Returns: memory_id, embedding_id, suggested_links."""
    # Parameter validation (remains the same)
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
    # now_iso removed as it was unused for DB, formatting happens at return

    start_time = time.time()
    embedding_db_id_final: Optional[str] = None
    suggested_links_list_final: List[Dict[str, Any]] = []

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Use transaction context manager
        # --- 1. Existence checks for foreign keys ---
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")
        if action_id:
            async with conn.execute(
                "SELECT 1 FROM actions WHERE action_id = ?", (action_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(f"Action {action_id} not found", param_name="action_id")
        if thought_id:
            async with conn.execute(
                "SELECT 1 FROM thoughts WHERE thought_id = ?", (thought_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(f"Thought {thought_id} not found", param_name="thought_id")
        if artifact_id:
            async with conn.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Artifact {artifact_id} not found", param_name="artifact_id"
                    )

        # --- Prepare tags and TTL ---
        # Correctly add memory_type and memory_level to tags
        base_tags = [str(t).lower().strip() for t in (tags or []) if str(t).strip()]
        final_tags_list = list(set(base_tags + [mem_type.value, mem_level.value]))
        final_tags_json = json.dumps(final_tags_list)

        # Set TTL based on memory level if not explicitly provided
        effective_ttl = ttl  # Retain explicitly passed ttl
        if ttl is None:  # Only apply default if ttl is not passed at all
            if mem_level == MemoryLevel.WORKING:
                effective_ttl = agent_memory_config.ttl_working
            elif mem_level == MemoryLevel.EPISODIC:
                effective_ttl = agent_memory_config.ttl_episodic
            # SEMANTIC and PROCEDURAL usually have ttl=0 (no expiry) by default, handled by schema.
            # If you want specific defaults for them if ttl=None, add here.
            else:  # Default for SEMANTIC, PROCEDURAL, or any other
                effective_ttl = 0  # No expiry
        else:  # ttl was passed, use it (could be 0 for no expiry)
            effective_ttl = int(ttl)

        # --- 2. Insert the main memory record ---
        await conn.execute(
            """
            INSERT INTO memories (memory_id, workflow_id, content, memory_level, memory_type, importance, confidence,
            description, reasoning, source, context, tags, created_at, updated_at, last_accessed, access_count, ttl,
            action_id, thought_id, artifact_id, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?, NULL)
            """,  # last_accessed=NULL, access_count=0, embedding_id=NULL initially
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
                final_tags_json,
                now_unix,
                now_unix,
                effective_ttl,
                action_id,
                thought_id,
                artifact_id,
            ),
        )

        # --- 3. Generate and store embedding (if requested) ---
        embedding_generated_successfully = False
        if generate_embedding:
            text_for_embedding = f"{description}: {content}" if description else content
            try:
                # _store_embedding now uses the passed 'conn'
                embedding_db_id_final = await _store_embedding(conn, memory_id, text_for_embedding)
                if embedding_db_id_final:
                    embedding_generated_successfully = True
                    logger.debug(
                        f"Successfully generated embedding {embedding_db_id_final} for memory {memory_id}"
                    )
                else:
                    logger.warning(f"Embedding generation skipped or failed for memory {memory_id}")
            except Exception as embed_err:
                logger.error(
                    f"Error during embedding generation/storage for memory {memory_id}: {embed_err}",
                    exc_info=True,
                )
                # Continue, but embedding won't be available

        # --- 4. Suggest Semantic Links (if requested and embedding succeeded) ---
        if suggest_links and embedding_generated_successfully and max_suggested_links > 0:
            logger.debug(
                f"Attempting to find similar memories for link suggestion (threshold={link_suggestion_threshold})..."
            )
            try:
                text_for_search = f"{description}: {content}" if description else content
                # _find_similar_memories uses the passed 'conn'
                similar_memories = await _find_similar_memories(
                    conn=conn,
                    query_text=text_for_search,
                    workflow_id=workflow_id,
                    limit=max_suggested_links + 1,
                    threshold=link_suggestion_threshold,
                )
                if similar_memories:
                    # ... (rest of the link suggestion logic using 'conn' remains the same as original) ...
                    # This part primarily reads, so direct transaction use is less critical than writes.
                    # It was:
                    similar_ids = [sim_id for sim_id, _ in similar_memories if sim_id != memory_id][
                        :max_suggested_links
                    ]
                    if similar_ids:
                        placeholders = ",".join("?" * len(similar_ids))
                        async with conn.execute(
                            f"SELECT memory_id, description, memory_type FROM memories WHERE memory_id IN ({placeholders})",
                            similar_ids,
                        ) as cursor:
                            target_details_rows = await cursor.fetchall()
                            target_details = {
                                row["memory_id"]: dict(row) for row in target_details_rows
                            }

                        score_map = dict(similar_memories)
                        for sim_id_val in similar_ids:  # Renamed to avoid conflict
                            if sim_id_val in target_details:
                                details = target_details[sim_id_val]
                                similarity = score_map.get(sim_id_val, 0.0)
                                suggested_type = LinkType.RELATED.value  # Default
                                # ... (your existing logic for suggested_type)
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

                                suggested_links_list_final.append(
                                    {
                                        "target_memory_id": sim_id_val,
                                        "target_description": details.get("description", ""),
                                        "target_type": details.get("memory_type", ""),
                                        "similarity": round(similarity, 4),
                                        "suggested_link_type": suggested_type,
                                    }
                                )
                        logger.info(
                            f"Generated {len(suggested_links_list_final)} link suggestions for memory {memory_id}"
                        )
            except Exception as link_err:
                logger.error(
                    f"Error suggesting links for memory {memory_id}: {link_err}", exc_info=True
                )

        # --- 5. Update Workflow Timestamp ---
        # Use now_unix directly, as now_iso was removed. 'last_active' expects Unix timestamp.
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # --- 6. Log Operation ---
        # _log_memory_operation now uses the passed 'conn'
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
                "links_suggested": len(suggested_links_list_final),
                "tags": final_tags_list,
            },
        )
        # Transaction will commit here if no exceptions

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after storing memory '{memory_id}': {te_wal}. Main operation succeeded."
        )

    # --- 8. Prepare Result (Timestamps formatted here) ---
    processing_time = time.time() - start_time
    result = {
        "memory_id": memory_id,
        "workflow_id": workflow_id,
        "memory_level": mem_level.value,
        "memory_type": mem_type.value,
        "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
        "importance": importance,
        "confidence": confidence,
        "created_at": to_iso_z(now_unix),  # Format for output
        "tags": final_tags_list,
        "embedding_id": embedding_db_id_final,  # Use the one set during embedding
        "linked_action_id": action_id,
        "linked_thought_id": thought_id,
        "linked_artifact_id": artifact_id,
        "suggested_links": suggested_links_list_final,
        "success": True,
        "processing_time": processing_time,
    }
    logger.info(
        f"Stored memory {memory_id} ({mem_type.value}) in workflow {workflow_id}. Links suggested: {len(suggested_links_list_final)}.",
        emoji_key="floppy_disk",
        time=processing_time,
    )
    return result


@with_tool_metrics
@with_error_handling
async def get_memory_by_id(
    memory_id: str,
    include_links: bool = True,
    include_context: bool = True,
    context_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves a specific UMS memory entry by its unique ID, with optional links/context.
    Args: memory_id, include_links, include_context, context_limit.
    Returns: Full memory object with details."""
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    start_time = time.time()
    result_memory_dict: Optional[Dict[str, Any]] = None  # Initialize

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Use transaction context manager
        # --- 1. Fetch Core Memory Data ---
        async with conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                # No transaction commit needed if memory not found, as no writes happened.
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
            result_memory_dict = dict(row)

        if result_memory_dict is None:  # Should be caught by above, but defensive
            raise ToolError(
                f"Internal error: Memory data for {memory_id} not loaded after initial fetch."
            )

        # --- 2. Check TTL (within transaction, before further processing) ---
        if result_memory_dict.get("ttl", 0) > 0:
            expiry_time = result_memory_dict["created_at"] + result_memory_dict["ttl"]
            if expiry_time <= int(time.time()):
                # Delete the expired memory within the current transaction
                logger.warning(f"Memory {memory_id} is expired. Deleting it now.")
                await conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                # Logging of deletion happens within this transaction.
                # No need to explicitly call _log_memory_operation here for this specific "access then expire"
                # unless you have a very specific "accessed_expired_memory" log type.
                # The delete_expired_memories tool handles batch expiration logging.
                # Let the transaction commit this deletion.
                raise ToolError(
                    f"Memory {memory_id} has expired and was deleted."
                )  # Signal that it was found then expired

        # --- 3. Parse JSON Fields ---
        result_memory_dict["tags"] = await MemoryUtils.deserialize(result_memory_dict.get("tags"))
        result_memory_dict["context"] = await MemoryUtils.deserialize(
            result_memory_dict.get("context")
        )

        # --- 4. Update Access Statistics (within the transaction) ---
        await MemoryUtils._update_memory_access(conn, memory_id)  # Pass conn
        await MemoryUtils._log_memory_operation(
            conn,
            result_memory_dict["workflow_id"],
            "access_by_id",
            memory_id,  # Pass conn
        )

        # --- 5. Fetch Links (Incoming & Outgoing) ---
        result_memory_dict["outgoing_links"] = []
        result_memory_dict["incoming_links"] = []
        if include_links:
            # Outgoing Links
            outgoing_query = """
            SELECT ml.link_id, ml.target_memory_id, ml.link_type, ml.strength, ml.description,
                   m.description AS target_description, m.memory_type AS target_type, ml.created_at AS link_created_at
            FROM memory_links ml
            JOIN memories m ON ml.target_memory_id = m.memory_id
            WHERE ml.source_memory_id = ?
            """
            async with conn.execute(outgoing_query, (memory_id,)) as cursor:
                async for link_row_data in cursor:
                    link_row = dict(link_row_data)
                    result_memory_dict["outgoing_links"].append(
                        {
                            "link_id": link_row["link_id"],
                            "target_memory_id": link_row["target_memory_id"],
                            "target_description": link_row["target_description"],
                            "target_type": link_row["target_type"],
                            "link_type": link_row["link_type"],
                            "strength": link_row["strength"],
                            "description": link_row["description"],
                            "created_at": safe_format_timestamp(
                                link_row.get("link_created_at")
                            ),  # Format link timestamp
                        }
                    )
            # Incoming Links
            incoming_query = """
            SELECT ml.link_id, ml.source_memory_id, ml.link_type, ml.strength, ml.description,
                   m.description AS source_description, m.memory_type AS source_type, ml.created_at AS link_created_at
            FROM memory_links ml
            JOIN memories m ON ml.source_memory_id = m.memory_id
            WHERE ml.target_memory_id = ?
            """
            async with conn.execute(incoming_query, (memory_id,)) as cursor:
                async for link_row_data in cursor:
                    link_row = dict(link_row_data)
                    result_memory_dict["incoming_links"].append(
                        {
                            "link_id": link_row["link_id"],
                            "source_memory_id": link_row["source_memory_id"],
                            "source_description": link_row["source_description"],
                            "source_type": link_row["source_type"],
                            "link_type": link_row["link_type"],
                            "strength": link_row["strength"],
                            "description": link_row["description"],
                            "created_at": safe_format_timestamp(
                                link_row.get("link_created_at")
                            ),  # Format link timestamp
                        }
                    )

        # --- 6. Fetch Semantic Context (Reads, so okay within transaction) ---
        result_memory_dict["semantic_context"] = []
        if include_context and result_memory_dict.get("embedding_id"):
            search_text = result_memory_dict.get("content", "")
            if result_memory_dict.get("description"):
                search_text = f"{result_memory_dict['description']}: {search_text}"

            if search_text:
                try:
                    similar_results = await _find_similar_memories(  # This function needs 'conn'
                        conn=conn,
                        query_text=search_text,
                        workflow_id=result_memory_dict["workflow_id"],
                        limit=context_limit + 1,
                        threshold=agent_memory_config.similarity_threshold * 0.9,
                    )
                    if similar_results:
                        similar_ids = [
                            mem_id for mem_id, _ in similar_results if mem_id != memory_id
                        ][:context_limit]
                        score_map = dict(similar_results)
                        if similar_ids:
                            placeholders = ",".join(["?"] * len(similar_ids))
                            context_query = "SELECT memory_id, description, memory_type, importance FROM memories WHERE memory_id IN ({})".format(
                                placeholders
                            )
                            async with conn.execute(context_query, similar_ids) as context_cursor:
                                context_rows = await context_cursor.fetchall()
                                ordered_context = sorted(
                                    context_rows,
                                    key=lambda r: score_map.get(r["memory_id"], -1.0),
                                    reverse=True,
                                )
                                for context_row_data in ordered_context:
                                    context_row = dict(context_row_data)
                                    result_memory_dict["semantic_context"].append(
                                        {
                                            "memory_id": context_row["memory_id"],
                                            "description": context_row["description"],
                                            "memory_type": context_row["memory_type"],
                                            "importance": context_row["importance"],
                                            "similarity": round(
                                                score_map.get(context_row["memory_id"], 0.0), 4
                                            ),
                                        }
                                    )
                except Exception as context_err:
                    logger.warning(
                        f"Could not retrieve semantic context for memory {memory_id}: {context_err}",
                        exc_info=True,
                    )
        # Transaction will commit here if no exceptions

    # WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after retrieving memory {memory_id}: {te_wal}. Main operation succeeded."
        )

    if (
        result_memory_dict is None
    ):  # Should not happen if ToolInputError wasn't raised for not found
        raise ToolError(
            f"Internal error: Memory data for {memory_id} became None after transaction."
        )

    # --- 7. Finalize and Return (Format timestamps AFTER transaction) ---
    for ts_key in ["created_at", "updated_at", "last_accessed"]:
        if ts_key in result_memory_dict:
            result_memory_dict[ts_key] = safe_format_timestamp(result_memory_dict.get(ts_key))

    # Note: Link timestamps were already formatted inside the loop.
    # Semantic context does not typically include timestamps in its summary.

    result_memory_dict["success"] = True
    result_memory_dict["processing_time"] = time.time() - start_time

    logger.info(
        f"Retrieved memory {_fmt_id(memory_id)} with links={include_links}, context={include_context}",
        emoji_key="inbox_tray",
        time=result_memory_dict["processing_time"],
    )
    return result_memory_dict


@with_tool_metrics
@with_error_handling
async def search_semantic_memories(
    query: str,
    workflow_id: Optional[str] = None,
    limit: int = 5,
    threshold: float = agent_memory_config.similarity_threshold,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    include_content: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Finds UMS memories semantically similar to a query text using embeddings.
    Args: query, workflow_id, limit, threshold, memory_level, memory_type, include_content.
    Returns: List of matching memory objects with similarity scores."""
    # --- Input validation (remains the same) ---
    if not query:
        raise ToolInputError("Search query required.", param_name="query")
    if not isinstance(limit, int) or limit < 1:
        raise ToolInputError("Limit must be positive integer.", param_name="limit")
    if not 0.0 <= threshold <= 1.0:
        raise ToolInputError("Threshold must be 0.0-1.0.", param_name="threshold")
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

    start_time = time.time()
    memories_data_list: List[Dict[str, Any]] = []  # Initialize

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Use transaction context manager
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
        # _find_similar_memories uses 'conn' internally for its reads.
        similar_results: List[Tuple[str, float]] = await _find_similar_memories(
            conn=conn,
            query_text=query,
            workflow_id=workflow_id,
            limit=limit,
            threshold=threshold,
            memory_level=memory_level,
            memory_type=memory_type,
        )

        if not similar_results:
            logger.info(
                f"Semantic search for '{query[:50]}...' found no results matching filters above threshold {threshold}.",
                emoji_key="zzz",
            )
            # No DB writes occurred yet, transaction will just close.
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
        select_cols = "memory_id, workflow_id, description, memory_type, memory_level, importance, confidence, created_at, updated_at, last_accessed, access_count, ttl, tags, action_id, thought_id, artifact_id"
        if include_content:
            select_cols += ", content"
        score_map = dict(similar_results)

        async with conn.execute(
            f"SELECT {select_cols} FROM memories WHERE memory_id IN ({placeholders})",
            memory_ids,
        ) as cursor:
            rows = await cursor.fetchall()

            ordered_rows_data = sorted(
                [dict(r) for r in rows],  # Convert to dicts first
                key=lambda r_dict: (score_map.get(r_dict["memory_id"], -1.0), r_dict["memory_id"]),
                reverse=True,
            )

            # --- Step 3: Process results, update access stats (within transaction) ---
            # Prepare for batch updates
            batch_update_params: List[Tuple[Any, ...]] = []
            batch_log_params: List[Tuple[Any, ...]] = []
            now_unix_batch = int(time.time())

            for row_data in ordered_rows_data:
                mem_dict = dict(row_data)  # Ensure it's a dictionary
                mem_dict["similarity"] = round(score_map.get(row_data["memory_id"], 0.0), 4)
                mem_dict["tags"] = await MemoryUtils.deserialize(mem_dict.get("tags"))

                # Format timestamps AFTER transaction for the final output
                # Store original Unix timestamps for now if needed by other logic
                mem_dict["created_at_unix_original"] = mem_dict.get("created_at")
                mem_dict["updated_at_unix_original"] = mem_dict.get("updated_at")
                mem_dict["last_accessed_unix_original"] = mem_dict.get("last_accessed")

                memories_data_list.append(mem_dict)

                # Prepare parameters for batch DB updates
                batch_update_params.append((now_unix_batch, row_data["memory_id"]))
                batch_log_params.append(
                    (
                        row_data["workflow_id"],
                        row_data["memory_id"],
                        None,  # action_id for log
                        json.dumps(
                            {"query": query[:100], "score": mem_dict["similarity"]}
                        ),  # operation_data
                        now_unix_batch,
                    )
                )
                if len(memories_data_list) >= limit:
                    break

            # Perform batch updates and logging within the transaction
            if batch_update_params:
                update_sql = """
                    UPDATE memories
                    SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1
                    WHERE memory_id = ?
                """
                await conn.executemany(update_sql, batch_update_params)

                log_sql = """
                    INSERT INTO memory_operations
                    (operation_log_id, workflow_id, memory_id, action_id, operation, operation_data, timestamp)
                    VALUES (?, ?, ?, ?, 'semantic_access', ?, ?)
                """
                # Prepend generated ID for log_sql
                log_params_with_ids = [
                    (MemoryUtils.generate_id(),) + params for params in batch_log_params
                ]
                await conn.executemany(log_sql, log_params_with_ids)
        # Transaction commits here

    # WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after semantic search: {te_wal}. Main operation succeeded."
        )

    # Format timestamps for the final output AFTER transaction
    for mem_dict_final in memories_data_list:
        for ts_key, original_ts_key in [
            ("created_at", "created_at_unix_original"),
            ("updated_at", "updated_at_unix_original"),
            (
                "last_accessed",
                "last_accessed_unix_original",
            ),  # last_accessed is updated to now_unix_batch in DB
        ]:
            # For created_at and updated_at, use original values for formatting
            # For last_accessed, it's now_unix_batch if updated, or original if not.
            # Since we batch updated, all last_accessed in memories_data_list would be now_unix_batch.
            if ts_key == "last_accessed":
                mem_dict_final[ts_key] = safe_format_timestamp(now_unix_batch)
            else:
                mem_dict_final[ts_key] = safe_format_timestamp(mem_dict_final.get(original_ts_key))
            mem_dict_final.pop(original_ts_key, None)  # Remove the temporary original key

    processing_time = time.time() - start_time
    logger.info(
        f"Semantic search found {len(memories_data_list)} results for query: '{query[:50]}...'",
        emoji_key="mag",
        time=processing_time,
    )

    return {
        "memories": memories_data_list,
        "query": query,
        "workflow_id": workflow_id,
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def hybrid_search_memories(
    query: str,
    workflow_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_importance: Optional[float] = None,
    max_importance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_created_at_unix: Optional[int] = None,
    max_created_at_unix: Optional[int] = None,
    include_content: bool = True,
    include_links: bool = False,
    link_direction: str = "outgoing",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Searches UMS memories combining keyword (FTS) and semantic similarity with filters.
    Args: query, workflow_id, limit, offset, semantic_weight, keyword_weight, filters (level, type, tags, etc.), include_content, include_links.
    Returns: List of ranked matching memory objects with scores."""
    start_time = time.time()

    # --- Input Validation (Copied from query_memories, ensure it's thorough) ---
    if not query:
        raise ToolInputError("Query string cannot be empty.", param_name="query")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ToolInputError("semantic_weight must be 0.0-1.0", param_name="semantic_weight")
    if not 0.0 <= keyword_weight <= 1.0:
        raise ToolInputError("keyword_weight must be 0.0-1.0", param_name="keyword_weight")
    if semantic_weight + keyword_weight <= 0 and (
        semantic_weight > 0 or keyword_weight > 0
    ):  # Allow one to be zero if other is positive
        raise ToolInputError(
            "Sum of weights must be positive if any weight is positive.",
            param_name="semantic_weight",
        )
    if limit < 1:
        raise ToolInputError("Limit must be >= 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("Offset must be >= 0", param_name="offset")
    if memory_level:
        try:
            MemoryLevel(memory_level.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid memory_level: {memory_level}", param_name="memory_level"
            ) from e
    if memory_type:
        try:
            MemoryType(memory_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid memory_type: {memory_type}", param_name="memory_type"
            ) from e

    valid_link_directions = ["outgoing", "incoming", "both"]
    link_direction_lower = link_direction.lower()
    if link_direction_lower not in valid_link_directions:
        raise ToolInputError(
            f"link_direction must be one of: {valid_link_directions}", param_name="link_direction"
        )

    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    norm_sem_weight = (
        (semantic_weight / total_weight) if total_weight > 0 else 0.5
    )  # Default to equal if both zero (though disallowed by check above)
    norm_key_weight = (keyword_weight / total_weight) if total_weight > 0 else 0.5

    combined_scores: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"semantic": 0.0, "keyword": 0.0, "hybrid": 0.0}
    )
    memories_results: List[Dict[str, Any]] = []
    total_candidates_considered = 0

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # --- Step 1: Semantic Search ---
        semantic_candidate_ids_scores: Dict[str, float] = {}
        if norm_sem_weight > 0:
            try:
                semantic_candidate_limit = min(
                    max(limit * 10, 100), agent_memory_config.max_semantic_candidates
                )  # Fetch more for better ranking
                # _find_similar_memories now returns List[Tuple[str, float]]
                semantic_search_results_tuples = await _find_similar_memories(
                    conn=conn,
                    query_text=query,
                    workflow_id=workflow_id,
                    limit=semantic_candidate_limit,
                    threshold=0.1,  # Lower threshold for candidate gathering
                    memory_level=memory_level,
                    memory_type=memory_type,
                )
                semantic_candidate_ids_scores = dict(semantic_search_results_tuples)
                for mem_id, score in semantic_candidate_ids_scores.items():
                    combined_scores[mem_id]["semantic"] = score
                logger.debug(
                    f"Hybrid search: Found {len(semantic_candidate_ids_scores)} semantic candidates."
                )
            except Exception as sem_err:
                logger.warning(
                    f"Hybrid search: Semantic component failed: {sem_err}", exc_info=True
                )

        # --- Step 2: Keyword/Filtered Search & Relevance Score ---
        keyword_candidate_relevance_scores: Dict[str, float] = {}
        if norm_key_weight > 0:
            select_cols_kw = "m.memory_id, m.importance, m.confidence, m.created_at, m.access_count, m.last_accessed"
            data_query_kw = f"SELECT {select_cols_kw} FROM memories m"
            where_clauses_kw = ["1=1"]
            params_kw: List[Any] = []
            fts_params_kw: List[Any] = []
            joins_kw = ""

            # Apply filters (similar to query_memories)
            if workflow_id:
                where_clauses_kw.append("m.workflow_id = ?")
                params_kw.append(workflow_id)
            if memory_level:
                where_clauses_kw.append("m.memory_level = ?")
                params_kw.append(memory_level.lower())
            # ... (all other filters: memory_type, min/max_importance, min_confidence, created_at, ttl, tags) ...
            if memory_type:
                where_clauses_kw.append("m.memory_type = ?")
                params_kw.append(memory_type.lower())
            if min_importance is not None:
                where_clauses_kw.append("m.importance >= ?")
                params_kw.append(min_importance)
            if max_importance is not None:
                where_clauses_kw.append("m.importance <= ?")
                params_kw.append(max_importance)
            if min_confidence is not None:
                where_clauses_kw.append("m.confidence >= ?")
                params_kw.append(min_confidence)
            if min_created_at_unix is not None:
                where_clauses_kw.append("m.created_at >= ?")
                params_kw.append(min_created_at_unix)
            if max_created_at_unix is not None:
                where_clauses_kw.append("m.created_at <= ?")
                params_kw.append(max_created_at_unix)

            current_time_unix_kw = int(time.time())
            where_clauses_kw.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
            params_kw.append(current_time_unix_kw)

            if tags and isinstance(tags, list) and len(tags) > 0:
                clean_tags_kw = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
                if clean_tags_kw:
                    tags_json_kw = json.dumps(clean_tags_kw)
                    where_clauses_kw.append("json_contains_all(m.tags, ?)")
                    params_kw.append(tags_json_kw)

            if query:  # Use the original, un-sanitized query for FTS, FTS handles it
                if "memory_fts" not in joins_kw:
                    joins_kw += " JOIN memory_fts fts ON m.rowid = fts.rowid"
                # FTS engines usually handle tokenization and operators. Pass query as is.
                # Simple split might be too naive if query contains FTS syntax.
                # For `porter unicode61` tokenizer, it handles terms well.
                # If more advanced FTS query construction is needed, it would be here.
                sanitized_query_for_fts_match = re.sub(
                    r'[^a-zA-Z0-9\s*+\-"]', "", query
                ).strip()  # Allow FTS operators
                fts_terms_for_match = [
                    term for term in sanitized_query_for_fts_match.split() if term
                ]

                if fts_terms_for_match:
                    where_clauses_kw.append("fts.memory_fts MATCH ?")
                    # Pass the sanitized query that retains operators
                    fts_params_kw.append(sanitized_query_for_fts_match)
                else:
                    logger.debug(
                        f"Hybrid FTS: Query '{query}' resulted in no valid FTS terms after sanitization. FTS match omitted."
                    )

            where_sql_kw = (
                " WHERE " + " AND ".join(where_clauses_kw) if len(where_clauses_kw) > 1 else ""
            )
            final_query_kw = data_query_kw + joins_kw + where_sql_kw
            # Potentially add a LIMIT here for keyword candidates if performance is an issue
            # final_query_kw += " LIMIT 200" # Example: Limit keyword candidates

            async with conn.execute(final_query_kw, params_kw + fts_params_kw) as cursor:
                keyword_candidate_rows = await cursor.fetchall()
                for row in keyword_candidate_rows:
                    mem_id_kw = row["memory_id"]
                    kw_relevance = _compute_memory_relevance(
                        row["importance"],
                        row["confidence"],
                        row["created_at"],
                        row["access_count"],
                        row["last_accessed"],
                    )
                    keyword_candidate_relevance_scores[mem_id_kw] = kw_relevance
                    combined_scores[mem_id_kw]  # Ensure entry for this ID exists

            # Normalize keyword relevance scores (0-1 range)
            max_kw_relevance = (
                max(keyword_candidate_relevance_scores.values())
                if keyword_candidate_relevance_scores
                else 0.0
            )
            norm_factor_kw = max(max_kw_relevance, 1e-6)  # Avoid division by zero
            for mem_id_kw, raw_rel_score in keyword_candidate_relevance_scores.items():
                normalized_kw_score = min(max(raw_rel_score / norm_factor_kw, 0.0), 1.0)
                combined_scores[mem_id_kw]["keyword"] = normalized_kw_score
            logger.debug(
                f"Hybrid search: Found and scored {len(keyword_candidate_rows)} keyword candidates."
            )

        # --- Step 3: Calculate Hybrid Score ---
        if not combined_scores:
            logger.info("Hybrid search: No candidates from semantic or keyword search.")
        else:
            for _mem_id_hybrid, scores_hybrid in combined_scores.items():
                scores_hybrid["hybrid"] = (scores_hybrid["semantic"] * norm_sem_weight) + (
                    scores_hybrid["keyword"] * norm_key_weight
                )

        sorted_hybrid_results = sorted(
            combined_scores.items(), key=lambda item: item[1]["hybrid"], reverse=True
        )
        total_candidates_considered = len(sorted_hybrid_results)
        paginated_hybrid_results = sorted_hybrid_results[offset : offset + limit]
        final_ranked_ids_with_scores = {item[0]: item[1] for item in paginated_hybrid_results}
        final_ranked_ids_list = [item[0] for item in paginated_hybrid_results]

        # --- Step 4: Fetch Full Details for Ranked IDs ---
        db_fetched_rows_map: Dict[str, Dict[str, Any]] = {}
        if final_ranked_ids_list:
            placeholders_final = ",".join("?" * len(final_ranked_ids_list))
            select_cols_final_parts = [
                "m.memory_id",
                "m.workflow_id",
                "m.memory_level",
                "m.memory_type",
                "m.importance",
                "m.confidence",
                "m.description",
                "m.reasoning",
                "m.source",
                "m.tags",
                "m.created_at",
                "m.updated_at",
                "m.last_accessed",
                "m.access_count",
                "m.ttl",
                "m.action_id",
                "m.thought_id",
                "m.artifact_id",
            ]
            if include_content:
                select_cols_final_parts.append("m.content")
            select_cols_final_str = ", ".join(select_cols_final_parts)

            query_final_details = f"SELECT {select_cols_final_str} FROM memories m WHERE m.memory_id IN ({placeholders_final})"
            async with conn.execute(query_final_details, final_ranked_ids_list) as cursor:
                db_fetched_rows_map = {
                    row["memory_id"]: dict(row) for row in await cursor.fetchall()
                }

        # --- Step 5: Batch Fetch Links (within transaction, for selected final IDs) ---
        links_map_for_final_results = defaultdict(lambda: {"outgoing": [], "incoming": []})
        if include_links and final_ranked_ids_list:
            placeholders_links_final = ",".join("?" * len(final_ranked_ids_list))
            # Outgoing
            if link_direction_lower in ["outgoing", "both"]:
                q_out_links = f"""
                SELECT ml.*, target_mem.description AS target_description, target_mem.memory_type AS target_type
                FROM memory_links ml JOIN memories target_mem ON ml.target_memory_id = target_mem.memory_id
                WHERE ml.source_memory_id IN ({placeholders_links_final})
                """
                async with conn.execute(q_out_links, final_ranked_ids_list) as cur:
                    async for lr in cur:
                        links_map_for_final_results[lr["source_memory_id"]]["outgoing"].append(
                            dict(lr)
                        )
            # Incoming
            if link_direction_lower in ["incoming", "both"]:
                q_in_links = f"""
                SELECT ml.*, source_mem.description AS source_description, source_mem.memory_type AS source_type
                FROM memory_links ml JOIN memories source_mem ON ml.source_memory_id = source_mem.memory_id
                WHERE ml.target_memory_id IN ({placeholders_links_final})
                """
                async with conn.execute(q_in_links, final_ranked_ids_list) as cur:
                    async for lr in cur:
                        links_map_for_final_results[lr["target_memory_id"]]["incoming"].append(
                            dict(lr)
                        )

        # --- Step 6: Reconstruct Results, Update Access, Log (within transaction) ---
        batch_update_access_params_hybrid: List[Tuple[int, str]] = []
        batch_log_op_params_hybrid: List[Tuple[str, str, str, Optional[str], Dict[str, Any]]] = []
        current_time_unix_hybrid_access = int(time.time())

        for mem_id_final_ranked in final_ranked_ids_list:  # Iterate in ranked order
            if mem_id_final_ranked in db_fetched_rows_map:
                memory_dict_reconstruct = db_fetched_rows_map[mem_id_final_ranked]
                scores_for_this_mem = final_ranked_ids_with_scores.get(mem_id_final_ranked, {})

                memory_dict_reconstruct["hybrid_score"] = round(
                    scores_for_this_mem.get("hybrid", 0.0), 4
                )
                memory_dict_reconstruct["semantic_score"] = round(
                    scores_for_this_mem.get("semantic", 0.0), 4
                )
                memory_dict_reconstruct["keyword_relevance_score"] = round(
                    scores_for_this_mem.get("keyword", 0.0), 4
                )
                memory_dict_reconstruct["tags"] = await MemoryUtils.deserialize(
                    memory_dict_reconstruct.get("tags")
                )

                # Timestamps formatted LATER

                if include_links:
                    memory_dict_reconstruct["links"] = {"outgoing": [], "incoming": []}
                    for link_data_raw_out in links_map_for_final_results[mem_id_final_ranked].get(
                        "outgoing", []
                    ):
                        link_data_out_copy = dict(link_data_raw_out)  # Timestamps formatted later
                        memory_dict_reconstruct["links"]["outgoing"].append(link_data_out_copy)
                    for link_data_raw_in in links_map_for_final_results[mem_id_final_ranked].get(
                        "incoming", []
                    ):
                        link_data_in_copy = dict(link_data_raw_in)  # Timestamps formatted later
                        memory_dict_reconstruct["links"]["incoming"].append(link_data_in_copy)

                memories_results.append(memory_dict_reconstruct)

                # Prepare for batch access update and logging
                wf_id_for_log_hybrid = memory_dict_reconstruct.get(
                    "workflow_id", "unknown_workflow"
                )
                batch_update_access_params_hybrid.append(
                    (current_time_unix_hybrid_access, mem_id_final_ranked)
                )
                batch_log_op_params_hybrid.append(
                    (
                        wf_id_for_log_hybrid,
                        "hybrid_access",
                        mem_id_final_ranked,
                        None,
                        {
                            "query": query[:100],
                            "hybrid_score": memory_dict_reconstruct["hybrid_score"],
                        },
                    )
                )

        # Perform batch updates for access and logging
        if batch_update_access_params_hybrid:
            update_access_sql_hybrid = """
                UPDATE memories SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1
                WHERE memory_id = ?
            """
            await conn.executemany(update_access_sql_hybrid, batch_update_access_params_hybrid)
            for log_params_tuple_hybrid in batch_log_op_params_hybrid:
                await MemoryUtils._log_memory_operation(conn, *log_params_tuple_hybrid)
    # Transaction commits here

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after hybrid search: {te_wal}. Main operation succeeded."
        )

    # --- Final Formatting for Output (Timestamps) ---
    for mem_dict_final_format_hybrid in memories_results:
        for ts_key_hybrid in ["created_at", "updated_at", "last_accessed"]:
            if (
                ts_key_hybrid in mem_dict_final_format_hybrid
                and mem_dict_final_format_hybrid[ts_key_hybrid] is not None
            ):
                mem_dict_final_format_hybrid[ts_key_hybrid] = safe_format_timestamp(
                    mem_dict_final_format_hybrid[ts_key_hybrid]
                )

        if include_links and "links" in mem_dict_final_format_hybrid:
            for direction_key_hybrid in ["outgoing", "incoming"]:
                for link_item_hybrid in mem_dict_final_format_hybrid["links"].get(
                    direction_key_hybrid, []
                ):
                    if link_item_hybrid.get("created_at") is not None:
                        link_item_hybrid["created_at"] = safe_format_timestamp(
                            link_item_hybrid["created_at"]
                        )

    processing_time = time.time() - start_time
    logger.info(
        f"Hybrid search returned {len(memories_results)} results for '{query[:50]}...'",
        emoji_key="magic_wand",
        time=processing_time,
    )
    return {
        "memories": memories_results,
        "total_candidates_considered": total_candidates_considered,
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def create_memory_link(
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,
    strength: float = 1.0,
    description: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Creates a typed association (e.g., related, supports) between two UMS memories.
    Args: source_memory_id, target_memory_id, link_type, strength, description.
    Returns: link_id."""
    if not source_memory_id:
        raise ToolInputError("Source memory ID required.", param_name="source_memory_id")
    if not target_memory_id:
        raise ToolInputError("Target memory ID required.", param_name="target_memory_id")
    if source_memory_id == target_memory_id:
        raise ToolInputError("Cannot link memory to itself.", param_name="source_memory_id")
    try:
        link_type_enum = LinkType(link_type.lower())
    except ValueError as e:
        valid_types_str = ", ".join([lt.value for lt in LinkType])
        raise ToolInputError(
            f"Invalid link_type. Must be one of: {valid_types_str}",
            param_name="link_type",
        ) from e
    if not 0.0 <= strength <= 1.0:
        raise ToolInputError("Strength must be 0.0-1.0.", param_name="strength")

    link_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    start_time = time.time()
    workflow_id: Optional[str] = None  # To store workflow_id for logging

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # Check source memory exists and get workflow_id
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

        # Check target memory exists (and implicitly belongs to some workflow)
        async with conn.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?", (target_memory_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(
                    f"Target memory {target_memory_id} not found.",
                    param_name="target_memory_id",
                )

        # Insert or Replace link (handle existing links gracefully if unique constraint is on source,target,type)
        # If your unique constraint is just on link_id (PK), and you want to allow multiple links of same type
        # between same memories, use INSERT. If you want to update an existing link of same type,
        # a different approach (SELECT then UPDATE or INSERT) would be needed if not using ON CONFLICT.
        # The schema has UNIQUE(source_memory_id, target_memory_id, link_type), so INSERT OR REPLACE is appropriate.
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

        if workflow_id:  # workflow_id should be set if source memory was found
            await MemoryUtils._log_memory_operation(
                conn,  # Pass the transaction connection
                workflow_id,
                "link_created",  # More specific operation name
                source_memory_id,
                None,  # action_id not directly relevant
                {
                    "target_memory_id": target_memory_id,
                    "link_type": link_type_enum.value,
                    "link_id": link_id,  # Log the new link_id
                    "strength": strength,
                    "link_description": description or "",
                },
            )
        else:
            # This case should ideally not be reached if source memory validation passed
            logger.warning(
                f"Workflow ID not found for source memory {source_memory_id} during link creation logging."
            )

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after creating memory link '{link_id}': {te_wal}. Main operation succeeded."
        )

    result = {
        "link_id": link_id,
        "source_memory_id": source_memory_id,
        "target_memory_id": target_memory_id,
        "link_type": link_type_enum.value,
        "strength": strength,
        "description": description or "",
        "created_at": to_iso_z(now_unix),  # Use consistent ISO Z format
        "success": True,
        "processing_time": time.time() - start_time,
    }
    logger.info(
        f"Created/Replaced link {link_id} from {_fmt_id(source_memory_id)} to {_fmt_id(target_memory_id)} (Type: {link_type_enum.value})",
        emoji_key="link",
        time=result["processing_time"],
    )
    return result


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
    include_links: bool = False,
    link_direction: str = "outgoing",
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves UMS memories based on structured filters (level, type, tags, importance, etc.) and optional keyword search.
    Args: workflow_id, filters (level, type, search_text, tags, etc.), sort_by, sort_order, include_content, include_links, limit, offset.
    Returns: List of matching memory objects and total count."""
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
    sort_order_upper = sort_order.upper()  # Use validated upper case

    valid_link_directions = ["outgoing", "incoming", "both"]
    link_direction_lower = link_direction.lower()
    if link_direction_lower not in valid_link_directions:
        raise ToolInputError(
            f"link_direction must be one of: {', '.join(valid_link_directions)}",
            param_name="link_direction",
        )

    if limit < 1:
        raise ToolInputError("Limit must be >= 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("Offset must be >= 0", param_name="offset")

    if memory_level:
        try:
            MemoryLevel(memory_level.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid memory_level: {memory_level}", param_name="memory_level"
            ) from e
    if memory_type:
        try:
            MemoryType(memory_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid memory_type: {memory_type}", param_name="memory_type"
            ) from e

    memories_results: List[Dict[str, Any]] = []
    total_matching_count = 0

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
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
        select_clause_parts = [
            "m.memory_id",
            "m.workflow_id",
            "m.memory_level",
            "m.memory_type",
            "m.importance",
            "m.confidence",
            "m.description",
            "m.reasoning",
            "m.source",
            "m.tags",
            "m.created_at",
            "m.updated_at",
            "m.last_accessed",
            "m.access_count",
            "m.ttl",
            "m.action_id",
            "m.thought_id",
            "m.artifact_id",
        ]
        if include_content:
            select_clause_parts.append("m.content")
        select_clause = ", ".join(select_clause_parts)

        count_query_base = "SELECT COUNT(m.memory_id) FROM memories m"
        data_query_base = f"SELECT {select_clause} FROM memories m"

        joins = ""
        where_clauses = ["1=1"]
        params: List[Any] = []

        # FTS search params are handled separately if search_text is present
        fts_params: List[Any] = []

        # Apply Filters
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
        # ... (add all other filters similarly) ...
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

        current_time_unix = int(time.time())
        where_clauses.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
        params.append(current_time_unix)

        if tags and isinstance(tags, list) and len(tags) > 0:
            clean_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
            if clean_tags:
                tags_json = json.dumps(clean_tags)
                where_clauses.append("json_contains_all(m.tags, ?)")
                params.append(tags_json)

        if search_text:
            if "memory_fts" not in joins:  # Add join only if FTS search is active
                joins += " JOIN memory_fts fts ON m.rowid = fts.rowid"
            # Sanitize for FTS MATCH, ensure terms are present
            sanitized_query_for_fts = re.sub(
                r'[^a-zA-Z0-9\s*+\-"]', "", search_text
            ).strip()  # Allow FTS operators
            fts_terms = [
                term for term in sanitized_query_for_fts.split() if term
            ]  # Basic split, FTS handles operators
            if fts_terms:
                # Use the sanitized query directly for FTS MATCH
                # The FTS engine will interpret operators like OR, AND, NEAR, prefixes, phrases.
                where_clauses.append("fts.memory_fts MATCH ?")
                fts_params.append(sanitized_query_for_fts)  # Use the full sanitized string
            else:
                logger.debug(
                    f"FTS Search: Query '{search_text}' resulted in no valid terms after sanitization. FTS match clause omitted."
                )
                # If no valid FTS terms, the MATCH clause is not added, effectively disabling FTS for this query.

        where_sql = " WHERE " + " AND ".join(where_clauses) if len(where_clauses) > 1 else ""

        final_count_query = count_query_base + joins + where_sql
        final_data_query_base = data_query_base + joins + where_sql

        # --- Get Total Count ---
        async with conn.execute(final_count_query, params + fts_params) as cursor:
            row = await cursor.fetchone()
            total_matching_count = row[0] if row else 0

        # --- Apply Sorting ---
        order_clause = ""
        if sort_by == "relevance":
            order_clause = " ORDER BY compute_memory_relevance(m.importance, m.confidence, m.created_at, m.access_count, m.last_accessed)"
        elif sort_by in valid_sort_fields:
            order_clause = f" ORDER BY m.{MemoryUtils._validate_sql_identifier(sort_by, 'sort_by_column')}"  # Validate identifier
        else:  # Should not happen due to initial validation
            order_clause = " ORDER BY m.created_at"
        order_clause += f" {sort_order_upper}"

        # --- Apply Pagination ---
        limit_clause = " LIMIT ? OFFSET ?"

        # --- Execute Data Query ---
        final_data_query_paginated = final_data_query_base + order_clause + limit_clause
        final_query_params_paginated = params + fts_params + [limit, offset]

        fetched_rows_data: List[Dict[str, Any]] = []
        async with conn.execute(final_data_query_paginated, final_query_params_paginated) as cursor:
            fetched_rows = await cursor.fetchall()
            for row_data in fetched_rows:
                memory_dict = dict(row_data)
                memory_dict["tags"] = await MemoryUtils.deserialize(memory_dict.get("tags"))

                # Format timestamps for output LATER, keep as unix for internal processing
                # (e.g., for _log_memory_operation if it expects unix)

                fetched_rows_data.append(memory_dict)

        # --- Batch Update Access Stats & Log Operations (within transaction) ---
        if fetched_rows_data:
            batch_update_access_params: List[Tuple[int, str]] = []
            batch_log_op_params: List[Tuple[str, str, str, Optional[str], Dict[str, Any]]] = []
            now_unix_for_access = int(time.time())

            for mem_dict_for_access_update in fetched_rows_data:
                mem_id_for_update = mem_dict_for_access_update["memory_id"]
                wf_id_for_log = mem_dict_for_access_update.get(
                    "workflow_id", "unknown_workflow"
                )  # Fallback

                batch_update_access_params.append((now_unix_for_access, mem_id_for_update))
                batch_log_op_params.append(
                    (
                        wf_id_for_log,
                        "query_access",
                        mem_id_for_update,
                        None,
                        {"query_filters": {"sort": sort_by, "limit": limit}},
                    )
                )

            # Batch update memory access
            update_access_sql = """
                UPDATE memories SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1
                WHERE memory_id = ?
            """
            await conn.executemany(update_access_sql, batch_update_access_params)

            # Batch log operations (assuming MemoryUtils._log_memory_operation is efficient or adapted for batching)
            # If _log_memory_operation cannot be batched easily, this loop is acceptable for moderate `limit` values.
            for log_params_tuple in batch_log_op_params:
                await MemoryUtils._log_memory_operation(conn, *log_params_tuple)

        memories_results = fetched_rows_data  # Assign processed rows
    # Transaction commits here

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after memory query: {te_wal}. Main operation succeeded."
        )

    # --- Final Formatting for Output (Timestamps) ---
    for mem_dict_final_format in memories_results:
        for ts_key in ["created_at", "updated_at", "last_accessed"]:
            if ts_key in mem_dict_final_format and mem_dict_final_format[ts_key] is not None:
                mem_dict_final_format[ts_key] = safe_format_timestamp(mem_dict_final_format[ts_key])

        # Fetch and format links if requested (AFTER main transaction for memories)
        if include_links and mem_dict_final_format.get("memory_id"):
            # This part happens outside the main transaction, as it's a read operation
            # and get_linked_memories will handle its own (potential) access update transaction.
            # For simplicity and to avoid overly complex transaction nesting in this example,
            # we call it separately. If strict atomicity for link fetching within this call
            # is paramount, get_linked_memories would need to accept `conn`.
            try:
                link_details = await get_linked_memories(
                    memory_id=mem_dict_final_format["memory_id"],
                    direction=link_direction_lower,
                    limit=5,  # Default reasonable limit for linked memories in a list view
                    include_memory_details=False,  # Keep it light for list view
                    db_path=db_path,
                )
                mem_dict_final_format["links"] = link_details.get(
                    "links", {"outgoing": [], "incoming": []}
                )
            except Exception as link_err:
                logger.warning(
                    f"Failed to fetch links for memory {mem_dict_final_format['memory_id']} in query_memories: {link_err}"
                )
                mem_dict_final_format["links"] = {
                    "outgoing": [],
                    "incoming": [],
                    "error": str(link_err),
                }

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
    """Gets a list of existing UMS workflows, with optional status, tag, and date filters.
    Args: status, tag, after_date, before_date, limit, offset.
    Returns: List of workflow summary objects and total count."""
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
                    "Invalid after_date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).",
                    param_name="after_date",
                ) from e
        before_ts: Optional[int] = None
        if before_date:
            try:
                dt_obj = datetime.fromisoformat(before_date.replace("Z", "+00:00"))
                before_ts = int(dt_obj.timestamp())
            except ValueError as e:
                raise ToolInputError(
                    "Invalid before_date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).",
                    param_name="before_date",
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
    """Retrieves full details of a specific UMS workflow, including its actions, artifacts, and thoughts.
    Args: workflow_id, include_actions, include_artifacts, include_thoughts, include_memories, memories_limit.
    Returns: Comprehensive workflow object."""
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

            workflow_details = dict(wf_row)  # Keep raw timestamps

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
                    action = dict(row)  # Keep raw timestamps
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
                    artifact = dict(row)  # Keep raw timestamp
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
                    thought_chain = dict(chain_row_data)  # Keep raw timestamp
                    thought_chain["thoughts"] = []
                    thought_cursor = await conn.execute(
                        "SELECT * FROM thoughts WHERE thought_chain_id = ? ORDER BY sequence_number ASC",
                        (thought_chain["thought_chain_id"],),
                    )
                    async for thought_row_data in thought_cursor:
                        thought = dict(thought_row_data)  # Keep raw timestamp
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
                    mem = dict(row)  # Keep raw timestamp
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

            timestamp_keys_to_convert = {
                "workflow": ["created_at", "updated_at", "completed_at", "last_active"],
                "action": ["started_at", "completed_at"],
                "artifact": ["created_at"],
                "thought_chain": ["created_at"],
                "thought": ["created_at"],
            }

            # Apply conversion safely using the helper to the main workflow details
            for key in timestamp_keys_to_convert["workflow"]:
                if key in workflow_details:
                    workflow_details[key] = safe_format_timestamp(workflow_details.get(key))

            # Apply to nested actions
            for action in workflow_details.get("actions", []):
                for key in timestamp_keys_to_convert["action"]:
                    if key in action:
                        action[key] = safe_format_timestamp(action.get(key))

            # Apply to nested artifacts
            for artifact in workflow_details.get("artifacts", []):
                for key in timestamp_keys_to_convert["artifact"]:
                    if key in artifact:
                        artifact[key] = safe_format_timestamp(artifact.get(key))

            # Apply to nested thought chains and thoughts
            for chain in workflow_details.get("thought_chains", []):
                for key in timestamp_keys_to_convert["thought_chain"]:
                    if key in chain:
                        chain[key] = safe_format_timestamp(chain.get(key))
                for thought in chain.get("thoughts", []):
                    for key in timestamp_keys_to_convert["thought"]:
                        if key in thought:
                            thought[key] = safe_format_timestamp(thought.get(key))

            workflow_details["success"] = True
            logger.info(f"Retrieved details for workflow {workflow_id}", emoji_key="books")
            return workflow_details  # Return the formatted details
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
    """Fetches a list of the most recent actions for a UMS workflow, with filters.
    Args: workflow_id, limit, action_type, status, include_tool_results, include_reasoning.
    Returns: List of action objects."""
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

            for action in actions_list:
                if action.get("started_at"):
                    action["started_at"] = safe_format_timestamp(action["started_at"])
                if action.get("completed_at"):
                    action["completed_at"] = safe_format_timestamp(action["completed_at"])

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
    """Lists artifacts associated with a UMS workflow, with optional type, tag, and output filters.
    Args: workflow_id, artifact_type, tag, is_output, include_content, limit.
    Returns: List of artifact objects."""
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

            for artifact in artifacts_list:
                if artifact.get("created_at"):
                    artifact["created_at"] = safe_format_timestamp(artifact["created_at"])

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
    """Retrieves a specific UMS artifact by its unique ID.
    Args: artifact_id, include_content.
    Returns: Artifact object."""
    if not artifact_id:
        raise ToolInputError("Artifact ID required.", param_name="artifact_id")

    start_time = time.time()
    artifact_data_for_return: Optional[Dict[str, Any]] = None

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Use transaction for read + potential writes
        select_cols = "a.*, GROUP_CONCAT(t.name) as tags_str"
        # No need to include_content in SELECT here, it's part of a.*
        # The include_content flag will be used to conditionally delete from dict later.

        query = f"""
            SELECT {select_cols}
            FROM artifacts a
            LEFT JOIN artifact_tags att ON a.artifact_id = att.artifact_id
            LEFT JOIN tags t ON att.tag_id = t.tag_id
            WHERE a.artifact_id = ?
            GROUP BY a.artifact_id
            """
        async with conn.execute(query, (artifact_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise ToolInputError(f"Artifact {artifact_id} not found.", param_name="artifact_id")

            artifact = dict(row)  # Keep raw timestamp for now
            artifact["metadata"] = await MemoryUtils.deserialize(artifact.get("metadata"))
            artifact["is_output"] = bool(artifact["is_output"])
            artifact["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
            artifact.pop("tags_str", None)  # Remove intermediate column

            # Store the artifact data before potentially modifying it
            artifact_data_for_return = artifact.copy()

            if not include_content:
                if "content" in artifact_data_for_return:
                    del artifact_data_for_return["content"]

            # Update access stats for related memory if possible (within the transaction)
            mem_cursor = await conn.execute(
                "SELECT memory_id, workflow_id FROM memories WHERE artifact_id = ?", (artifact_id,)
            )
            mem_row = await mem_cursor.fetchone()
            await mem_cursor.close()  # Close cursor explicitly

            if mem_row:
                artifact_workflow_id = artifact.get(
                    "workflow_id"
                )  # Get workflow_id from artifact data
                if artifact_workflow_id:
                    # Ensure MemoryUtils helpers use the passed 'conn'
                    await MemoryUtils._update_memory_access(conn, mem_row["memory_id"])
                    await MemoryUtils._log_memory_operation(
                        conn,
                        artifact_workflow_id,
                        "access_via_artifact",
                        mem_row["memory_id"],
                        None,  # action_id
                        {"artifact_id": artifact_id},
                    )
                else:
                    logger.warning(
                        f"Cannot log memory access via artifact {artifact_id} as workflow_id is missing from artifact record."
                    )
            # Transaction will commit here

    # WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after retrieving artifact {artifact_id}: {te_wal}. Main operation succeeded."
        )

    if artifact_data_for_return is None:  # Should not happen if artifact was found
        raise ToolError("Internal error: Artifact data not available after transaction.")

    # Format timestamp AFTER transaction and before returning
    if "created_at" in artifact_data_for_return:
        artifact_data_for_return["created_at"] = safe_format_timestamp(
            artifact_data_for_return.get("created_at")
        )

    artifact_data_for_return["success"] = True
    artifact_data_for_return["processing_time"] = time.time() - start_time  # Add processing time

    logger.info(f"Retrieved artifact {artifact_id}", emoji_key="page_facing_up")
    return artifact_data_for_return


# --- 10.5 Goals ---


@with_tool_metrics
@with_error_handling
async def create_goal(
    workflow_id: str,
    description: str,
    parent_goal_id: Optional[str] = None,
    title: Optional[str] = None,
    priority: int = 3,
    reasoning: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    initial_status: str = GoalStatus.ACTIVE.value,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Defines a new UMS goal or sub-goal within a workflow.
    Agent uses this for planning and goal decomposition.
    Args: workflow_id, description, parent_goal_id, title, priority, reasoning, acceptance_criteria, metadata, initial_status.
    Returns: Full created UMS goal object."""
    if not description:
        raise ToolInputError("Goal description is required.", param_name="description")
    try:
        status_enum_val = GoalStatus(initial_status.lower())  # Validate status
    except ValueError as e:
        valid_statuses_str = ", ".join([gs.value for gs in GoalStatus])
        raise ToolInputError(
            f"Invalid initial_status '{initial_status}'. Must be one of: {valid_statuses_str}",
            param_name="initial_status",
        ) from e

    goal_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    start_time = time.time()

    db_manager = DBConnection(db_path)  # Get DB manager instance
    created_goal_dict_for_return: Optional[Dict[str, Any]] = None

    async with db_manager.transaction() as conn:  # Use transaction context manager
        # Validate workflow_id
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                # This error is critical. If it happens, the UMS DB is inconsistent.
                logger.error(
                    f"CRITICAL UMS INCONSISTENCY: Workflow '{_fmt_id(workflow_id)}' "
                    f"not found by create_goal, though it should exist for goal: '{description[:50]}...'."
                )
                raise ToolInputError(
                    f"Workflow {workflow_id} not found. Cannot create goal.",
                    param_name="workflow_id",
                )

        # Validate parent_goal_id if provided
        if parent_goal_id:
            async with conn.execute(
                "SELECT 1 FROM goals WHERE goal_id = ? AND workflow_id = ?",
                (parent_goal_id, workflow_id),  # Ensure parent is in same workflow
            ) as cursor:
                if not await cursor.fetchone():
                    raise ToolInputError(
                        f"Parent goal {parent_goal_id} not found or not in workflow {workflow_id}.",
                        param_name="parent_goal_id",
                    )

        # Get sequence number for ordering under the same parent
        if parent_goal_id:
            sql_max_seq = "SELECT MAX(sequence_number) FROM goals WHERE parent_goal_id = ? AND workflow_id = ?"
            params_max_seq = (parent_goal_id, workflow_id)
        else:  # Root goal, sequence within workflow where parent is NULL
            sql_max_seq = "SELECT MAX(sequence_number) FROM goals WHERE workflow_id = ? AND parent_goal_id IS NULL"
            params_max_seq = (workflow_id,)

        async with conn.execute(sql_max_seq, params_max_seq) as cursor:
            row = await cursor.fetchone()
            max_seq = row[0] if row and row[0] is not None else 0
            sequence_number = max_seq + 1

        # Serialize complex fields
        acceptance_criteria_json = await MemoryUtils.serialize(acceptance_criteria or [])
        metadata_json = await MemoryUtils.serialize(metadata or {})

        # Insert the new goal record, including completed_at as NULL
        await conn.execute(
            """
            INSERT INTO goals (
                goal_id, workflow_id, parent_goal_id, title, description, status,
                priority, reasoning, acceptance_criteria, metadata,
                created_at, updated_at, sequence_number, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                goal_id,
                workflow_id,
                parent_goal_id,
                title,
                description,
                status_enum_val.value,  # Use validated enum value
                priority,
                reasoning,
                acceptance_criteria_json,
                metadata_json,
                now_unix,
                now_unix,
                sequence_number,
                # completed_at is handled by NULL in the VALUES clause
            ),
        )

        # Fetch the created goal to return it (use the same transaction conn)
        async with conn.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)) as cursor:
            created_goal_row = await cursor.fetchone()
            if not created_goal_row:
                # This should not happen if insert was successful within a transaction
                logger.error(
                    f"CRITICAL: Failed to retrieve created goal '{goal_id}' immediately after insert."
                )
                raise ToolError("Failed to retrieve created goal from database post-insert.")

        created_goal_dict_for_return = dict(created_goal_row)
        # Deserialize for return (timestamps formatted after transaction)
        created_goal_dict_for_return["acceptance_criteria"] = await MemoryUtils.deserialize(
            created_goal_dict_for_return.get("acceptance_criteria")
        )
        created_goal_dict_for_return["metadata"] = await MemoryUtils.deserialize(
            created_goal_dict_for_return.get("metadata")
        )

        # Log operation (within the transaction)
        await MemoryUtils._log_memory_operation(
            conn,  # Pass the transaction connection
            workflow_id,
            "create_goal",
            None,  # memory_id not directly relevant
            None,  # action_id not directly relevant
            {
                "goal_id": goal_id,
                "title": title or description[:50],  # Use title or truncated description for log
                "parent_goal_id": parent_goal_id,
                "status": status_enum_val.value,
            },
        )
    # Transaction commits here if no exceptions were raised

    # Perform WAL checkpoint after successful transaction
    # Using PASSIVE is less disruptive if other operations are frequent
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after creating goal {goal_id}: {te_wal}. Main operation succeeded."
        )

    if created_goal_dict_for_return is None:  # Should not be None if transaction succeeded
        raise ToolError("Internal error: Goal data not available after transaction.")

    # Format timestamps for the final return dictionary AFTER the transaction
    created_goal_dict_for_return["created_at"] = safe_format_timestamp(
        created_goal_dict_for_return.get("created_at")
    )
    created_goal_dict_for_return["updated_at"] = safe_format_timestamp(
        created_goal_dict_for_return.get("updated_at")
    )
    created_goal_dict_for_return["completed_at"] = safe_format_timestamp(
        created_goal_dict_for_return.get("completed_at")  # Will be None if not completed
    )

    processing_time = time.time() - start_time
    log_display_title = title if title else description[:50]
    logger.info(
        f"Created goal '{log_display_title}...' ({_fmt_id(goal_id)}) in workflow {_fmt_id(workflow_id)}.",
        time=processing_time,
    )
    return {
        "goal": created_goal_dict_for_return,  # Return the full goal object with formatted timestamps
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def update_goal_status(
    goal_id: str,
    status: str,
    reason: Optional[str] = None,  # Optional reason for status change
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Changes a UMS goal's status (e.g., active, completed, failed).
    Agent uses this to mark progress on its objectives.
    Args: goal_id, status, reason.
    Returns: Updated UMS goal details, its parent_goal_id, and if a root goal finished."""
    if not goal_id:
        raise ToolInputError("Goal ID is required.", param_name="goal_id")
    try:
        status_enum = GoalStatus(status.lower())
    except ValueError as e:
        valid_statuses_str = ", ".join([gs.value for gs in GoalStatus])
        raise ToolInputError(
            f"Invalid goal status '{status}'. Must be one of: {valid_statuses_str}",
            param_name="status",
        ) from e

    now_unix = int(time.time())
    start_time = time.time()

    db_manager = DBConnection(db_path)
    updated_goal_dict_for_return: Optional[Dict[str, Any]] = None
    parent_goal_id_for_return: Optional[str] = None
    is_root_finished_for_return: bool = False

    async with db_manager.transaction() as conn:
        # Fetch current goal to get workflow_id and parent_goal_id
        async with conn.execute(
            "SELECT workflow_id, parent_goal_id FROM goals WHERE goal_id = ?", (goal_id,)
        ) as cursor:
            goal_info_row = await cursor.fetchone()
            if not goal_info_row:
                logger.error(f"Goal '{goal_id}' not found in UMS during update_goal_status.")
                raise ToolInputError(f"Goal {goal_id} not found.", param_name="goal_id")

            workflow_id = goal_info_row["workflow_id"]
            parent_goal_id_for_return = goal_info_row["parent_goal_id"]

        update_fields = ["status = ?", "updated_at = ?"]
        update_params: List[Any] = [status_enum.value, now_unix]

        is_terminal_status = status_enum in [
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        ]
        if is_terminal_status:
            update_fields.append("completed_at = ?")
            update_params.append(now_unix)

        update_params.append(goal_id)  # For WHERE clause

        await conn.execute(
            f"UPDATE goals SET {', '.join(update_fields)} WHERE goal_id = ?", update_params
        )

        # Fetch the updated goal details to return
        async with conn.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)) as cursor:
            updated_goal_row = await cursor.fetchone()
            if not updated_goal_row:  # Should not happen if update was successful
                logger.error(
                    f"CRITICAL: Failed to retrieve goal '{goal_id}' after status update within transaction."
                )
                raise ToolError("Failed to retrieve updated goal details after update.")

        updated_goal_dict_for_return = dict(updated_goal_row)
        # Deserialize complex fields if any (acceptance_criteria, metadata)
        updated_goal_dict_for_return["acceptance_criteria"] = await MemoryUtils.deserialize(
            updated_goal_dict_for_return.get("acceptance_criteria")
        )
        updated_goal_dict_for_return["metadata"] = await MemoryUtils.deserialize(
            updated_goal_dict_for_return.get("metadata")
        )

        # Determine if a root goal was terminally finished
        # A root goal has no parent_goal_id
        if not parent_goal_id_for_return and is_terminal_status:
            is_root_finished_for_return = True

        # Log operation (within the transaction)
        log_data = {
            "goal_id": goal_id,
            "new_status": status_enum.value,
            "reason": reason,
            "is_terminal": is_terminal_status,
            "parent_id_returned": parent_goal_id_for_return is not None,
            "root_finished_flag_returned": is_root_finished_for_return,
        }
        await MemoryUtils._log_memory_operation(
            conn, workflow_id, "update_goal_status", None, None, log_data
        )
    # Transaction commits here if no exceptions

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after updating goal status {goal_id}: {te_wal}. Main operation succeeded."
        )

    if updated_goal_dict_for_return is None:  # Defensive check
        raise ToolError("Internal error: Updated goal data not available after transaction.")

    # Format timestamps for the final return dictionary AFTER the transaction
    updated_goal_dict_for_return["created_at"] = safe_format_timestamp(
        updated_goal_dict_for_return.get("created_at")
    )
    updated_goal_dict_for_return["updated_at"] = safe_format_timestamp(
        updated_goal_dict_for_return.get("updated_at")
    )
    updated_goal_dict_for_return["completed_at"] = safe_format_timestamp(
        updated_goal_dict_for_return.get("completed_at")  # Will be None if not completed
    )

    processing_time = time.time() - start_time
    logger.info(
        f"Updated status for goal '{_fmt_id(goal_id)}' to '{status_enum.value}'.",
        time=processing_time,
    )

    return {
        "updated_goal_details": updated_goal_dict_for_return,
        "parent_goal_id": parent_goal_id_for_return,
        "is_root_finished": is_root_finished_for_return,
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def get_goal_details(
    goal_id: str,
    # Optional: include_sub_goals: bool = False, # To fetch immediate children
    # Optional: include_parent_details: bool = False, # To fetch immediate parent
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves detailed information about a specific UMS goal by its ID.
    Agent uses this to understand current objectives.
    Args: goal_id.
    Returns: Full UMS goal object."""
    start_time = time.time()

    if not goal_id:
        raise ToolInputError("Goal ID is required.", param_name="goal_id")

    try:
        async with DBConnection(db_path) as conn:
            # Fetch the goal
            # Ensure all relevant columns from your 'goals' table are selected
            query = """
                SELECT
                    goal_id, workflow_id, parent_goal_id, description, status,
                    created_at, updated_at, completed_at,
                    priority, reasoning, acceptance_criteria, metadata
                FROM goals
                WHERE goal_id = ?
            """
            async with conn.execute(query, (goal_id,)) as cursor:
                goal_row = await cursor.fetchone()

            if not goal_row:
                raise ToolInputError(f"Goal with ID '{goal_id}' not found.", param_name="goal_id")

            goal_data = dict(goal_row)

            # Deserialize JSON fields
            try:
                goal_data["acceptance_criteria"] = (
                    json.loads(goal_data["acceptance_criteria"])
                    if goal_data.get("acceptance_criteria")
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Could not deserialize acceptance_criteria for goal {goal_id}. Raw: {goal_data.get('acceptance_criteria')}"
                )
                goal_data["acceptance_criteria"] = []  # Fallback to empty list

            try:
                goal_data["metadata"] = (
                    json.loads(goal_data["metadata"]) if goal_data.get("metadata") else {}
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Could not deserialize metadata for goal {goal_id}. Raw: {goal_data.get('metadata')}"
                )
                goal_data["metadata"] = {}  # Fallback to empty dict

            # Format timestamps
            goal_data["created_at"] = safe_format_timestamp(goal_data.get("created_at"))
            goal_data["updated_at"] = safe_format_timestamp(goal_data.get("updated_at"))
            goal_data["completed_at"] = safe_format_timestamp(
                goal_data.get("completed_at")
            )  # Will be None if NULL in DB

            processing_time = time.time() - start_time
            logger.info(f"Retrieved details for goal '{_fmt_id(goal_id)}'.", time=processing_time)

            return {
                "goal": goal_data,
                "success": True,
                "processing_time": processing_time,
            }

    except ToolInputError:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving goal details for ID '{_fmt_id(goal_id)}': {e}", exc_info=True
        )
        raise ToolError(f"Failed to retrieve goal details: {str(e)}") from e


# --- 11. Thought Details ---
@with_tool_metrics
@with_error_handling
async def create_thought_chain(
    workflow_id: str,
    title: str,
    initial_thought_content: Optional[str] = None,  # Renamed for clarity
    initial_thought_type: str = ThoughtType.GOAL.value,  # Default to goal for new chains
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Starts a new, distinct line of reasoning or sub-problem analysis in UMS.
    Args: workflow_id, title, initial_thought_content, initial_thought_type.
    Returns: thought_chain_id, initial_thought_id (if created)."""
    if not title:
        raise ToolInputError("Thought chain title required", param_name="title")
    if not workflow_id:
        raise ToolInputError(
            "Workflow ID required for creating thought chain.", param_name="workflow_id"
        )

    initial_thought_type_enum: Optional[ThoughtType] = None
    if initial_thought_content:  # Only validate type if content is provided
        try:
            initial_thought_type_enum = ThoughtType(initial_thought_type.lower())
        except ValueError as e:
            valid_types_str = ", ".join([t.value for t in ThoughtType])
            raise ToolInputError(
                f"Invalid initial_thought_type '{initial_thought_type}'. Must be one of: {valid_types_str}",
                param_name="initial_thought_type",
            ) from e

    thought_chain_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    start_time = time.time()
    created_initial_thought_id: Optional[str] = None

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # Check workflow exists
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(
                    f"Workflow {workflow_id} not found. Cannot create thought chain.",
                    param_name="workflow_id",
                )

        # Insert chain using Unix timestamp
        await conn.execute(
            "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, ?, ?)",
            (thought_chain_id, workflow_id, title, now_unix),
        )
        # Update workflow timestamp
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # Add initial thought if specified, passing the transaction connection
        if initial_thought_content and initial_thought_type_enum:
            # Call record_thought, ensuring it uses the same transaction connection 'conn'
            thought_result = await record_thought(
                workflow_id=workflow_id,
                content=initial_thought_content,
                thought_type=initial_thought_type_enum.value,
                thought_chain_id=thought_chain_id,  # Link to this new chain
                db_path=db_path,  # db_path is still needed for record_thought if it might open its own connection
                conn=conn,  # Pass the active transaction connection
            )
            if thought_result.get("success"):
                created_initial_thought_id = thought_result.get("thought_id")
            else:
                # If record_thought fails, the transaction should roll back.
                # Log the error from record_thought's result.
                error_msg = thought_result.get("error", "Failed to record initial thought.")
                logger.error(
                    f"Failed to record initial thought for new chain '{title}': {error_msg}"
                )
                # Raise an error to ensure transaction rollback if not already handled by record_thought
                raise ToolError(f"Failed to create initial thought in new chain: {error_msg}")
    # Transaction commits here if all operations (including nested record_thought) succeeded.

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after creating thought chain '{title}': {te_wal}. Main operation succeeded."
        )

    result = {
        "thought_chain_id": thought_chain_id,
        "workflow_id": workflow_id,
        "title": title,
        "created_at": to_iso_z(now_unix),
        "initial_thought_id": created_initial_thought_id,
        "success": True,
        "processing_time": time.time() - start_time,
    }
    logger.info(
        f"Created thought chain '{title}' ({_fmt_id(thought_chain_id)}) in workflow {_fmt_id(workflow_id)}."
        f"{(' Initial thought: ' + _fmt_id(created_initial_thought_id)) if created_initial_thought_id else ''}",
        emoji_key="thought_balloon",
    )
    return result


@with_tool_metrics
@with_error_handling
async def get_thought_chain(
    thought_chain_id: str, include_thoughts: bool = True, db_path: str = agent_memory_config.db_path
) -> Dict[str, Any]:
    """Retrieves a specific UMS thought chain and its thoughts.
    Args: thought_chain_id, include_thoughts.
    Returns: Thought chain object with thoughts list."""
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

            if thought_chain_details.get("created_at"):
                thought_chain_details["created_at"] = safe_format_timestamp(
                    thought_chain_details["created_at"]
                )

            for thought in thought_chain_details.get("thoughts", []):
                if thought.get("created_at"):
                    thought["created_at"] = safe_format_timestamp(thought["created_at"])

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
            await cursor.close()  # Ensure cursor is closed
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
            await cursor.close()  # Ensure cursor is closed
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
                    async for score_row in cursor:  # Use different variable name
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
                        removed_id = None  # Ensure removed_id is None if removal failed

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
                    logger.warning(
                        f"Could not fetch relevance scores to determine which memory to remove from context {context_id}. Cannot add new memory."
                    )
                    return False  # Don't add if we couldn't determine which to remove

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
            f"Error in _add_to_active_memories for context {context_id}, memory {memory_id}: {e}",
            exc_info=True,
        )
        return False


# --- 12. Working Memory Management ---


@with_tool_metrics
@with_error_handling
async def get_working_memory(
    context_id: str,
    include_content: bool = True,
    include_links: bool = True,  # Default for links is now True as per UMS spec
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves current working memory items for a UMS cognitive context.
    Args: context_id, include_content, include_links.
    Returns: Working memory details including focal_memory_id and list of memory objects."""
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    start_time = time.time()

    # Initialize structure for the result payload
    result_data: Dict[str, Any] = {
        "context_id": context_id,
        "workflow_id": None,
        "focal_memory_id": None,
        "working_memories": [],
        "success": True,  # Assume success unless an error occurs
        "processing_time": 0.0,
    }

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Start transaction
        # --- 1. Get Cognitive State & Working Memory IDs ---
        async with conn.execute(
            "SELECT * FROM cognitive_states WHERE state_id = ?", (context_id,)
        ) as cursor:
            state_row_dict = await cursor.fetchone()  # Fetch as dict-like Row
            if not state_row_dict:
                logger.warning(
                    f"Cognitive state for context {context_id} not found. Returning empty working memory."
                )
                # No transaction commit needed as no DB write occurred.
                # 'success' remains true as the tool executed, but found no data for context.
                result_data["processing_time"] = time.time() - start_time
                return result_data  # Returns pre-initialized empty structure

        workflow_id_from_state = state_row_dict.get("workflow_id")
        focal_memory_id_from_state = state_row_dict.get("focal_memory_id")
        working_memory_ids_json = state_row_dict.get("working_memory")

        # Update result_data with fetched state info
        result_data["workflow_id"] = workflow_id_from_state
        result_data["focal_memory_id"] = focal_memory_id_from_state

        working_memory_ids = await MemoryUtils.deserialize(working_memory_ids_json) or []

        working_memories_list_for_result: List[Dict[str, Any]] = []  # Store processed memories

        if (
            working_memory_ids and workflow_id_from_state
        ):  # Ensure we have IDs and a workflow_id to proceed
            # --- 2. Fetch Memory Details ---
            placeholders = ", ".join(["?"] * len(working_memory_ids))
            select_cols_list = [
                "memory_id",
                "workflow_id",
                "description",
                "memory_type",
                "memory_level",
                "importance",
                "confidence",
                "created_at",
                "tags",
                "action_id",
                "thought_id",
                "artifact_id",
                "reasoning",
                "source",
                "context",
                "updated_at",
                "last_accessed",
                "access_count",
                "ttl",
                "embedding_id",
            ]
            if include_content:
                select_cols_list.append("content")
            select_cols_str = ", ".join(select_cols_list)

            memory_map: Dict[str, Dict[str, Any]] = {}
            # Also filter by workflow_id from the cognitive state for data integrity
            query_mem_details = f"SELECT {select_cols_str} FROM memories WHERE memory_id IN ({placeholders}) AND workflow_id = ?"

            async with conn.execute(
                query_mem_details, working_memory_ids + [workflow_id_from_state]
            ) as cursor:
                rows = await cursor.fetchall()
                for row_dict in rows:  # row is already dict-like
                    mem_dict = dict(row_dict)  # Ensure it's a plain dict
                    mem_dict["tags"] = await MemoryUtils.deserialize(mem_dict.get("tags"))
                    mem_dict["context"] = await MemoryUtils.deserialize(
                        mem_dict.get("context")
                    )  # Creation context

                    # Format timestamps after fetching, before adding to map
                    for ts_key in ["created_at", "updated_at", "last_accessed"]:
                        mem_dict[ts_key] = safe_format_timestamp(mem_dict.get(ts_key))

                    memory_map[row_dict["memory_id"]] = mem_dict

            # --- 3. Fetch Links if Requested ---
            if include_links:
                links_by_source_id = defaultdict(
                    list
                )  # Maps source_memory_id to list of its outgoing links
                # Fetch outgoing links for all working memories in one go
                links_query = f"""
                    SELECT ml.source_memory_id, ml.target_memory_id, ml.link_type, ml.strength, 
                           ml.description AS link_description, ml.link_id, ml.created_at,
                           target_mem.description AS target_description, target_mem.memory_type AS target_type
                    FROM memory_links ml
                    JOIN memories target_mem ON ml.target_memory_id = target_mem.memory_id
                    WHERE ml.source_memory_id IN ({placeholders}) 
                          AND target_mem.workflow_id = ? 
                """  # Ensure linked target is also in the same workflow
                async with conn.execute(
                    links_query, working_memory_ids + [workflow_id_from_state]
                ) as link_cursor:
                    async for link_row_dict in link_cursor:  # link_row_dict is already dict-like
                        link_data = dict(link_row_dict)
                        link_data["created_at"] = safe_format_timestamp(link_data.get("created_at"))
                        links_by_source_id[link_row_dict["source_memory_id"]].append(link_data)

                # Attach links to memories
                for mem_id_key in memory_map:
                    memory_map[mem_id_key]["links"] = {
                        "outgoing": links_by_source_id.get(mem_id_key, [])
                    }
                    # Note: This simplified version only fetches outgoing links.
                    # If incoming links for each working memory are also needed, a similar query for incoming links would be required.

            # --- 4. Reconstruct List & Update Access Stats (within transaction) ---
            update_access_params_list: List[Tuple[str, str]] = []  # (workflow_id, memory_id)
            for mem_id_ordered in working_memory_ids:
                if mem_id_ordered in memory_map:
                    working_memories_list_for_result.append(memory_map[mem_id_ordered])
                    update_access_params_list.append((workflow_id_from_state, mem_id_ordered))

            # --- 5. Perform Batch Access Updates and Logging (within transaction) ---
            if update_access_params_list:
                now_unix_batch_access = int(time.time())
                # Batch update access_count and last_accessed
                batch_update_sql_access = """
                    UPDATE memories
                    SET last_accessed = ?,
                        access_count = COALESCE(access_count, 0) + 1
                    WHERE memory_id = ? AND workflow_id = ? 
                """  # Include workflow_id in WHERE for safety
                # Parameters: (timestamp, memory_id, workflow_id)
                update_params_for_executemany = [
                    (now_unix_batch_access, mem_id_access, wf_id_access)
                    for wf_id_access, mem_id_access in update_access_params_list
                ]
                await conn.executemany(batch_update_sql_access, update_params_for_executemany)

                # Batch log operations
                for wf_id_log_access, mem_id_log_access in update_access_params_list:
                    await MemoryUtils._log_memory_operation(
                        conn,  # Pass transaction connection
                        wf_id_log_access,
                        "access_working",  # Or a more specific operation type
                        mem_id_log_access,
                        None,  # action_id
                        {"context_id": context_id},
                    )
            result_data["working_memories"] = working_memories_list_for_result
        # Transaction commits here if no exceptions

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after retrieving working memories: {te_wal}. Main operation succeeded."
        )

    # --- 6. Return Result ---
    processing_time = time.time() - start_time
    result_data["processing_time"] = processing_time
    logger.info(
        f"Retrieved {len(result_data['working_memories'])} working memories for context {context_id}",
        emoji_key="brain",  # Using brain emoji as it's cognitive context related
        time=processing_time,
    )
    return result_data


@with_tool_metrics
@with_error_handling
async def focus_memory(
    memory_id: str,
    context_id: str,
    add_to_working: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Sets a specific UMS memory as the current primary focus for a cognitive context.
    Args: memory_id, context_id, add_to_working.
    Returns: Confirmation with focused_memory_id."""
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    start_time = time.time()
    added_to_wm_successfully = False  # Track if it was actually added

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # Check memory exists and get its workflow_id
        mem_workflow_id: Optional[str] = None
        async with conn.execute(
            "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
        ) as cursor:
            mem_row = await cursor.fetchone()
            if not mem_row:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
            mem_workflow_id = mem_row["workflow_id"]

        # Check context exists and belongs to the same workflow
        context_workflow_id: Optional[str] = None
        async with conn.execute(
            "SELECT workflow_id FROM cognitive_states WHERE state_id = ?", (context_id,)
        ) as cursor:
            state_row = await cursor.fetchone()
            if not state_row:
                raise ToolInputError(f"Context {context_id} not found.", param_name="context_id")
            context_workflow_id = state_row["workflow_id"]

        if context_workflow_id != mem_workflow_id:
            raise ToolInputError(
                f"Memory {_fmt_id(memory_id)} (wf={_fmt_id(mem_workflow_id)}) does not belong to context {context_id}'s workflow ({_fmt_id(context_workflow_id)})"
            )

        # Add to working memory if requested (uses helper which knows the correct column name)
        if add_to_working:
            # Ensure _add_to_active_memories uses the passed 'conn'
            added_to_wm_successfully = await _add_to_active_memories(conn, context_id, memory_id)
            if not added_to_wm_successfully:
                logger.warning(
                    f"Failed to add memory {_fmt_id(memory_id)} to working set for context {context_id}, but proceeding to set focus."
                )
                # Continue, as focus can be set even if WM add fails (though it's not ideal)

        # Update the focal memory and last_active timestamp in the cognitive state
        now_unix = int(time.time())
        await conn.execute(
            "UPDATE cognitive_states SET focal_memory_id = ?, last_active = ? WHERE state_id = ?",
            (memory_id, now_unix, context_id),
        )

        # Log focus operation (within the transaction)
        # Ensure _log_memory_operation uses the passed 'conn'
        await MemoryUtils._log_memory_operation(
            conn,
            mem_workflow_id,
            "focus",
            memory_id,
            None,
            {"context_id": context_id},  # mem_workflow_id is confirmed to be non-None
        )
        # Transaction commits here if no exceptions were raised

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after focusing memory {memory_id}: {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    logger.info(
        f"Set memory {_fmt_id(memory_id)} as focus for context {context_id}. Added to WM: {added_to_wm_successfully}",
        emoji_key="target",
    )
    return {
        "context_id": context_id,
        "focused_memory_id": memory_id,
        "workflow_id": mem_workflow_id,  # mem_workflow_id is confirmed to be non-None
        "added_to_working": added_to_wm_successfully,  # Reflect the actual outcome of the add attempt
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def optimize_working_memory(
    context_id: str,
    target_size: int = agent_memory_config.max_working_memory_size,
    strategy: str = "balanced",  # 'balanced', 'importance', 'recency', 'diversity'
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal) Calculates an optimized UMS working memory set for a cognitive context.
    Agent loop triggers this periodically, LLM should not call directly.
    Args: context_id, target_size, strategy.
    Returns: Lists of memory IDs to retain and remove."""
    # --- Input Validation ---
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if not isinstance(target_size, int) or target_size < 0:
        raise ToolInputError(
            "Target size must be a non-negative integer.", param_name="target_size"
        )
    valid_strategies = ["balanced", "importance", "recency", "diversity"]
    if strategy not in valid_strategies:
        raise ToolInputError(
            f"Strategy must be one of: {', '.join(valid_strategies)}", param_name="strategy"
        )
    start_time = time.time()

    workflow_id_for_log: Optional[str] = None
    current_memory_ids_from_state: List[str] = []
    before_count = 0

    # Phase 1: Read cognitive state (does not need to be in the same transaction as the log write,
    # but doing it with a short-lived connection is fine)
    db_manager_read = DBConnection(db_path)
    async with db_manager_read as conn_read:
        async with conn_read.execute(
            "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        ) as cursor:
            state_row = await cursor.fetchone()
            if not state_row:
                logger.warning(
                    f"Context state {context_id} not found during optimize_working_memory read. "
                    "Cannot perform optimization calculation."
                )
                # Return early as no optimization can be calculated
                return {
                    "context_id": context_id,
                    "workflow_id": None,
                    "strategy_used": strategy,
                    "target_size": target_size,
                    "before_count": 0,
                    "after_count": 0,
                    "removed_count": 0,
                    "retained_memories": [],
                    "removed_memories": [],
                    "success": False,  # Indicate failure to find context
                    "reason": "Context ID not found.",
                    "processing_time": time.time() - start_time,
                }
            workflow_id_for_log = state_row["workflow_id"]
            current_memory_ids_from_state = (
                await MemoryUtils.deserialize(state_row["working_memory"]) or []
            )
            before_count = len(current_memory_ids_from_state)

    # --- Handle Edge Case: Already Optimized (No DB write needed yet) ---
    if before_count <= target_size:
        logger.info(
            f"Working memory list in state {context_id} already at/below target size ({before_count}/{target_size}). "
            "No optimization calculation needed. Logging this check."
        )
        # Log this event even if no optimization happens
        if workflow_id_for_log:  # Only log if we have a workflow_id
            db_manager_log_no_op = DBConnection(db_path)
            async with db_manager_log_no_op.transaction() as conn_log_no_op:
                await MemoryUtils._log_memory_operation(
                    conn_log_no_op,
                    workflow_id_for_log,
                    "calculate_wm_optimization_skipped",  # Different op name
                    None,
                    None,
                    {
                        "context_id": context_id,
                        "strategy": strategy,
                        "target_size": target_size,
                        "before_count": before_count,
                        "reason": "already_optimal_size",
                    },
                )
            try:
                await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
            except ToolError as te_wal:
                logger.warning(
                    f"Passive WAL checkpoint failed after logging WM optimization skipped: {te_wal}. Main operation succeeded."
                )

        return {
            "context_id": context_id,
            "workflow_id": workflow_id_for_log,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": before_count,
            "after_count": before_count,
            "removed_count": 0,
            "retained_memories": current_memory_ids_from_state,
            "removed_memories": [],
            "success": True,  # The tool's operation (checking) succeeded.
            "processing_time": time.time() - start_time,
        }

    # --- Fetch Details for Scoring (Read phase) ---
    memories_to_consider = []
    if current_memory_ids_from_state:  # Should be true if before_count > target_size
        db_manager_details = DBConnection(db_path)
        async with db_manager_details as conn_details:
            placeholders = ", ".join(["?"] * len(current_memory_ids_from_state))
            query = f"""
            SELECT memory_id, memory_type, importance, confidence, created_at, last_accessed, access_count
            FROM memories WHERE memory_id IN ({placeholders}) AND workflow_id = ?
            """  # Also filter by workflow_id for safety
            async with conn_details.execute(
                query, current_memory_ids_from_state + [workflow_id_for_log]
            ) as mem_cursor:
                memories_to_consider = [dict(row) for row in await mem_cursor.fetchall()]

    if (
        not memories_to_consider and current_memory_ids_from_state
    ):  # If IDs were present but no details found
        logger.warning(
            f"Working memory ID list for state {context_id} was not empty ({before_count}), "
            f"but failed to fetch details for scoring (workflow_id: {workflow_id_for_log}). "
            "Optimization calculation might be inaccurate. Logging this and returning empty optimization."
        )
        if workflow_id_for_log:
            db_manager_log_fail = DBConnection(db_path)
            async with db_manager_log_fail.transaction() as conn_log_fail:
                await MemoryUtils._log_memory_operation(
                    conn_log_fail,
                    workflow_id_for_log,
                    "calculate_wm_optimization_failed_fetch",
                    None,
                    None,
                    {
                        "context_id": context_id,
                        "strategy": strategy,
                        "target_size": target_size,
                        "before_count": before_count,
                        "reason": "failed_to_fetch_memory_details",
                    },
                )
            try:
                await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
            except ToolError as te_wal:
                logger.warning(
                    f"Passive WAL checkpoint failed after logging WM optimization failed: {te_wal}. Main operation succeeded."
                )

        return {
            "context_id": context_id,
            "workflow_id": workflow_id_for_log,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": before_count,
            "after_count": 0,
            "removed_count": before_count,
            "retained_memories": [],
            "removed_memories": current_memory_ids_from_state,
            "success": True,  # Tool calculation itself "succeeded" but result is empty
            "processing_time": time.time() - start_time,
        }

    # --- Score Memories Based on Strategy (CPU-bound, no DB access here) ---
    scored_memories = []
    now_unix = int(time.time())
    for memory in memories_to_consider:
        mem_id = memory["memory_id"]
        importance_val = memory.get("importance", 5.0)
        confidence_val = memory.get("confidence", 1.0)
        created_at_val = memory.get("created_at", now_unix)
        last_accessed_val = memory.get("last_accessed")
        access_count_val = memory.get("access_count", 0)
        mem_type_val = memory.get("memory_type")

        relevance = _compute_memory_relevance(
            importance_val, confidence_val, created_at_val, access_count_val, last_accessed_val
        )
        recency = 1.0 / (1.0 + (now_unix - (last_accessed_val or created_at_val)) / 86400)

        score = 0.0
        if strategy == "balanced":
            score = relevance
        elif strategy == "importance":
            score = (
                (importance_val * 0.6)
                + (confidence_val * 0.2)
                + (relevance * 0.1)
                + (recency * 0.1)
            )
        elif strategy == "recency":
            score = (recency * 0.5) + (min(1.0, access_count_val / 5.0) * 0.2) + (relevance * 0.3)
        elif strategy == "diversity":
            score = relevance

        scored_memories.append({"id": mem_id, "score": score, "type": mem_type_val})

    # --- Select Memories to Retain (CPU-bound) ---
    retained_memory_ids: List[str] = []
    if strategy == "diversity":
        type_groups: Dict[Any, List[Dict]] = defaultdict(list)
        for mem in scored_memories:
            type_groups[mem["type"]].append(mem)
        for (
            group_list
        ) in type_groups.values():  # Changed from group to group_list to avoid conflict
            group_list.sort(key=lambda x: x["score"], reverse=True)

        group_iters = {
            mem_type: iter(group_val) for mem_type, group_val in type_groups.items()
        }  # Use group_val
        active_groups = list(group_iters.keys())
        while len(retained_memory_ids) < target_size and active_groups:
            group_type_to_select = active_groups.pop(0)
            try:
                selected_mem = next(group_iters[group_type_to_select])
                retained_memory_ids.append(selected_mem["id"])
                active_groups.append(group_type_to_select)
            except StopIteration:
                pass
    else:
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        retained_memory_ids = [m["id"] for m in scored_memories[:target_size]]

    removed_memory_ids = list(set(current_memory_ids_from_state) - set(retained_memory_ids))
    after_count = len(retained_memory_ids)
    removed_count = len(removed_memory_ids)

    # --- Log the *Outcome* of the Optimization Calculation (Write phase) ---
    if workflow_id_for_log:  # Ensure we have workflow_id before logging
        db_manager_log_op = DBConnection(db_path)
        async with db_manager_log_op.transaction() as conn_log_op:
            await MemoryUtils._log_memory_operation(
                conn_log_op,  # Pass the transaction connection
                workflow_id_for_log,
                "calculate_wm_optimization",
                None,
                None,
                {
                    "context_id": context_id,
                    "strategy": strategy,
                    "target_size": target_size,
                    "before_count": before_count,
                    "after_count": after_count,
                    "removed_count": removed_count,
                    "retained_ids_sample": retained_memory_ids[:5],
                    "removed_ids_sample": removed_memory_ids[:5],
                },
            )
        try:
            await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
        except ToolError as te_wal:
            logger.warning(
                f"Passive WAL checkpoint failed after logging WM optimization calc for '{context_id}': {te_wal}. Main operation succeeded."
            )
    else:
        logger.warning(
            f"Cannot log wm_optimization_calculation for context {context_id} as workflow_id is missing."
        )

    processing_time = time.time() - start_time
    logger.info(
        f"Calculated working memory optimization for state {context_id} using '{strategy}'. "
        f"Input: {before_count}, Retained: {after_count}, Removed: {removed_count}",
        emoji_key="brain",  # Assuming brain emoji is appropriate
        time=processing_time,
    )
    return {
        "context_id": context_id,
        "workflow_id": workflow_id_for_log,
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


# --- 13. Cognitive State Persistence ---


@with_tool_metrics
@with_error_handling
async def save_cognitive_state(
    workflow_id: str,
    title: str,
    working_memory_ids: List[str],  # Assumed to be UMS Memory IDs
    focus_area_ids: Optional[List[str]] = None,  # Assumed to be UMS Memory IDs
    context_action_ids: Optional[List[str]] = None,  # UMS Action IDs
    current_goal_thought_ids: Optional[List[str]] = None,  # UMS Thought IDs representing goals
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal) Checkpoints the agent's current cognitive state to UMS.
    Agent loop manages this, LLM should not call directly.
    Args: workflow_id, title, working_memory_ids, focus_area_ids, context_action_ids, current_goal_thought_ids.
    Returns: state_id."""
    if not title:
        raise ToolInputError("State title required.", param_name="title")
    if not workflow_id:
        raise ToolInputError("Workflow ID required for saving state.", param_name="workflow_id")

    state_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    start_time = time.time()

    # Prepare sets for efficient validation
    all_memory_ids_to_validate: Set[str] = set(working_memory_ids or [])
    if focus_area_ids:
        all_memory_ids_to_validate.update(focus_area_ids)

    all_action_ids_to_validate: Set[str] = set(context_action_ids or [])
    all_thought_ids_to_validate: Set[str] = set(current_goal_thought_ids or [])

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # --- Validation Step (within the transaction for consistency) ---
        # 1. Check workflow exists
        async with conn.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise ToolInputError(
                    f"Workflow {workflow_id} not found. Cannot save state.",
                    param_name="workflow_id",
                )

        # 2. Validate Memory IDs belong to this workflow
        if all_memory_ids_to_validate:
            placeholders = ",".join("?" * len(all_memory_ids_to_validate))
            query = f"SELECT memory_id FROM memories WHERE memory_id IN ({placeholders}) AND workflow_id = ?"
            params = list(all_memory_ids_to_validate) + [workflow_id]
            async with conn.execute(query, params) as cursor:
                found_mem_ids = {row["memory_id"] for row in await cursor.fetchall()}

            missing_mem_ids = all_memory_ids_to_validate - found_mem_ids
            if missing_mem_ids:
                # Use _fmt_id for concise logging if many IDs
                missing_ids_str = ", ".join([_fmt_id(mid) for mid in list(missing_mem_ids)[:5]])
                if len(missing_mem_ids) > 5:
                    missing_ids_str += "..."
                raise ToolInputError(
                    f"Memory IDs not found or not in workflow '{_fmt_id(workflow_id)}': {missing_ids_str}",
                    param_name="working_memory_ids/focus_area_ids",
                )

        # 3. Validate Action IDs belong to this workflow
        if all_action_ids_to_validate:
            placeholders = ",".join("?" * len(all_action_ids_to_validate))
            query = f"SELECT action_id FROM actions WHERE action_id IN ({placeholders}) AND workflow_id = ?"
            params = list(all_action_ids_to_validate) + [workflow_id]
            async with conn.execute(query, params) as cursor:
                found_action_ids = {row["action_id"] for row in await cursor.fetchall()}

            missing_action_ids = all_action_ids_to_validate - found_action_ids
            if missing_action_ids:
                missing_ids_str = ", ".join([_fmt_id(aid) for aid in list(missing_action_ids)[:5]])
                if len(missing_action_ids) > 5:
                    missing_ids_str += "..."
                raise ToolInputError(
                    f"Action IDs not found or not in workflow '{_fmt_id(workflow_id)}': {missing_ids_str}",
                    param_name="context_action_ids",
                )

        # 4. Validate Thought IDs belong to this workflow
        if all_thought_ids_to_validate:
            placeholders = ",".join("?" * len(all_thought_ids_to_validate))
            query = f"""
                SELECT t.thought_id FROM thoughts t
                JOIN thought_chains tc ON t.thought_chain_id = tc.thought_chain_id
                WHERE t.thought_id IN ({placeholders}) AND tc.workflow_id = ?
            """
            params = list(all_thought_ids_to_validate) + [workflow_id]
            async with conn.execute(query, params) as cursor:
                found_thought_ids = {row["thought_id"] for row in await cursor.fetchall()}

            missing_thought_ids = all_thought_ids_to_validate - found_thought_ids
            if missing_thought_ids:
                missing_ids_str = ", ".join([_fmt_id(tid) for tid in list(missing_thought_ids)[:5]])
                if len(missing_thought_ids) > 5:
                    missing_ids_str += "..."
                raise ToolInputError(
                    f"Thought IDs (goals) not found or not in workflow '{_fmt_id(workflow_id)}': {missing_ids_str}",
                    param_name="current_goal_thought_ids",
                )

        # --- Proceed with Saving State (all writes within the transaction) ---
        # Mark previous states as not latest for this workflow
        await conn.execute(
            "UPDATE cognitive_states SET is_latest = 0 WHERE workflow_id = ?", (workflow_id,)
        )

        # Serialize state data (using the validated lists)
        # Ensure working_memory_ids is not None before serialization
        working_mem_json = await MemoryUtils.serialize(
            working_memory_ids if working_memory_ids is not None else []
        )
        focus_json = await MemoryUtils.serialize(focus_area_ids or [])
        context_actions_json = await MemoryUtils.serialize(context_action_ids or [])
        current_goals_json = await MemoryUtils.serialize(current_goal_thought_ids or [])

        # Determine focal_memory_id based on logic (e.g., first of focus_area_ids or working_memory_ids)
        # This assumes focal_memory_id is one of the memory IDs in focus_areas or working_memory.
        # Agent/AML should ideally determine this and pass it explicitly if a specific one is needed.
        # For now, let's pick from focus_area_ids or working_memory_ids if available.
        determined_focal_memory_id: Optional[str] = None
        if focus_area_ids:
            determined_focal_memory_id = focus_area_ids[0]
        elif working_memory_ids:  # Ensure working_memory_ids is not None before accessing
            determined_focal_memory_id = working_memory_ids[0]

        # Insert new state
        await conn.execute(
            """
            INSERT INTO cognitive_states (
                state_id, workflow_id, title, working_memory, focus_areas,
                context_actions, current_goals, created_at, is_latest,
                focal_memory_id, last_active 
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                True,  # is_latest
                determined_focal_memory_id,  # focal_memory_id
                now_unix,  # last_active
            ),
        )

        # Update workflow timestamp
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # Log operation (within the transaction)
        log_data = {
            "state_id": state_id,
            "title": title,
            "working_memory_count": len(working_memory_ids or []),
            "focus_count": len(focus_area_ids or []),
            "action_context_count": len(context_action_ids or []),
            "goal_count": len(current_goal_thought_ids or []),
            "focal_memory_id_set": _fmt_id(determined_focal_memory_id)
            if determined_focal_memory_id
            else "None",
        }
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "save_state",
            None,
            None,
            log_data,  # Pass conn
        )
    # Transaction commits here if no exceptions

    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after saving cognitive state '{title}' ({_fmt_id(state_id)}) for workflow {_fmt_id(workflow_id)}: {te_wal}. Main operation succeeded."
        )

    result = {
        "state_id": state_id,
        "workflow_id": workflow_id,
        "title": title,
        "created_at": to_iso_z(now_unix),  # Format timestamp for output
        "success": True,
        "processing_time": time.time() - start_time,
    }

    logger.info(
        f"Saved cognitive state '{title}' ({_fmt_id(state_id)}) for workflow {_fmt_id(workflow_id)}",
        emoji_key="save",
    )
    return result


@with_tool_metrics
@with_error_handling
async def load_cognitive_state(
    workflow_id: str,
    state_id: Optional[str] = None,  # If None, load latest
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal) Restores a previously saved UMS cognitive state.
    Agent loop manages this, LLM should not call directly.
    Args: workflow_id, state_id (optional, loads latest if None).
    Returns: Full cognitive state object."""
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")
    start_time = time.time()

    state_dict: Optional[Dict[str, Any]] = None
    created_at_unix_for_log: Optional[int] = None  # To store the raw unix timestamp for logging

    db_manager_read = DBConnection(db_path)
    async with db_manager_read as conn_read:  # Read phase
        # Check workflow exists
        async with conn_read.execute(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ) as cursor:
            wf_exists = await cursor.fetchone()
            if not wf_exists:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        # Build query
        query = "SELECT * FROM cognitive_states WHERE workflow_id = ?"
        params: List[Any] = [workflow_id]
        if state_id:
            query += " AND state_id = ?"
            params.append(state_id)
        else:
            # Prefer is_latest = 1, then fallback to most recent by created_at.
            # SQLite treats boolean TRUE as 1.
            query += " ORDER BY is_latest DESC, created_at DESC LIMIT 1"

        async with conn_read.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if not row:
                err_msg = (
                    f"State {state_id} not found."
                    if state_id
                    else f"No states found for workflow {workflow_id}."
                )
                # Log this as a warning, as it might be a valid state (e.g., new workflow)
                logger.warning(
                    f"load_cognitive_state: {err_msg} (Workflow: {workflow_id}, State: {state_id or 'latest'})"
                )
                # Return a structure indicating no state found, but success=True as the query executed.
                return {
                    "state_id": None,
                    "workflow_id": workflow_id,
                    "title": None,
                    "working_memory_ids": [],
                    "focus_areas": [],
                    "context_action_ids": [],
                    "current_goals": [],
                    "created_at": None,
                    "success": True,  # Query succeeded, but no state found
                    "message": err_msg,
                    "processing_time": time.time() - start_time,
                }
            state_dict = dict(row)
            created_at_unix_for_log = state_dict["created_at"]  # Keep raw for logging

    if state_dict is None:  # Should not happen if row was fetched
        raise ToolError("Internal error: state_dict not populated after fetch.")

    # Log operation (Write phase - new transaction)
    db_manager_log = DBConnection(db_path)
    async with db_manager_log.transaction() as conn_log:
        await MemoryUtils._log_memory_operation(
            conn_log,  # Pass the transaction connection
            workflow_id,
            "load_state",
            None,
            None,
            {"state_id": state_dict["state_id"], "title": state_dict["title"]},
        )
    # WAL checkpoint after successful log transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after loading cognitive state for workflow {_fmt_id(workflow_id)}: {te_wal}. Main operation succeeded."
        )

    # Deserialize data and format timestamp for the final return
    # Ensure 'created_at' is correctly used, which should be the Unix timestamp from DB
    result = {
        "state_id": state_dict["state_id"],
        "workflow_id": state_dict["workflow_id"],
        "title": state_dict["title"],
        "working_memory_ids": await MemoryUtils.deserialize(state_dict.get("working_memory")) or [],
        "focus_areas": await MemoryUtils.deserialize(state_dict.get("focus_areas"))
        or [],  # For UMS this is "focus_area_ids"
        "context_action_ids": await MemoryUtils.deserialize(state_dict.get("context_actions"))
        or [],
        "current_goals": await MemoryUtils.deserialize(state_dict.get("current_goals"))
        or [],  # For UMS this is "current_goal_thought_ids"
        "created_at": safe_format_timestamp(
            created_at_unix_for_log
        ),  # Use the raw unix ts for formatting
        "focal_memory_id": state_dict.get("focal_memory_id"),  # Add focal_memory_id to the return
        "success": True,
        "processing_time": time.time() - start_time,
    }

    logger.info(
        f"Loaded cognitive state '{result['title']}' ({_fmt_id(result['state_id'])}) for workflow {_fmt_id(workflow_id)}",
        emoji_key="inbox_tray",  # Assuming inbox_tray emoji is appropriate
        time=result["processing_time"],
    )
    return result


# --- 14. Comprehensive Context Retrieval ---
async def get_workflow_context(
    workflow_id: str,
    recent_actions_limit: int = 10,  # Reduced default
    important_memories_limit: int = 5,
    key_thoughts_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Retrieves a comprehensive context summary for a workflow (called internally by get_rich_context_package tool)

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


def _calculate_focus_score_internal_ums(
    memory: Dict, recent_action_ids: List[str], now_unix: int
) -> float:
    """
    Calculates a score for prioritizing focus, based on memory attributes.
    This is an internal version for UMS, mirroring the agent's logic.
    """
    score = 0.0

    # Base relevance score (importance, confidence, recency, usage)
    # Ensure _compute_memory_relevance is accessible (e.g., imported or defined in UMS)
    relevance = _compute_memory_relevance(
        memory.get("importance", 5.0),
        memory.get("confidence", 1.0),
        memory.get("created_at", now_unix),
        memory.get("access_count", 0),
        memory.get("last_accessed", None),  # Pass None if not available
    )
    score += relevance * 0.6  # Base relevance is weighted heavily

    # Boost for being linked to recent actions
    if memory.get("action_id") and memory["action_id"] in recent_action_ids:
        score += 3.0  # Significant boost if directly related to recent work

    # Boost for certain types often indicating current context
    # Ensure MemoryType enum is accessible
    if memory.get("memory_type") in [
        MemoryType.QUESTION.value,
        MemoryType.PLAN.value,
        MemoryType.INSIGHT.value,
    ]:
        score += 1.5

    # Slight boost for higher memory levels (semantic/procedural over episodic)
    # Ensure MemoryLevel enum is accessible
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
    """(Agent Internal) Automatically determines and sets the best UMS focal memory for a cognitive context.
    Agent loop triggers this periodically, LLM should not call directly.
    Args: context_id, recent_actions_count.
    Returns: Result of focus update including new_focal_memory_id."""
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if not isinstance(recent_actions_count, int) or recent_actions_count < 0:
        raise ToolInputError(
            "Recent actions count must be a non-negative integer.",
            param_name="recent_actions_count",
        )
    start_time = time.time()

    db_manager = DBConnection(db_path)  # Get DB manager instance
    async with db_manager.transaction() as conn:  # Use transaction context manager
        # --- 1. Get Current Context & Working Memory ---
        async with conn.execute(
            "SELECT workflow_id, focal_memory_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        ) as cursor:
            state_row = await cursor.fetchone()
            if not state_row:
                # Log this specific case clearly. If context_id is invalid, no update can occur.
                logger.warning(f"Context '{context_id}' not found. Cannot auto-update focus.")
                # Return a result indicating failure or no change due to missing context
                return {
                    "context_id": context_id,
                    "workflow_id": None,
                    "previous_focal_memory_id": None,
                    "new_focal_memory_id": None,
                    "focus_changed": False,
                    "reason": "Context ID not found in UMS.",
                    "success": False,  # Indicate failure to find context
                    "processing_time": time.time() - start_time,
                }
            workflow_id = state_row["workflow_id"]
            previous_focal_id = state_row["focal_memory_id"]
            current_memory_ids_json = state_row["working_memory"]

        current_memory_ids = await MemoryUtils.deserialize(current_memory_ids_json) or []

        if not current_memory_ids:
            logger.info(
                f"Working memory for context {context_id} is empty. Cannot determine focus."
            )
            # No transaction commit needed if no DB write occurs.
            return {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "previous_focal_memory_id": previous_focal_id,
                "new_focal_memory_id": None,
                "focus_changed": previous_focal_id
                is not None,  # Changed if previous was set and now it's None (though it stays same here)
                "reason": "Working memory is empty.",
                "success": True,  # Operation itself is successful (no error), just no change.
                "processing_time": time.time() - start_time,
            }

        # --- 2. Get Details for Working Memories ---
        working_memories_details = []
        if current_memory_ids:  # Ensure list is not empty before creating placeholders
            placeholders = ", ".join(["?"] * len(current_memory_ids))
            query = f"""
                SELECT memory_id, action_id, memory_type, memory_level, importance, confidence,
                       created_at, last_accessed, access_count
                FROM memories WHERE memory_id IN ({placeholders}) AND workflow_id = ?
            """  # Also filter by workflow_id for safety
            # All reads are within the transaction, ensuring consistent view
            async with conn.execute(query, current_memory_ids + [workflow_id]) as cursor:
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
            # Use the UMS-internal scoring function
            score = _calculate_focus_score_internal_ums(memory, recent_action_ids, now_unix)
            logger.debug(
                f"Memory {_fmt_id(memory['memory_id'])} in context {_fmt_id(context_id)} focus score: {score:.2f}"
            )
            if score > highest_score:
                highest_score = score
                best_candidate_id = memory["memory_id"]

        # --- 5. Update Focus if Changed (within the transaction) ---
        focus_changed = False
        reason = "Focus unchanged or no suitable candidate."
        if best_candidate_id and best_candidate_id != previous_focal_id:
            await conn.execute(
                "UPDATE cognitive_states SET focal_memory_id = ?, last_active = ? WHERE state_id = ?",
                (best_candidate_id, now_unix, context_id),
            )
            focus_changed = True
            reason = f"Memory {_fmt_id(best_candidate_id)}... has highest score ({highest_score:.2f}) based on relevance and recent activity."
            logger.info(
                f"Auto-shifting focus for context {context_id} to memory {best_candidate_id}. Previous: {previous_focal_id}",
                emoji_key="compass",
            )
            # Log the operation (within the transaction)
            await MemoryUtils._log_memory_operation(
                conn,  # Pass the transaction connection
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
            # The transaction will commit at the end of the 'async with db_manager.transaction() as conn:' block
        elif not best_candidate_id:
            reason = "No suitable memory found in working set to focus on."
            logger.info(f"Auto-focus update for context {context_id}: No suitable candidate found.")
        else:  # best_candidate_id is same as previous_focal_id
            reason = f"Focus remains on {_fmt_id(previous_focal_id)} (score: {highest_score:.2f})."
            logger.info(
                f"Auto-focus update for context {context_id}: Focus remains on {previous_focal_id}."
            )
            # Optionally update last_active even if focus didn't change, if this tool implies activity
            await conn.execute(
                "UPDATE cognitive_states SET last_active = ? WHERE state_id = ?",
                (now_unix, context_id),
            )

    # Transaction commits automatically here if no exceptions were raised
    # WAL checkpoint can be done after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after auto-focus update for context {context_id}: {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    return {
        "context_id": context_id,
        "workflow_id": workflow_id,
        "previous_focal_memory_id": previous_focal_id,
        "new_focal_memory_id": best_candidate_id,  # Will be None if no candidate or empty working mem
        "focus_changed": focus_changed,
        "reason": reason,
        "success": True,  # The operation itself succeeded, even if focus didn't change
        "processing_time": processing_time,
    }


# --- Tool: Promote Memory Level ---
@with_tool_metrics
@with_error_handling
async def promote_memory_level(
    memory_id: str,
    target_level: Optional[str] = None,
    min_access_count_episodic: int = 5,
    min_confidence_episodic: float = 0.8,
    min_access_count_semantic: int = 10,
    min_confidence_semantic: float = 0.9,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal) Attempts to elevate a UMS memory's cognitive level.
    Args: memory_id, target_level, thresholds.
    Returns: Promotion status and new_level."""
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    target_level_enum: Optional[MemoryLevel] = None
    if target_level:
        try:
            target_level_enum = MemoryLevel(target_level.lower())
        except ValueError as e:
            valid_levels_str = ", ".join([ml.value for ml in MemoryLevel])
            raise ToolInputError(
                f"Invalid target_level. Use one of: {valid_levels_str}",
                param_name="target_level",
            ) from e

    start_time = time.time()
    db_manager = DBConnection(db_path)  # DB manager instance for transaction
    promoted = False
    new_level_enum: Optional[MemoryLevel] = None
    reason = "Criteria not met or already at highest/target level."
    current_level_val_str = "unknown"  # For logging if initial fetch fails

    # --- 1. Get Current Memory Details (Read operation, can be outside main transaction if needed) ---
    # However, for consistency and to ensure we operate on potentially fresh data if this tool
    # were part of a larger operation, it's fine to do this read within a transaction context
    # that might only commit if an update happens. Or, fetch, decide, then open transaction for write.
    # For this specific tool, fetching outside and then opening transaction for write is cleaner.

    async with (
        db_manager.transaction() as conn_read
    ):  # Use a read transaction or just direct execute
        mem_row_cursor = await conn_read.execute(  # Ensure cursor is awaited and closed
            "SELECT workflow_id, memory_level, memory_type, access_count, confidence, importance FROM memories WHERE memory_id = ?",
            (memory_id,),
        )
        mem_row = await mem_row_cursor.fetchone()
        await mem_row_cursor.close()

    if not mem_row:
        raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

    current_level = MemoryLevel(mem_row["memory_level"])
    current_level_val_str = current_level.value  # For logging
    mem_type = MemoryType(mem_row["memory_type"])
    access_count = mem_row["access_count"] or 0
    confidence = mem_row["confidence"] or 0.0
    workflow_id = mem_row["workflow_id"]

    # --- 2. Determine Target Level and Check Criteria (Logic, no DB writes) ---
    potential_next_level = None
    if current_level == MemoryLevel.EPISODIC:
        potential_next_level = MemoryLevel.SEMANTIC
    elif current_level == MemoryLevel.SEMANTIC and mem_type in [
        MemoryType.PROCEDURE,
        MemoryType.SKILL,
    ]:
        potential_next_level = MemoryLevel.PROCEDURAL

    level_to_check_for = target_level_enum or potential_next_level
    criteria_met = False
    criteria_desc = ""

    if level_to_check_for and level_to_check_for.value > current_level.value:
        if level_to_check_for == MemoryLevel.SEMANTIC and current_level == MemoryLevel.EPISODIC:
            criteria_met = (
                access_count >= min_access_count_episodic and confidence >= min_confidence_episodic
            )
            criteria_desc = (
                f"Met criteria for Semantic: access_count ({access_count}) >= {min_access_count_episodic}, "
                f"confidence ({confidence:.2f}) >= {min_confidence_episodic}"
            )
        elif level_to_check_for == MemoryLevel.PROCEDURAL and current_level == MemoryLevel.SEMANTIC:
            if mem_type in [MemoryType.PROCEDURE, MemoryType.SKILL]:
                criteria_met = (
                    access_count >= min_access_count_semantic
                    and confidence >= min_confidence_semantic
                )
                criteria_desc = (
                    f"Met criteria for Procedural: type ('{mem_type.value}'), "
                    f"access_count ({access_count}) >= {min_access_count_semantic}, "
                    f"confidence ({confidence:.2f}) >= {min_confidence_semantic}"
                )
            else:
                criteria_desc = f"Criteria not met for Procedural: memory type '{mem_type.value}' is not procedure/skill."

        if criteria_met:
            promoted = True
            new_level_enum = level_to_check_for
            reason = criteria_desc
        else:
            reason = (
                criteria_desc or f"Criteria not met for promotion to {level_to_check_for.value}."
            )
    elif level_to_check_for and level_to_check_for.value <= current_level.value:
        reason = f"Memory is already at or above the target level '{level_to_check_for.value}'."
    elif not level_to_check_for:
        reason = f"Memory is already at the highest promotable level ({current_level.value}) or not eligible (type: {mem_type.value})."

    # --- 3. Update Memory if Promoted (DB Write - Needs Transaction) ---
    db_write_occurred = False
    if promoted and new_level_enum:
        async with db_manager.transaction() as conn_write:  # Start transaction for write
            now_unix = int(time.time())
            await conn_write.execute(
                "UPDATE memories SET memory_level = ?, updated_at = ? WHERE memory_id = ?",
                (new_level_enum.value, now_unix, memory_id),
            )
            await MemoryUtils._log_memory_operation(  # Pass the transaction connection
                conn_write,
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
            db_write_occurred = True
        # Transaction commits here if no exceptions

        if db_write_occurred:
            try:
                await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
            except ToolError as te_wal:
                logger.warning(
                    f"Passive WAL checkpoint failed after promoting memory '{memory_id}': {te_wal}. Main operation succeeded."
                )
            logger.info(
                f"Promoted memory {memory_id} from {current_level.value} to {new_level_enum.value}",
                emoji_key="arrow_up",
            )
    else:
        logger.info(
            f"Memory {memory_id} (Level: {current_level_val_str}) not promoted. Reason: {reason}"
        )

    # --- 4. Return Result ---
    processing_time = time.time() - start_time
    return {
        "memory_id": memory_id,
        "promoted": promoted,
        "previous_level": current_level.value,  # Return the original level
        "new_level": new_level_enum.value if promoted and new_level_enum else None,
        "reason": reason,
        "success": True,
        "processing_time": processing_time,
    }


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
    tags: Optional[List[str]] = None,
    ttl: Optional[int] = None,
    memory_level: Optional[str] = None,
    regenerate_embedding: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Modifies fields of an existing UMS memory entry.
    Args: memory_id, content, importance, confidence, description, reasoning, tags, ttl, memory_level, regenerate_embedding.
    Returns: Confirmation with updated_at timestamp and embedding_regenerated flag."""

    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    start_time = time.time()

    # Parameter validations
    if importance is not None and not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0-10.0.", param_name="importance")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ToolInputError("Confidence must be 0.0-1.0.", param_name="confidence")

    final_tags_json: Optional[str] = None
    if tags is not None:  # If tags is an empty list, it means clear tags. If None, don't touch.
        final_tags_json = json.dumps(list(set(str(t).lower() for t in tags if str(t).strip())))

    memory_level_value: Optional[str] = None
    if memory_level:
        try:
            memory_level_value = MemoryLevel(memory_level.lower()).value
        except ValueError as e:
            valid_levels_str = ", ".join([ml.value for ml in MemoryLevel])
            raise ToolInputError(
                f"Invalid memory_level. Must be one of: {valid_levels_str}",
                param_name="memory_level",
            ) from e

    update_clauses = []
    params: List[Any] = []  # Params for the SQL query
    updated_fields_log: List[str] = []  # For logging which fields were actually changed

    # Build dynamic SET clause
    if content is not None:
        update_clauses.append("content = ?")
        params.append(content)
        updated_fields_log.append("content")
    if importance is not None:
        update_clauses.append("importance = ?")
        params.append(importance)
        updated_fields_log.append("importance")
    if confidence is not None:
        update_clauses.append("confidence = ?")
        params.append(confidence)
        updated_fields_log.append("confidence")
    if description is not None:
        update_clauses.append("description = ?")
        params.append(description)
        updated_fields_log.append("description")
    if reasoning is not None:
        update_clauses.append("reasoning = ?")
        params.append(reasoning)
        updated_fields_log.append("reasoning")
    if final_tags_json is not None:  # Distinguish from tags=None
        update_clauses.append("tags = ?")
        params.append(final_tags_json)
        updated_fields_log.append("tags")
    if ttl is not None:
        update_clauses.append("ttl = ?")
        params.append(ttl)
        updated_fields_log.append("ttl")
    if memory_level_value is not None:
        update_clauses.append("memory_level = ?")
        params.append(memory_level_value)
        updated_fields_log.append("memory_level")

    if not update_clauses and not regenerate_embedding:
        raise ToolInputError(
            "No fields provided for update and regenerate_embedding is False.", param_name="content"
        )

    now_unix_for_update = int(time.time())
    if update_clauses:  # Only add updated_at if other fields are changing via SQL
        update_clauses.append("updated_at = ?")
        params.append(now_unix_for_update)

    embedding_regenerated_flag = False
    new_embedding_id: Optional[str] = None  # To store ID from embeddings table if regenerated

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        # Check memory exists and get current state for embedding logic
        async with conn.execute(
            "SELECT workflow_id, description AS current_description, content AS current_content FROM memories WHERE memory_id = ?",
            (memory_id,),
        ) as cursor:
            mem_row = await cursor.fetchone()
            if not mem_row:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
            workflow_id = mem_row["workflow_id"]
            db_current_description = mem_row["current_description"]
            db_current_content = mem_row["current_content"]

        if update_clauses:
            # Add memory_id to params for the WHERE clause
            final_params = params + [memory_id]
            update_sql = f"UPDATE memories SET {', '.join(update_clauses)} WHERE memory_id = ?"
            await conn.execute(update_sql, final_params)
            logger.debug(
                f"Executed SQL update for memory {memory_id}. Clauses: {', '.join(update_clauses)}"
            )

        # Regenerate embedding if requested OR if content/description changed significantly
        # For now, explicit `regenerate_embedding` flag is primary trigger.
        # Content change check can be added later if desired.
        if regenerate_embedding:
            # Determine the text to embed using new values if provided, else current DB values
            text_for_embedding_desc = (
                description if "description" in updated_fields_log else db_current_description
            )
            text_for_embedding_content = (
                content if "content" in updated_fields_log else db_current_content
            )

            final_text_for_embedding = (
                f"{text_for_embedding_desc}: {text_for_embedding_content}"
                if text_for_embedding_desc
                else text_for_embedding_content
            )

            if not final_text_for_embedding:
                logger.warning(
                    f"Cannot regenerate embedding for memory {memory_id}: effective text for embedding is empty."
                )
            else:
                try:
                    # _store_embedding handles inserting/updating in 'embeddings' and updating 'memories.embedding_id'
                    new_embedding_id_from_store = await _store_embedding(
                        conn, memory_id, final_text_for_embedding
                    )
                    if new_embedding_id_from_store:
                        embedding_regenerated_flag = True
                        new_embedding_id = (
                            new_embedding_id_from_store  # Store the DB ID of the embedding
                        )
                        logger.info(
                            f"Regenerated embedding (ID: {new_embedding_id}) for updated memory {memory_id}",
                            emoji_key="brain",
                        )
                        # If only embedding was regenerated, explicitly update `updated_at` for the memory
                        if not update_clauses:
                            await conn.execute(
                                "UPDATE memories SET updated_at = ? WHERE memory_id = ?",
                                (now_unix_for_update, memory_id),
                            )
                    else:
                        logger.warning(
                            f"Embedding regeneration failed for memory {memory_id}, _store_embedding returned None."
                        )
                except Exception as embed_err:
                    logger.error(
                        f"Error during embedding regeneration for memory {memory_id}: {embed_err}",
                        exc_info=True,
                    )

        # Log operation (within the transaction)
        log_data = {"updated_fields": updated_fields_log}
        if embedding_regenerated_flag:
            log_data["embedding_regenerated"] = True
            log_data["new_embedding_id"] = new_embedding_id

        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "update_memory",
            memory_id,
            None,
            log_data,  # More specific op name
        )
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after updating memory '{memory_id}': {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    logger.info(
        f"Updated memory {memory_id}. Fields: {', '.join(updated_fields_log) or 'None (Embedding only)'}. Embedding regenerated: {embedding_regenerated_flag}",
        emoji_key="pencil2",
        time=processing_time,
    )
    return {
        "memory_id": memory_id,
        "updated_fields": updated_fields_log,
        "embedding_regenerated": embedding_regenerated_flag,
        "new_embedding_id": new_embedding_id,  # Return the ID from the embeddings table
        "updated_at": to_iso_z(now_unix_for_update),  # Use consistent ISO Z format
        "success": True,
        "processing_time": processing_time,
    }


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
    """Retrieves UMS memories linked to or from a specified memory.
    Args: memory_id, direction, link_type, limit, include_memory_details.
    Returns: Dictionary of outgoing and incoming linked memories."""
    start_time = time.time()

    if not memory_id:
        raise ToolInputError("Memory ID is required", param_name="memory_id")

    valid_directions = ["outgoing", "incoming", "both"]
    direction_lower = direction.lower()  # Store lowercase version
    if direction_lower not in valid_directions:
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

    if limit <= 0:
        raise ToolInputError("Limit must be a positive integer.", param_name="limit")

    # Initialize result structure
    result_payload: Dict[str, Any] = {  # Renamed to avoid conflict with 'result' variable name
        "memory_id": memory_id,
        "links": {"outgoing": [], "incoming": []},
        "success": True,  # Assume success unless an error occurs
        "processing_time": 0.0,
    }

    source_memory_workflow_id: Optional[str] = None  # To store workflow_id for logging

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:  # Start transaction
        # Check if memory exists and get its workflow_id for logging
        async with conn.execute(
            "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
        ) as cursor:
            source_mem_row = await cursor.fetchone()
            if not source_mem_row:
                raise ToolInputError(f"Memory {memory_id} not found", param_name="memory_id")
            source_memory_workflow_id = source_mem_row["workflow_id"]

        # Process outgoing links (memory_id is the source)
        if direction_lower in ["outgoing", "both"]:
            outgoing_query = """
                SELECT ml.*, m.memory_type AS target_type, m.description AS target_description
                FROM memory_links ml
                JOIN memories m ON ml.target_memory_id = m.memory_id
                WHERE ml.source_memory_id = ?
            """
            params_outgoing: List[Any] = [memory_id]  # Type hint for clarity

            if link_type:
                outgoing_query += " AND ml.link_type = ?"
                params_outgoing.append(link_type.lower())

            outgoing_query += " ORDER BY ml.created_at DESC LIMIT ?"
            params_outgoing.append(limit)

            async with conn.execute(outgoing_query, params_outgoing) as cursor:
                rows_outgoing = await cursor.fetchall()
                for row_out in rows_outgoing:
                    link_data_out = dict(row_out)
                    if include_memory_details:
                        target_memory_id_out = link_data_out["target_memory_id"]
                        async with conn.execute(
                            """
                            SELECT memory_id, memory_level, memory_type, importance, confidence,
                                   description, created_at, updated_at, tags
                            FROM memories WHERE memory_id = ?
                            """,
                            (target_memory_id_out,),
                        ) as mem_cursor_out:
                            target_memory_out = await mem_cursor_out.fetchone()
                            if target_memory_out:
                                mem_dict_out = dict(target_memory_out)
                                for ts_key in ["created_at", "updated_at"]:
                                    mem_dict_out[ts_key] = safe_format_timestamp(
                                        mem_dict_out.get(ts_key)
                                    )
                                mem_dict_out["tags"] = await MemoryUtils.deserialize(
                                    mem_dict_out.get("tags")
                                )
                                link_data_out["target_memory"] = mem_dict_out

                    link_data_out["created_at"] = safe_format_timestamp(
                        link_data_out.get("created_at")
                    )
                    link_data_out.pop("created_at_unix", None)
                    result_payload["links"]["outgoing"].append(link_data_out)

        # Process incoming links (memory_id is the target)
        if direction_lower in ["incoming", "both"]:
            incoming_query = """
                SELECT ml.*, m.memory_type AS source_type, m.description AS source_description
                FROM memory_links ml
                JOIN memories m ON ml.source_memory_id = m.memory_id
                WHERE ml.target_memory_id = ?
            """
            params_incoming: List[Any] = [memory_id]

            if link_type:
                incoming_query += " AND ml.link_type = ?"
                params_incoming.append(link_type.lower())

            incoming_query += " ORDER BY ml.created_at DESC LIMIT ?"
            params_incoming.append(limit)

            async with conn.execute(incoming_query, params_incoming) as cursor:
                rows_incoming = await cursor.fetchall()
                for row_in in rows_incoming:
                    link_data_in = dict(row_in)
                    if include_memory_details:
                        source_memory_id_in = link_data_in["source_memory_id"]
                        async with conn.execute(
                            """
                            SELECT memory_id, memory_level, memory_type, importance, confidence,
                                   description, created_at, updated_at, tags
                            FROM memories WHERE memory_id = ?
                            """,
                            (source_memory_id_in,),
                        ) as mem_cursor_in:
                            source_memory_in = await mem_cursor_in.fetchone()
                            if source_memory_in:
                                mem_dict_in = dict(source_memory_in)
                                for ts_key in ["created_at", "updated_at"]:
                                    mem_dict_in[ts_key] = safe_format_timestamp(
                                        mem_dict_in.get(ts_key)
                                    )
                                mem_dict_in["tags"] = await MemoryUtils.deserialize(
                                    mem_dict_in.get("tags")
                                )
                                link_data_in["source_memory"] = mem_dict_in

                    link_data_in["created_at"] = safe_format_timestamp(
                        link_data_in.get("created_at")
                    )
                    link_data_in.pop("created_at_unix", None)
                    result_payload["links"]["incoming"].append(link_data_in)

        # Record access stats for the source memory (within the transaction)
        if source_memory_workflow_id:  # Ensure workflow_id was fetched
            await MemoryUtils._update_memory_access(conn, memory_id)
            await MemoryUtils._log_memory_operation(
                conn,  # Pass transaction connection
                source_memory_workflow_id,
                "access_links",
                memory_id,
                None,
                {"direction": direction_lower, "link_type_filter": link_type},
            )
        else:
            # This case should ideally not be reached if memory_id was validated
            logger.warning(
                f"Could not log access for get_linked_memories on {memory_id} as its workflow_id was not determined."
            )

    # Transaction commits here if no exceptions
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after recording access for get_linked_memories on {memory_id}: {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    logger.info(
        f"Retrieved {len(result_payload['links']['outgoing'])} outgoing and {len(result_payload['links']['incoming'])} incoming links for memory {memory_id}",
        emoji_key="link",
    )

    result_payload["processing_time"] = processing_time
    return result_payload


# ======================================================
# Meta-Cognition Tools
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
            mem_info = memories.get(mem_id)  # Use .get() which returns None if key missing
            if mem_info:
                mem_desc_text = f"Mem({mem_id[:6]}..)"
                # Safely get description and type
                mem_desc = mem_info.get("description", "N/A")
                mem_type_info = mem_info.get("memory_type")
                mem_desc_text += (
                    f" Desc: {mem_desc[:40] if mem_desc else 'N/A'}"  # Handle None description
                )
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
    provider: Optional[str] = None,
    model: Optional[str] = None,
    store_result: bool = True,
    store_as_level: str = MemoryLevel.SEMANTIC.value,
    store_as_type: Optional[str] = None,
    max_tokens: int = 1000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Meta-cognition) Uses LLM to synthesize multiple UMS memories into a summary, insight, or procedure.
    Args: workflow_id, target_memories (IDs) or query_filter, consolidation_type, max_source_memories, store_result, store_as_level/type, provider, model.
    Returns: Consolidated content and stored_memory_id (if stored)."""
    start_time = time.time()
    valid_types = ["summary", "insight", "procedural", "question"]
    if consolidation_type not in valid_types:
        raise ToolInputError(
            f"consolidation_type must be one of: {valid_types}", param_name="consolidation_type"
        )

    source_memories_list: List[Dict[str, Any]] = []
    source_memory_ids: List[str] = []
    effective_workflow_id = (
        workflow_id  # Will be inferred if not provided and target_memories are used
    )

    db_manager_for_reads = DBConnection(db_path)  # For initial reads, no transaction needed yet

    try:
        # --- 1. Select Source Memories ---
        # This part is read-only, so it doesn't need to be in the main write transaction.
        async with db_manager_for_reads as read_conn:
            if target_memories:
                if not isinstance(target_memories, list) or len(target_memories) < 2:
                    raise ToolInputError(
                        "target_memories must be a list containing at least 2 memory IDs.",
                        param_name="target_memories",
                    )
                placeholders = ", ".join(["?"] * len(target_memories))
                query_sql = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
                params_sql = list(target_memories)
                async with read_conn.execute(query_sql, params_sql) as cursor:
                    rows = await cursor.fetchall()

                found_memories_details = {r["memory_id"]: dict(r) for r in rows}

                if not effective_workflow_id and rows:
                    effective_workflow_id = rows[0]["workflow_id"]
                    if not effective_workflow_id:
                        raise ToolError(
                            f"Memory {rows[0]['memory_id']} exists but lacks a workflow ID."
                        )
                    logger.debug(
                        f"Inferred effective_workflow_id: {effective_workflow_id} from target memories."
                    )
                elif not effective_workflow_id and not rows:
                    raise ToolInputError(
                        "Workflow ID must be provided if target_memories are specified but none are found.",
                        param_name="workflow_id",
                    )

                problematic_ids_details = []
                for mem_id_req in target_memories:
                    if mem_id_req not in found_memories_details:
                        problematic_ids_details.append(f"ID '{_fmt_id(mem_id_req)}' not found")
                    elif (
                        found_memories_details[mem_id_req].get("workflow_id")
                        != effective_workflow_id
                    ):
                        problematic_ids_details.append(
                            f"ID '{_fmt_id(mem_id_req)}' not in workflow '{_fmt_id(effective_workflow_id)}'"
                        )
                    else:
                        source_memories_list.append(found_memories_details[mem_id_req])

                if problematic_ids_details:
                    err_msg = f"Target memories issue: {'; '.join(problematic_ids_details)}."
                    raise ToolInputError(err_msg, param_name="target_memories")
                source_memory_ids = [mem["memory_id"] for mem in source_memories_list]

            elif query_filter:
                if not effective_workflow_id:
                    raise ToolInputError(
                        "workflow_id is required when using query_filter.", param_name="workflow_id"
                    )

                # Build filter query dynamically (ensure this is safe and validated)
                # This part is adapted from query_memories logic
                filter_where_clauses = ["workflow_id = ?"]
                filter_query_params: List[Any] = [effective_workflow_id]

                valid_filter_keys = {
                    "memory_level",
                    "memory_type",
                    "source",
                    "min_importance",
                    "min_confidence",
                }
                for key, value in query_filter.items():
                    if key not in valid_filter_keys:
                        logger.warning(f"Consolidation: Ignoring unsupported filter key: {key}")
                        continue
                    if key in ["memory_level", "memory_type", "source"] and value:
                        filter_where_clauses.append(f"{key} = ?")
                        filter_query_params.append(str(value).lower())
                    elif key == "min_importance" and value is not None:
                        filter_where_clauses.append("importance >= ?")
                        filter_query_params.append(float(value))
                    elif key == "min_confidence" and value is not None:
                        filter_where_clauses.append("confidence >= ?")
                        filter_query_params.append(float(value))

                now_unix_filter = int(time.time())
                filter_where_clauses.append("(ttl = 0 OR created_at + ttl > ?)")
                filter_query_params.append(now_unix_filter)

                where_sql_filter = " AND ".join(filter_where_clauses)
                query_sql = f"SELECT * FROM memories WHERE {where_sql_filter} ORDER BY importance DESC, created_at DESC LIMIT ?"
                filter_query_params.append(max_source_memories)

                async with read_conn.execute(query_sql, filter_query_params) as cursor:
                    source_memories_list = [dict(row) for row in await cursor.fetchall()]
                    source_memory_ids = [m["memory_id"] for m in source_memories_list]

            else:  # Default: Get recent, important memories from the specified workflow
                if not effective_workflow_id:
                    raise ToolInputError(
                        "workflow_id is required if not using target_memories or query_filter.",
                        param_name="workflow_id",
                    )
                query_sql = "SELECT * FROM memories WHERE workflow_id = ? AND (ttl = 0 OR created_at + ttl > ?) ORDER BY importance DESC, created_at DESC LIMIT ?"
                now_unix_filter = int(time.time())
                async with read_conn.execute(
                    query_sql, [effective_workflow_id, now_unix_filter, max_source_memories]
                ) as cursor:
                    source_memories_list = [dict(row) for row in await cursor.fetchall()]
                    source_memory_ids = [m["memory_id"] for m in source_memories_list]

            if len(source_memories_list) < 2:
                raise ToolError(
                    f"Insufficient source memories ({len(source_memories_list)}) found for consolidation."
                )
            if not effective_workflow_id:  # Should be set by now if sources were found
                raise ToolError("Could not determine a workflow ID for consolidation.")

        # --- 2. Generate Consolidation Prompt ---
        prompt = prompt_override or _generate_consolidation_prompt(
            source_memories_list, consolidation_type
        )

        # --- 3. Call LLM via Gateway ---
        consolidated_content = ""
        llm_config = get_config()  # Get full config for provider defaults
        provider_to_use = provider or llm_config.default_provider or LLMGatewayProvider.OPENAI.value
        provider_instance = await get_provider(provider_to_use)
        if not provider_instance:
            raise ToolError(f"Failed to initialize LLM provider '{provider_to_use}'.")

        model_to_use = model or provider_instance.get_default_model()
        logger.info(
            f"Consolidating memories using LLM: {provider_to_use}/{model_to_use or 'provider_default'}..."
        )

        try:
            llm_result = await provider_instance.generate_completion(
                prompt=prompt, model=model_to_use, max_tokens=max_tokens, temperature=0.6
            )
            consolidated_content = llm_result.text.strip()
            if not consolidated_content:
                logger.warning("LLM returned empty content for consolidation.")
                # Do not raise error here, return empty content and let caller decide
        except Exception as llm_err:
            logger.error(f"LLM call failed during consolidation: {llm_err}", exc_info=True)
            raise ToolError(f"Consolidation failed due to LLM error: {llm_err}") from llm_err

        # --- 4. Store Result and Log Operation (within a transaction for logging) ---
        stored_memory_id = None
        db_manager_for_writes = DBConnection(db_path)  # New manager for write transaction

        async with db_manager_for_writes.transaction() as write_conn:
            if store_result and consolidated_content:
                result_type_val = store_as_type or {
                    "summary": MemoryType.SUMMARY.value,
                    "insight": MemoryType.INSIGHT.value,
                    "procedural": MemoryType.PROCEDURE.value,
                    "question": MemoryType.QUESTION.value,
                }.get(consolidation_type, MemoryType.INSIGHT.value)
                try:
                    result_type_enum = MemoryType(result_type_val.lower())
                except ValueError:
                    result_type_enum = MemoryType.INSIGHT
                try:
                    result_level_enum = MemoryLevel(store_as_level.lower())
                except ValueError:
                    result_level_enum = MemoryLevel.SEMANTIC

                result_desc = (
                    f"Consolidated {consolidation_type} from {len(source_memory_ids)} memories."
                )
                result_tags = [
                    "consolidated",
                    consolidation_type,
                    result_type_enum.value,
                    result_level_enum.value,
                ]
                result_context = {
                    "source_memories": source_memory_ids,
                    "consolidation_type": consolidation_type,
                }

                derived_importance, derived_confidence = 5.0, 0.75
                if source_memories_list:
                    src_imps = [
                        m.get("importance", 5.0)
                        for m in source_memories_list
                        if m.get("importance") is not None
                    ]
                    src_confs = [
                        m.get("confidence", 0.5)
                        for m in source_memories_list
                        if m.get("confidence") is not None
                    ]
                    if src_imps:
                        derived_importance = min(max(src_imps) + 0.5, 10.0)
                    if src_confs:
                        derived_confidence = min(sum(src_confs) / len(src_confs), 1.0)
                        derived_confidence = max(
                            0.1,
                            derived_confidence
                            * (1.0 - min(0.2, (len(source_memories_list) - 1) * 0.02)),
                        )
                    else:
                        derived_confidence = 0.5

                logger.debug(
                    f"Consolidation: Derived Importance: {derived_importance:.2f}, Confidence: {derived_confidence:.2f}"
                )

                # Call store_memory. Assumed to be atomic itself.
                # It will use its own DBConnection and transaction if conn is not passed.
                try:
                    store_result_dict = await store_memory(
                        workflow_id=effective_workflow_id,
                        content=consolidated_content,
                        memory_type=result_type_enum.value,
                        memory_level=result_level_enum.value,
                        importance=round(derived_importance, 2),
                        confidence=round(derived_confidence, 3),
                        description=result_desc,
                        source=f"consolidation_{consolidation_type}",
                        tags=result_tags,
                        context_data=result_context,
                        generate_embedding=True,
                        suggest_links=True,
                        db_path=db_path,
                        # Not passing 'conn=write_conn' here, as store_memory isn't (yet) designed to take it
                    )
                    stored_memory_id = store_result_dict.get("memory_id")
                    if not stored_memory_id:
                        logger.error(
                            f"store_memory call during consolidation succeeded but returned no memory_id. Result: {store_result_dict}"
                        )
                        # Proceed with logging the consolidation attempt, but no links can be made.
                except Exception as store_err:
                    logger.error(
                        f"Failed to store consolidated memory result: {store_err}", exc_info=True
                    )
                    # Don't re-raise here, allow logging of the consolidation attempt itself.
                    # The fact that stored_memory_id is None will indicate storage failure.

                # Link Result to Sources (each create_memory_link is assumed atomic)
                if stored_memory_id:
                    link_tasks = []
                    for source_id in source_memory_ids:
                        link_task = create_memory_link(  # Also assumed atomic
                            source_memory_id=stored_memory_id,
                            target_memory_id=source_id,
                            link_type=LinkType.GENERALIZES.value,
                            description=f"Source for consolidated {consolidation_type}",
                            db_path=db_path,
                            # Not passing 'conn=write_conn'
                        )
                        link_tasks.append(link_task)

                    link_results = await asyncio.gather(*link_tasks, return_exceptions=True)
                    failed_links = [
                        res
                        for res in link_results
                        if isinstance(res, Exception)
                        or (isinstance(res, dict) and not res.get("success"))
                    ]
                    if failed_links:
                        logger.warning(
                            f"Failed to create {len(failed_links)} links from consolidated memory {stored_memory_id} to sources. Errors: {[str(e) for e in failed_links]}"
                        )

            # Log Consolidation Operation (this is the part atomic to *this* function)
            log_data = {
                "consolidation_type": consolidation_type,
                "source_count": len(source_memory_ids),
                "llm_provider": provider_to_use,
                "llm_model": model_to_use or "provider_default",
                "stored": bool(stored_memory_id),  # True if store_memory returned an ID
                "stored_memory_id": stored_memory_id,
                "content_generated_length": len(consolidated_content),
            }
            await MemoryUtils._log_memory_operation(
                write_conn,  # Use the transaction connection here
                effective_workflow_id,
                "consolidate",
                None,
                None,  # memory_id, action_id for the log op itself
                log_data,
            )
        # Transaction for logging commits here

        try:
            await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
        except ToolError as te_wal:
            logger.warning(
                f"Passive WAL checkpoint failed after logging consolidation for workflow {_fmt_id(effective_workflow_id)}: {te_wal}. Main operation succeeded."
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Consolidated {len(source_memory_ids)} memories ({consolidation_type}). Stored as: {stored_memory_id or 'Not Stored'}",
            emoji_key="sparkles",
            time=processing_time,
        )
        return {
            "consolidated_content": consolidated_content
            or "Consolidation failed or produced no content.",
            "consolidation_type": consolidation_type,
            "source_memory_ids": source_memory_ids,
            "workflow_id": effective_workflow_id,
            "stored_memory_id": stored_memory_id,  # Will be None if storing failed
            "success": True,  # Success of the overall consolidation *attempt*
            "processing_time": processing_time,
        }

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)
        raise ToolError(f"Unexpected error during memory consolidation: {str(e)}") from e


@with_tool_metrics
@with_error_handling
async def generate_reflection(
    workflow_id: str,
    reflection_type: str = "summary",  # summary, progress, gaps, strengths, plan
    recent_ops_limit: int = 30,
    provider: str = LLMGatewayProvider.OPENAI.value,  # Using constant from UMS if available
    model: Optional[str] = None,
    max_tokens: int = 1000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Meta-cognition) Uses LLM to analyze UMS workflow progress, gaps, or plan next steps based on recent UMS operations.
    Args: workflow_id, reflection_type, recent_ops_limit, provider, model.
    Returns: Reflection content and reflection_id."""
    start_time = time.time()
    valid_types = ["summary", "progress", "gaps", "strengths", "plan"]
    if reflection_type not in valid_types:
        raise ToolInputError(
            f"Invalid reflection_type. Must be one of: {valid_types}", param_name="reflection_type"
        )

    db_manager = DBConnection(db_path)  # For both read and write operations

    try:
        # --- 1. Fetch Workflow Info & Recent Operations (Read-only part) ---
        workflow_name: Optional[str] = None
        workflow_desc: Optional[str] = None
        operations: List[Dict[str, Any]] = []
        mem_ids_from_ops: Set[str] = set()
        memories_details_for_prompt: Dict[str, Dict[str, Any]] = {}

        async with db_manager as read_conn:  # Separate read context for these initial fetches
            async with read_conn.execute(
                "SELECT title, description FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                wf_row = await cursor.fetchone()
                if not wf_row:
                    raise ToolInputError(
                        f"Workflow {workflow_id} not found.", param_name="workflow_id"
                    )
                workflow_name = wf_row["title"]
                workflow_desc = wf_row["description"]

            async with read_conn.execute(
                "SELECT * FROM memory_operations WHERE workflow_id = ? ORDER BY timestamp DESC LIMIT ?",
                (workflow_id, recent_ops_limit),
            ) as cursor:
                operations = [dict(row) for row in await cursor.fetchall()]

            if not operations:
                raise ToolError(
                    f"No operations found for workflow {workflow_id} to generate reflection."
                )

            for op in operations:
                if op.get("memory_id"):
                    mem_ids_from_ops.add(op["memory_id"])

            if mem_ids_from_ops:
                placeholders = ",".join("?" * len(mem_ids_from_ops))
                # Fetch memory_type as well for better context in prompt
                query_mem_details = f"SELECT memory_id, description, memory_type FROM memories WHERE memory_id IN ({placeholders})"
                async with read_conn.execute(query_mem_details, list(mem_ids_from_ops)) as cursor:
                    async for row in cursor:
                        memories_details_for_prompt[row["memory_id"]] = dict(row)

        if workflow_name is None:  # Should not happen if wf_row check passed
            raise ToolError("Failed to retrieve workflow name.")

        # --- 2. Generate Reflection Prompt ---
        prompt = _generate_reflection_prompt(
            workflow_name, workflow_desc, operations, memories_details_for_prompt, reflection_type
        )

        # --- 3. Call LLM via Gateway ---
        llm_config = get_config()  # Get full config for provider defaults
        provider_to_use = provider or llm_config.default_provider or LLMGatewayProvider.OPENAI.value
        provider_instance = await get_provider(provider_to_use)
        if not provider_instance:
            raise ToolError(f"Failed to initialize LLM provider '{provider_to_use}'.")

        model_to_use = model or provider_instance.get_default_model()

        try:
            llm_result = await provider_instance.generate_completion(
                prompt=prompt, model=model_to_use, max_tokens=max_tokens, temperature=0.7
            )
            reflection_content = llm_result.text.strip()
            if not reflection_content:
                raise ToolError("LLM returned empty reflection.")
        except Exception as llm_err:
            logger.error(f"LLM call failed during reflection: {llm_err}", exc_info=True)
            raise ToolError(f"Reflection failed due to LLM error: {llm_err}") from llm_err

        # --- 4. Store Reflection and Log Operation (within a transaction) ---
        reflection_id = MemoryUtils.generate_id()
        now_unix = int(time.time())
        title = (
            reflection_content.split("\n", 1)[0].strip("# ")[:100]
            or f"{reflection_type.capitalize()} Reflection"
        )
        referenced_memory_ids_json = json.dumps(list(mem_ids_from_ops))

        async with db_manager.transaction() as write_conn:  # New transaction for writes
            await write_conn.execute(
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
                    referenced_memory_ids_json,
                ),
            )
            await MemoryUtils._log_memory_operation(
                write_conn,
                workflow_id,
                "reflect",
                None,
                None,
                {
                    "reflection_id": reflection_id,
                    "reflection_type": reflection_type,
                    "ops_analyzed": len(operations),
                    "title": title,
                },
            )
        # Transaction commits here

        try:
            await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
        except ToolError as te_wal:
            logger.warning(
                f"Passive WAL checkpoint failed after logging reflection for workflow {_fmt_id(workflow_id)}: {te_wal}. Main operation succeeded."
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Generated reflection '{title}' ({_fmt_id(reflection_id)}) for workflow {_fmt_id(workflow_id)}",
            emoji_key="mirror",
            time=processing_time,
        )
        return {
            "reflection_id": reflection_id,
            "reflection_type": reflection_type,
            "title": title,
            "content": reflection_content,
            "workflow_id": workflow_id,
            "operations_analyzed": len(operations),
            "success": True,
            "processing_time": processing_time,
        }

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(
            f"Failed to generate reflection for workflow {workflow_id}: {str(e)}", exc_info=True
        )
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
    provider: str = "openai",  # Consider making this use LLMGatewayProvider enum default
    model: Optional[str] = None,
    workflow_id: Optional[str] = None,
    record_summary: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """Generates a concise summary of a given text block using an LLM.
    Can be used for general text summarization tasks.
    Args: text_to_summarize, target_tokens, prompt_template, provider, model, workflow_id (optional for storing), record_summary.
    Returns: Summary text and stored_memory_id (if recorded)."""
    start_time = time.time()

    if not text_to_summarize:
        raise ToolInputError("Text to summarize cannot be empty", param_name="text_to_summarize")

    if record_summary and not workflow_id:
        raise ToolInputError(
            "Workflow ID is required when record_summary=True", param_name="workflow_id"
        )

    target_tokens = max(50, min(2000, target_tokens))  # Ensure reasonable target

    # Determine provider and model (this logic can stay outside transaction)
    # Assuming get_config() and get_provider() are available as in your original UMS file.
    # If `provider` arg is from LLMGatewayProvider enum, adjust type hint.
    # For now, assuming provider is string.
    loaded_config = get_config()  # Fetch config for provider/model defaults
    provider_to_use = provider or loaded_config.default_provider or LLMGatewayProvider.OPENAI.value

    # Default models for common providers if none specified
    # You might have a more sophisticated way to get default model from provider instance
    default_models_map = {
        LLMGatewayProvider.OPENAI.value: "gpt-4.1-mini",  # Example, adjust to your config
        LLMGatewayProvider.ANTHROPIC.value: "claude-3-5-haiku-20241022",  # Example
    }
    model_to_use = model or default_models_map.get(provider_to_use)

    if not model_to_use:  # Fallback if no default for provider or model not specified
        try:
            temp_provider_instance_for_default = await get_provider(provider_to_use)
            model_to_use = temp_provider_instance_for_default.get_default_model()
        except Exception:
            logger.warning(
                f"Could not determine default model for provider {provider_to_use}. LLM call might fail if model is required."
            )
            # If model is strictly required by provider_instance.generate_completion, this could be an error.

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
    summary_text = ""
    try:
        provider_instance = await get_provider(provider_to_use)
        if not provider_instance:
            raise ToolError(
                f"Failed to initialize provider '{provider_to_use}'. Check configuration."
            )

        prompt_for_llm = prompt_template.format(
            text_to_summarize=text_to_summarize, target_tokens=target_tokens
        )

        generation_result = await provider_instance.generate_completion(
            prompt=prompt_for_llm,
            model=model_to_use,
            max_tokens=target_tokens + 100,
            temperature=0.3,
        )
        summary_text = generation_result.text.strip()
        if not summary_text:
            raise ToolError("LLM returned empty summary.")

    except Exception as e:
        logger.error(f"Error generating summary via LLM: {e}", exc_info=True)
        raise ToolError(f"Failed to generate summary using LLM: {str(e)}") from e

    # Optional: Store summary as a memory (this part needs transaction)
    stored_memory_id = None
    db_write_occurred = False

    if (
        record_summary and workflow_id
    ):  # workflow_id confirmed to be present if record_summary is True
        db_manager = DBConnection(db_path)
        async with db_manager.transaction() as conn:
            # Validate workflow exists (within the transaction)
            async with conn.execute(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    # This should ideally not happen if workflow_id was validated before calling,
                    # but good to double-check within the transaction context.
                    raise ToolInputError(
                        f"Workflow {workflow_id} not found during summary recording.",
                        param_name="workflow_id",
                    )

            memory_id = MemoryUtils.generate_id()
            now_unix = int(time.time())
            description = f"Summary of text ({len(text_to_summarize)} chars)"
            # Ensure MemoryLevel and MemoryType enums are correctly used
            tags_list = [
                "summary",
                "automated",
                "text_summary",
                MemoryLevel.SEMANTIC.value,
                MemoryType.SUMMARY.value,
            ]
            tags_json = json.dumps(list(set(tags_list)))  # Ensure unique tags

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
                    "summarize_text_tool",
                    tags_json,
                    now_unix,
                    now_unix,
                    0,
                ),
            )
            stored_memory_id = memory_id  # Set after successful insert

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create_summary_memory",
                memory_id,
                None,
                {"original_length": len(text_to_summarize), "summary_length": len(summary_text)},
            )
            db_write_occurred = True
        # Transaction commits here if no exceptions

        if db_write_occurred:
            try:
                await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
            except ToolError as te_wal:
                logger.warning(
                    f"Passive WAL checkpoint failed after recording summary memory for workflow {_fmt_id(workflow_id)}: {te_wal}. Main operation succeeded."
                )

    processing_time = time.time() - start_time
    logger.info(
        f"Generated summary of {len(text_to_summarize)} chars text to {len(summary_text)} chars. Stored as: {stored_memory_id or 'Not Stored'}",
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


# --- 17. Maintenance (Adapted from cognitive_memory) ---
@with_tool_metrics
@with_error_handling
async def delete_expired_memories(db_path: str = agent_memory_config.db_path) -> Dict[str, Any]:
    """(Agent Internal) System maintenance: removes UMS memories past their TTL.
    Agent loop triggers this periodically, LLM should not call directly.
    Returns: Count of deleted memories."""
    start_time = time.time()
    deleted_count_final = 0
    workflows_affected_final: Set[str] = set()

    db_manager = DBConnection(db_path)
    async with db_manager.transaction() as conn:
        now_unix = int(time.time())

        # Find expired memory IDs and their workflows within the transaction
        # This ensures we are operating on a consistent snapshot for deletion.
        expired_memories_to_delete: List[Dict[str, Any]] = []
        async with conn.execute(
            "SELECT memory_id, workflow_id FROM memories WHERE ttl > 0 AND created_at + ttl < ?",
            (now_unix,),
        ) as cursor:
            # Fetch all rows into a list of dicts directly
            expired_memories_to_delete = [dict(row) for row in await cursor.fetchall()]

        if not expired_memories_to_delete:
            logger.info("No expired memories found to delete.")
            # No need to commit if nothing changed, but transaction manager handles it.
            # Still need to return success structure.
            return {
                "deleted_count": 0,
                "workflows_affected": [],
                "success": True,
                "processing_time": time.time() - start_time,
            }

        expired_ids = [row["memory_id"] for row in expired_memories_to_delete]
        workflows_affected_final = {row["workflow_id"] for row in expired_memories_to_delete}
        deleted_count_final = len(expired_ids)

        # Delete in batches to avoid issues with too many placeholders
        batch_size = 500  # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999
        for i in range(0, deleted_count_final, batch_size):
            batch_ids = expired_ids[i : i + batch_size]
            placeholders = ", ".join(["?"] * len(batch_ids))
            # Delete from memories table (FK constraints handle related embeddings/links)
            # This happens within the transaction.
            delete_cursor = await conn.execute(  # Capture cursor to check rowcount
                f"DELETE FROM memories WHERE memory_id IN ({placeholders})", batch_ids
            )
            logger.debug(
                f"Deleted batch of {delete_cursor.rowcount} expired memories."
            )  # Log actual deleted count for batch

        # Log the operation for each affected workflow (within the transaction)
        for wf_id in workflows_affected_final:
            # Count how many memories were deleted specifically for this workflow from the fetched list
            count_for_this_wf = sum(
                1 for mem_info in expired_memories_to_delete if mem_info["workflow_id"] == wf_id
            )
            await MemoryUtils._log_memory_operation(
                conn,  # Pass the transaction connection
                wf_id,
                "expire_batch",
                None,  # memory_id not applicable for batch log
                None,  # action_id not applicable
                {
                    "expired_count_in_workflow": count_for_this_wf,
                    "total_expired_this_run": deleted_count_final,
                },
            )
    # Transaction commits here if no exceptions

    # Perform WAL checkpoint after successful transaction
    try:
        await DBConnection.force_wal_checkpoint(modes=("PASSIVE",), log_if_unable=True)
    except ToolError as te_wal:
        logger.warning(
            f"Passive WAL checkpoint failed after deleting expired memories: {te_wal}. Main operation succeeded."
        )

    processing_time = time.time() - start_time
    logger.success(
        f"Deleted {deleted_count_final} expired memories across {len(workflows_affected_final)} workflows.",
        emoji_key="wastebasket",
        time=processing_time,
    )
    return {
        "deleted_count": deleted_count_final,
        "workflows_affected": list(workflows_affected_final),
        "success": True,
        "processing_time": processing_time,
    }


@with_tool_metrics
@with_error_handling
async def get_rich_context_package(
    workflow_id: str,
    context_id: Optional[str] = None,
    current_plan_step_description: Optional[str] = None,
    focal_memory_id_hint: Optional[str] = None,
    fetch_limits: Optional[Dict[str, int]] = None,
    show_limits: Optional[Dict[str, int]] = None,  # For UMS-side truncation/summarization decisions
    include_core_context: bool = True,
    include_working_memory: bool = True,
    include_proactive_memories: bool = True,
    include_relevant_procedures: bool = True,
    include_contextual_links: bool = True,
    # Note: include_goal_details / include_goal_stack_summary are NOT parameters here,
    # as per Scenario A, the agent handles its own goal stack for context.
    compression_token_threshold: Optional[int] = None,
    compression_target_tokens: Optional[int] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Core Internal) Retrieves a UMS context package.
    Agent loop calls this to gather information for the LLM. LLM should not call this tool directly.
    Args: workflow_id, context_id, current_plan_step_description, focal_memory_id_hint, fetch_limits, show_limits, include_*, compression_*.
    Returns: Rich context package dictionary from UMS. (AML processes this for LLM prompt)."""
    start_time = time.time()
    ums_package_retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    # This dictionary will be the main payload under "context_package"
    assembled_context_package: Dict[str, Any] = {
        "retrieval_timestamp_ums_package": ums_package_retrieval_timestamp
    }
    errors_in_package: List[str] = []
    focal_mem_id_for_links: Optional[str] = focal_memory_id_hint  # Start with hint

    fetch_limits = fetch_limits or {}
    show_limits = show_limits or {}

    # Use provided limits or UMS defaults
    lim_actions = fetch_limits.get("recent_actions", UMS_PKG_DEFAULT_FETCH_RECENT_ACTIONS)
    lim_imp_mems = fetch_limits.get("important_memories", UMS_PKG_DEFAULT_FETCH_IMPORTANT_MEMORIES)
    lim_key_thts = fetch_limits.get("key_thoughts", UMS_PKG_DEFAULT_FETCH_KEY_THOUGHTS)
    lim_proactive = fetch_limits.get("proactive_memories", UMS_PKG_DEFAULT_FETCH_PROACTIVE)
    lim_procedural = fetch_limits.get("procedural_memories", UMS_PKG_DEFAULT_FETCH_PROCEDURAL)
    lim_links = fetch_limits.get("link_traversal", UMS_PKG_DEFAULT_FETCH_LINKS)

    lim_show_links_summary = show_limits.get("link_traversal", UMS_PKG_DEFAULT_SHOW_LINKS_SUMMARY)

    # --- 0. Initial Workflow Validation (already done by agent, but good UMS practice) ---
    try:
        async with DBConnection(db_path) as conn:
            async with conn.execute(
                "SELECT title FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    # This case should ideally be caught by the agent before calling,
                    # but the UMS tool should also validate its inputs.
                    raise ToolInputError(
                        f"Target workflow_id '{workflow_id}' not found in UMS.",
                        param_name="workflow_id",
                    )
    except Exception as e:
        logger.error(
            f"UMS Package: Workflow ID validation failed for {workflow_id}: {e}", exc_info=True
        )
        # Return immediately if workflow is invalid, as other operations depend on it
        return {
            "success": False,
            "error": f"Workflow ID validation failed: {e}",
            "processing_time": time.time() - start_time,
        }

    # --- 1. Fetch Core Context (Recent Actions, Important Memories, Key Thoughts) ---
    if include_core_context:
        try:
            core_ctx_result = await get_workflow_context(  # UMS internal call
                workflow_id=workflow_id,
                recent_actions_limit=lim_actions,
                important_memories_limit=lim_imp_mems,
                key_thoughts_limit=lim_key_thts,
                db_path=db_path,
            )
            if core_ctx_result.get("success"):
                assembled_context_package["core_context"] = {
                    # Exclude UMS tool's own success/timing from the nested package
                    k: v
                    for k, v in core_ctx_result.items()
                    if k not in ["success", "processing_time"]
                }
                assembled_context_package["core_context"]["retrieved_at"] = (
                    ums_package_retrieval_timestamp
                )
            else:
                err_msg = (
                    f"UMS Package: Core context retrieval failed: {core_ctx_result.get('error')}"
                )
                errors_in_package.append(err_msg)
                logger.warning(err_msg)
        except Exception as e:
            err_msg = f"UMS Package: Exception in get_core_context: {e}"
            errors_in_package.append(err_msg)
            logger.error(f"UMS Package: Core context error for {workflow_id}: {e}", exc_info=True)

    # --- 2. Fetch Working Memory ---
    if include_working_memory and context_id:
        try:
            wm_result = await get_working_memory(  # UMS internal call
                context_id=context_id,
                include_content=False,  # Agent usually doesn't need full WM content in context
                include_links=False,  # Links are handled by contextual_links section below
                db_path=db_path,
            )
            if wm_result.get("success"):
                assembled_context_package["current_working_memory"] = {
                    k: v for k, v in wm_result.items() if k not in ["success", "processing_time"]
                }
                assembled_context_package["current_working_memory"]["retrieved_at"] = (
                    ums_package_retrieval_timestamp
                )
                if wm_result.get("focal_memory_id"):  # Update focal_mem_id if WM provided it
                    focal_mem_id_for_links = wm_result.get("focal_memory_id")
            else:
                err_msg = f"UMS Package: Working memory retrieval failed: {wm_result.get('error')}"
                errors_in_package.append(err_msg)
                logger.warning(err_msg)
        except Exception as e:
            err_msg = f"UMS Package: Exception in get_working_memory: {e}"
            errors_in_package.append(err_msg)
            logger.error(f"UMS Package: WM error for context {context_id}: {e}", exc_info=True)

    # --- 3. Proactive & Procedural Memory Retrieval ---
    # Uses current_plan_step_description passed by the agent
    search_query_source = current_plan_step_description or "current agent objectives"

    if include_proactive_memories:
        try:
            proactive_query_text = (
                f"Information relevant to current task or goal: {search_query_source}"
            )
            proactive_res = await hybrid_search_memories(  # UMS internal call
                query=proactive_query_text,
                workflow_id=workflow_id,
                limit=lim_proactive,
                include_content=False,  # Summaries/descriptions are usually enough for context
                semantic_weight=0.7,
                keyword_weight=0.3,
                db_path=db_path,
            )
            if proactive_res.get("success"):
                assembled_context_package["proactive_memories"] = {
                    "retrieved_at": ums_package_retrieval_timestamp,
                    "query_used": proactive_query_text,  # Good for debugging
                    "memories": proactive_res.get("memories", []),
                }
            else:
                errors_in_package.append(
                    f"UMS Package: Proactive memory search failed: {proactive_res.get('error')}"
                )
        except Exception as e:
            errors_in_package.append(f"UMS Package: Proactive search exception: {e}")
            logger.error(f"UMS Package: Proactive err for {workflow_id}: {e}", exc_info=True)

    if include_relevant_procedures:
        try:
            procedural_query_text = f"How to accomplish, perform, or execute: {search_query_source}"
            procedural_res = await hybrid_search_memories(  # UMS internal call
                query=procedural_query_text,
                workflow_id=workflow_id,
                limit=lim_procedural,
                memory_level=MemoryLevel.PROCEDURAL.value,  # Ensure MemoryLevel is imported/defined
                include_content=False,
                db_path=db_path,
            )
            if procedural_res.get("success"):
                assembled_context_package["relevant_procedures"] = {
                    "retrieved_at": ums_package_retrieval_timestamp,
                    "query_used": procedural_query_text,
                    "procedures": procedural_res.get("memories", []),
                }
            else:
                errors_in_package.append(
                    f"UMS Package: Procedural memory search failed: {procedural_res.get('error')}"
                )
        except Exception as e:
            errors_in_package.append(f"UMS Package: Procedural search exception: {e}")
            logger.error(f"UMS Package: Procedural err for {workflow_id}: {e}", exc_info=True)

    # --- 4. Contextual Links ---
    # Uses focal_mem_id_for_links (either passed as hint or derived from WM call)
    if include_contextual_links:
        id_for_links_to_check = focal_mem_id_for_links
        if not id_for_links_to_check:  # Fallback if no focal_mem_id_hint and WM didn't provide one
            core_mems = assembled_context_package.get("core_context", {}).get(
                "important_memories", []
            )
            if core_mems and isinstance(core_mems, list) and core_mems[0].get("memory_id"):
                id_for_links_to_check = core_mems[0]["memory_id"]
                logger.debug(
                    f"UMS Package: No focal ID for links, using first important memory: {id_for_links_to_check}"
                )

        if id_for_links_to_check:
            try:
                links_res = await get_linked_memories(  # UMS internal call
                    memory_id=id_for_links_to_check,
                    direction="both",
                    limit=lim_links,
                    include_memory_details=False,  # Keep details light
                    db_path=db_path,
                )
                if links_res.get("success"):
                    links_payload = links_res.get("links", {})
                    link_summary_for_agent = {
                        "source_memory_id": id_for_links_to_check,
                        "outgoing_count": len(links_payload.get("outgoing", [])),
                        "incoming_count": len(links_payload.get("incoming", [])),
                        "top_outgoing_links_summary": [
                            {
                                "target_memory_id": _fmt_id(link.get("target_memory_id")),
                                "link_type": link.get("link_type"),
                                "description": (link.get("description") or "")[:70] + "...",
                            }
                            for link in links_payload.get("outgoing", [])[:lim_show_links_summary]
                        ],
                        "top_incoming_links_summary": [
                            {
                                "source_memory_id": _fmt_id(link.get("source_memory_id")),
                                "link_type": link.get("link_type"),
                                "description": (link.get("description") or "")[:70] + "...",
                            }
                            for link in links_payload.get("incoming", [])[:lim_show_links_summary]
                        ],
                    }
                    assembled_context_package["contextual_links"] = {
                        "retrieved_at": ums_package_retrieval_timestamp,
                        "summary": link_summary_for_agent,
                    }
                else:
                    errors_in_package.append(
                        f"UMS Package: Link retrieval failed for {id_for_links_to_check}: {links_res.get('error')}"
                    )
            except Exception as e:
                errors_in_package.append(
                    f"UMS Package: Link retrieval exception for {id_for_links_to_check}: {e}"
                )
                logger.error(
                    f"UMS Package: Links err for {id_for_links_to_check}: {e}", exc_info=True
                )
        else:
            logger.debug("UMS Package: No suitable memory ID found to initiate link traversal.")

    # --- 5. UMS-Side Context Compression ---
    # This step uses the accurate `count_tokens` from your server utils.
    if compression_token_threshold is not None and compression_target_tokens is not None:
        try:
            current_package_json = json.dumps(assembled_context_package, default=str)
            estimated_tokens_before_compress = count_tokens(
                current_package_json
            )  # Uses your accurate counter

            if estimated_tokens_before_compress > compression_token_threshold:
                logger.info(
                    f"UMS Package: Context for {workflow_id} ({estimated_tokens_before_compress} tokens) exceeds threshold {compression_token_threshold}. UMS attempting compression."
                )

                # Identify largest text-heavy component(s) for summarization
                # More sophisticated: could rank components by token size
                component_to_summarize_key = None
                text_for_summarization = ""
                max_component_tokens = (
                    compression_target_tokens * 0.5
                )  # Don't try to summarize tiny blocks

                # Example: Prioritize summarizing raw lists of memories/actions if they are large
                candidates_for_summarization = {
                    "core_context.recent_actions": assembled_context_package.get(
                        "core_context", {}
                    ).get("recent_actions"),
                    "core_context.important_memories": assembled_context_package.get(
                        "core_context", {}
                    ).get("important_memories"),
                    "proactive_memories.memories": assembled_context_package.get(
                        "proactive_memories", {}
                    ).get("memories"),
                    "relevant_procedures.procedures": assembled_context_package.get(
                        "relevant_procedures", {}
                    ).get("procedures"),
                }

                for key_path, component_data in candidates_for_summarization.items():
                    if (
                        component_data
                        and isinstance(component_data, list)
                        and len(component_data) > 3
                    ):  # Only summarize if reasonably sized list
                        component_str = json.dumps(component_data, default=str)
                        component_tok_count = count_tokens(component_str)
                        if (
                            component_tok_count > max_component_tokens
                        ):  # If this component is largest so far AND worth summarizing
                            max_component_tokens = component_tok_count
                            component_to_summarize_key = key_path
                            text_for_summarization = component_str

                if component_to_summarize_key and text_for_summarization:
                    logger.debug(
                        f"UMS Package: Compressing component '{component_to_summarize_key}' ({max_component_tokens} estimated tokens)."
                    )

                    summary_tool_res = await summarize_text(  # UMS internal call
                        text_to_summarize=text_for_summarization,
                        target_tokens=int(
                            compression_target_tokens * 0.6
                        ),  # Target slightly less for one block
                        context_type=f"ums_package_component:{component_to_summarize_key}",
                        workflow_id=workflow_id,  # For logging within summarize_text
                        record_summary=False,  # Ephemeral
                        db_path=db_path,
                    )
                    if summary_tool_res.get("success") and summary_tool_res.get("summary"):
                        compressed_text_block = summary_tool_res["summary"]

                        # Update the context_package by replacing/annotating the summarized component
                        keys = component_to_summarize_key.split(".")
                        temp_ref = assembled_context_package
                        for _k_idx, k_val in enumerate(keys[:-1]):
                            temp_ref = temp_ref.setdefault(k_val, {})

                        # Get original retrieval time if available
                        original_retrieval_time = ums_package_retrieval_timestamp
                        if isinstance(temp_ref.get(keys[-1]), dict):
                            original_retrieval_time = temp_ref[keys[-1]].get(
                                "retrieved_at", ums_package_retrieval_timestamp
                            )

                        temp_ref[keys[-1]] = {  # Replace the component with a summary marker
                            "retrieved_at": original_retrieval_time,  # Keep original retrieval time
                            "_ums_compressed_": True,
                            "original_token_estimate": max_component_tokens,
                            "summary_preview": compressed_text_block[:150] + "...",
                        }

                        # Store the full summary in a dedicated section
                        compression_details = assembled_context_package.setdefault(
                            "ums_compression_details", {}
                        )
                        compression_details[component_to_summarize_key] = {
                            "summary_content": compressed_text_block,
                            "retrieved_at": ums_package_retrieval_timestamp,  # When summary was made
                        }
                        logger.info(
                            f"UMS Package: Successfully compressed '{component_to_summarize_key}' for context {workflow_id}."
                        )
                    else:
                        errors_in_package.append(
                            f"UMS Package: Compression of '{component_to_summarize_key}' failed: {summary_tool_res.get('error')}"
                        )
                else:
                    logger.info(
                        f"UMS Package: No single large component identified for targeted compression for {workflow_id}, or strategy not met."
                    )
            else:
                logger.debug(
                    f"UMS Package: Context ({estimated_tokens_before_compress} tokens) for {workflow_id} is within UMS compression threshold."
                )
        except Exception as e:
            errors_in_package.append(f"UMS Package: Exception during context compression: {e}")
            logger.error(
                f"UMS Package: Error during context compression for {workflow_id}: {e}",
                exc_info=True,
            )

    # --- Final Return ---
    final_response = {
        "success": not bool(errors_in_package),
        "context_package": assembled_context_package,  # This is the payload for the agent
        "errors": errors_in_package if errors_in_package else None,
        "processing_time": time.time() - start_time,
    }

    if errors_in_package:
        logger.warning(
            f"UMS: get_rich_context_package for {workflow_id} completed with {len(errors_in_package)} errors."
        )
    else:
        logger.info(f"UMS: Successfully assembled rich context package for {workflow_id}.")
    return final_response


# --- 18. Statistics ---
@with_tool_metrics
@with_error_handling
async def compute_memory_statistics(
    workflow_id: Optional[str] = None,  # Optional: If None, compute global stats
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """(Agent Internal) Gets UMS statistics (memory counts, link counts, etc.) for a workflow or globally.
    Agent loop may use this for adapting behavior, LLM should not call directly.
    Args: workflow_id (optional).
    Returns: Statistics dictionary."""
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


# --- 19. Reporting ---
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
    """Generates a diagram (Mermaid or JSON) of a specific UMS thought chain.
    Args: thought_chain_id, output_format.
    Returns: Visualization content."""
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
    if workflow.get("completed_at"):
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
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
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
                report_lines.append(
                    f"**Completed:** {safe_format_timestamp(action['completed_at'])}"
                )

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
                    except Exception:  # Catch potential errors during dump
                        args_str = str(tool_args)
                        lang = ""
                    report_lines.append(f"**Arguments:**\n```{lang}\n{args_str}\n```")

                # tool_result might already be deserialized by get_workflow_details
                tool_result = action.get("tool_result")
                if tool_result is not None:  # Check for None explicitly
                    result_repr = tool_result
                    try:
                        # Attempt to format as JSON if it's dict/list
                        if isinstance(result_repr, (dict, list)):
                            result_str = json.dumps(result_repr, indent=2)
                            lang = "json"
                        else:
                            result_str = str(result_repr)
                            lang = ""
                    except Exception:  # Catch potential errors during dump
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
            thoughts = sorted(
                chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float("inf"))
            )
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
            sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))[-1]
            if actions
            else None
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
        "\n---\n*Report generated on "
        + safe_format_timestamp(datetime.now(timezone.utc).timestamp())
        + "*"
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
        sorted_actions = sorted(
            actions, key=lambda a: a.get("sequence_number", float("inf")), reverse=True
        )
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
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
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
                report_lines.append(
                    f"Around {start_time_action}, we decided to skip the step: **{title}**."
                )
            elif action.get("status") == ActionStatus.PLANNED.value:
                report_lines.append(f"The plan included the step: **{title}** (not yet started).")
            report_lines.append("")  # Add spacing between actions

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
        sorted_thoughts = sorted(key_thoughts, key=lambda t: t.get("sequence_number", float("inf")))
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
        display_artifacts.sort(key=lambda a: a.get("created_at", 0))  # Sort combined list

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
    report_lines.append(
        "\n---\n*Narrative recorded on "
        + safe_format_timestamp(datetime.now(timezone.utc).timestamp())
        + "*"
    )
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
        sorted_actions = sorted(actions, key=lambda a: a.get("sequence_number", float("inf")))
        for action in sorted_actions:
            report_lines.append(f"### Action Sequence: {action.get('sequence_number')}\n```yaml")
            report_lines.append(f"action_id: {action.get('action_id')}")
            report_lines.append(f"title: {action.get('title')}")
            report_lines.append(f"type: {action.get('action_type')}")
            report_lines.append(f"status: {action.get('status')}")
            # Use safe_format_timestamp for action timestamps
            report_lines.append(f"started_at: {safe_format_timestamp(action.get('started_at'))}")
            if action.get("completed_at"):
                report_lines.append(
                    f"completed_at: {safe_format_timestamp(action.get('completed_at'))}"
                )
            if action.get("tool_name"):
                report_lines.append(f"tool_name: {action['tool_name']}")
            # Use the already deserialized data if present
            tool_args_repr = str(action.get("tool_args", "N/A"))
            tool_result_repr = str(action.get("tool_result", "N/A"))
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
            thoughts = sorted(
                chain.get("thoughts", []), key=lambda t: t.get("sequence_number", float("inf"))
            )
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
            return f"{prefix}_MISSING_{str(uuid.uuid4()).replace('-', '_')}"  # Use uuid directly
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
    """Generates a diagram (Mermaid) of linked UMS memories.
    Args: workflow_id or center_memory_id, depth, max_nodes, memory_level/type filters.
    Returns: Visualization content."""
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
    # Automated Cognitive Management
    "auto_update_focus",
    "promote_memory_level",
    # Meta-Cognition & Maintenance
    "consolidate_memories",
    "generate_reflection",
    "get_rich_context_package",
    "get_goal_details",
    "create_goal",
    "update_goal_status",
    "summarize_text",
    "delete_expired_memories",
    "compute_memory_statistics",
    # Reporting & Visualization
    "generate_workflow_report",
    "visualize_reasoning_chain",
    "visualize_memory_network",
]
