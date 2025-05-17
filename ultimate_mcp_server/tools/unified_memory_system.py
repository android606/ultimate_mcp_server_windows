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


# --- BEGIN UMS MONKEY PATCH FOR AIOSQLITE CONNECTION METHODS ---
# This patch adds helper methods to aiosqlite.Connection if they are missing,
# to provide execute_fetchone, execute_fetchall, and execute_fetchval functionality
# without altering widespread existing code. This addresses AttributeErrors if the
# aiosqlite version in use doesn't have these convenience methods directly.


async def _ums_patched_execute_fetchone(self, sql, parameters=None):
    """Patched version of execute_fetchone."""
    async with self.execute(sql, parameters) as cursor:
        return await cursor.fetchone()


async def _ums_patched_execute_fetchall(self, sql, parameters=None):
    """Patched version of execute_fetchall."""
    async with self.execute(sql, parameters) as cursor:
        return await cursor.fetchall()


async def _ums_patched_execute_fetchval(self, sql, parameters=None):
    """Patched version of execute_fetchval (fetches first column of first row)."""
    async with self.execute(sql, parameters) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None


try:
    # Ensure aiosqlite is imported to access its Connection class
    if not hasattr(aiosqlite.Connection, "execute_fetchone"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchone' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchone = _ums_patched_execute_fetchone

    if not hasattr(aiosqlite.Connection, "execute_fetchall"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchall' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchall = _ums_patched_execute_fetchall

    if not hasattr(aiosqlite.Connection, "execute_fetchval"):
        logger.info("UMS MONKEY-PATCH: Adding 'execute_fetchval' to aiosqlite.Connection")
        aiosqlite.Connection.execute_fetchval = _ums_patched_execute_fetchval
except ImportError:
    logger.error("UMS MONKEY-PATCH: aiosqlite module not found. Cannot apply patches.")
except AttributeError:
    logger.error(
        "UMS MONKEY-PATCH: aiosqlite.Connection not found or attribute error during patching."
    )
# --- END UMS MONKEY PATCH ---


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
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS ums_internal_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
    );""",
    """CREATE TABLE IF NOT EXISTS workflows (
        workflow_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT, goal TEXT, status TEXT NOT NULL,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
        parent_workflow_id TEXT, metadata TEXT, last_active INTEGER,
        idempotency_key TEXT UNIQUE NULL
    );""",
    """CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, parent_action_id TEXT, action_type TEXT NOT NULL,
        title TEXT, reasoning TEXT, tool_name TEXT, tool_args TEXT, tool_result TEXT, status TEXT NOT NULL,
        started_at INTEGER NOT NULL, completed_at INTEGER, sequence_number INTEGER,
        idempotency_key TEXT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (parent_action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, artifact_type TEXT NOT NULL,
        name TEXT NOT NULL, description TEXT, path TEXT, content TEXT, metadata TEXT,
        created_at INTEGER NOT NULL, is_output BOOLEAN DEFAULT FALSE,
        idempotency_key TEXT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS thought_chains (
        thought_chain_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, action_id TEXT, title TEXT NOT NULL, created_at INTEGER NOT NULL,
        idempotency_key TEXT NULL,
        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        FOREIGN KEY (action_id) REFERENCES actions(action_id) ON DELETE SET NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        id          TEXT PRIMARY KEY,
        memory_id   TEXT UNIQUE REFERENCES memories(memory_id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
        model       TEXT    NOT NULL,
        embedding   BLOB    NOT NULL,
        dimension   INTEGER NOT NULL,
        created_at  INTEGER NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS thoughts (
        thought_id      TEXT PRIMARY KEY,
        thought_chain_id TEXT NOT NULL REFERENCES thought_chains(thought_chain_id) ON DELETE CASCADE,
        parent_thought_id TEXT REFERENCES thoughts(thought_id) ON DELETE SET NULL,
        thought_type    TEXT NOT NULL,
        content         TEXT NOT NULL,
        sequence_number INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        relevant_action_id   TEXT REFERENCES actions(action_id) ON DELETE SET NULL,
        relevant_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
        relevant_memory_id   TEXT REFERENCES memories(memory_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
        idempotency_key TEXT NULL,
        UNIQUE(thought_chain_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS memories (
        memory_id   TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
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
        embedding_id TEXT REFERENCES embeddings(id) ON DELETE SET NULL,
        action_id   TEXT REFERENCES actions(action_id) ON DELETE SET NULL,
        thought_id  TEXT REFERENCES thoughts(thought_id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
        artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
        idempotency_key TEXT NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS goals (
        goal_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
        parent_goal_id TEXT REFERENCES goals(goal_id) ON DELETE SET NULL,
        title TEXT, description TEXT NOT NULL, status TEXT NOT NULL,
        priority INTEGER DEFAULT 3, reasoning TEXT, acceptance_criteria TEXT, metadata TEXT,
        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER,
        sequence_number INTEGER,
        idempotency_key TEXT NULL,
        UNIQUE(workflow_id, idempotency_key)
    );""",
    """CREATE TABLE IF NOT EXISTS memory_links (
        link_id TEXT PRIMARY KEY, source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL,
        link_type TEXT NOT NULL, strength REAL DEFAULT 1.0, description TEXT, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        FOREIGN KEY (target_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        UNIQUE(source_memory_id, target_memory_id, link_type)
    );""",
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
    """CREATE TABLE IF NOT EXISTS dependencies (
        dependency_id INTEGER PRIMARY KEY AUTOINCREMENT, source_action_id TEXT NOT NULL, target_action_id TEXT NOT NULL,
        dependency_type TEXT NOT NULL, created_at INTEGER NOT NULL,
        FOREIGN KEY (source_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        FOREIGN KEY (target_action_id) REFERENCES actions (action_id) ON DELETE CASCADE,
        UNIQUE(source_action_id, target_action_id, dependency_type)
    );""",
    """CREATE TABLE IF NOT EXISTS cognitive_states (
        state_id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        title TEXT NOT NULL,
        working_memory TEXT, focus_areas TEXT, context_actions TEXT, current_goals TEXT,
        created_at INTEGER NOT NULL, is_latest BOOLEAN NOT NULL,
        focal_memory_id TEXT REFERENCES memories(memory_id) ON DELETE SET NULL,
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
    "CREATE INDEX IF NOT EXISTS idx_workflows_idempotency_key ON workflows(idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_actions_idempotency ON actions(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_artifacts_idempotency ON artifacts(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_memories_idempotency ON memories(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_goals_idempotency ON goals(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_thought_chains_idempotency ON thought_chains(workflow_id, idempotency_key);",  # NEW
    "CREATE INDEX IF NOT EXISTS idx_thoughts_idempotency ON thoughts(thought_chain_id, idempotency_key);",  # NEW
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
    "CREATE INDEX IF NOT EXISTS idx_thoughts_relevant_memory ON thoughts(relevant_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_workflow ON memories(workflow_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(memory_level);",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC);",
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_action_id ON memories(action_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_thought_id ON memories(thought_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_artifact_id ON memories(artifact_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_type ON memory_links(link_type);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_memory_id ON embeddings(memory_id);",
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
    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        content, description, reasoning, tags,
        workflow_id UNINDEXED, memory_id UNINDEXED,
        content='memories', content_rowid='rowid', tokenize='porter unicode61'
    );""",
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


# ─────────────────────────────────────────────
# Primary DB helper – WAL-friendly, read-only
# ─────────────────────────────────────────────
class DBConnection:
    """
    Lightweight helper for SQLite / aiosqlite with:

    • single-call schema bootstrap (thread-safe)
    • per-connection UDF registration
    • explicit read-only vs read-write transactions
    • best-effort WAL checkpoint utility
    """

    _schema_initialized_paths: Set[str] = set()
    _schema_init_lock = asyncio.Lock()

    # --- WAL Checkpoint Control ---
    _last_wal_checkpoint_times: Dict[str, float] = {}  # db_path -> last_checkpoint_ts
    _wal_checkpoint_min_interval_seconds: int = 30  # Checkpoint at most every 30s per DB
    _wal_checkpoint_min_size_bytes: int = 5 * 1024 * 1024  # Checkpoint if WAL > 5MB
    _wal_checkpoint_lock = asyncio.Lock()  # Global lock for all checkpoint operations

    def __init__(self, db_path: str = agent_memory_config.db_path):
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._managed_conn: Optional[aiosqlite.Connection] = None
        # Initialize last checkpoint time for this specific db_path if not present
        if self.db_path not in DBConnection._last_wal_checkpoint_times:
            DBConnection._last_wal_checkpoint_times[self.db_path] = 0.0

    # ───────── UDF / PRAGMA configuration ─────────
    async def _configure_connection(self, conn: aiosqlite.Connection) -> None:
        await conn.execute("PRAGMA journal_mode=DELETE;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute("PRAGMA busy_timeout=30000;")

        await conn.create_function("json_contains", 2, _json_contains, deterministic=True)
        await conn.create_function("json_contains_any", 2, _json_contains_any, deterministic=True)
        await conn.create_function("json_contains_all", 2, _json_contains_all, deterministic=True)
        await conn.create_function(
            "compute_memory_relevance", 5, _compute_memory_relevance, deterministic=True
        )

    # ───────── One-time schema bootstrap ─────────
    async def _ensure_schema_for_path(self) -> None:
        async with DBConnection._schema_init_lock:
            if self.db_path in DBConnection._schema_initialized_paths:
                return

            conn_init: Optional[aiosqlite.Connection] = None
            try:
                logger.info(f"Bootstrapping schema for {self.db_path}")
                conn_init = await aiosqlite.connect(
                    self.db_path, timeout=agent_memory_config.connection_timeout
                )
                conn_init.row_factory = aiosqlite.Row
                await self._configure_connection(conn_init)

                # Non-WAL mode (rollback journal) + performance pragmas
                await conn_init.executescript(
                    """
                    PRAGMA journal_mode=DELETE;      -- Switch to rollback journal mode (default)
                    PRAGMA synchronous=NORMAL;       -- Still safe for most purposes; change to FULL for max durability
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-32000;
                    PRAGMA mmap_size=2147483647;
                    """
                )

                await conn_init.execute("BEGIN IMMEDIATE;")
                for stmt in SCHEMA_STATEMENTS:
                    await conn_init.execute(stmt)
                await conn_init.commit()

                DBConnection._schema_initialized_paths.add(self.db_path)
                logger.info("Schema ready.")
            except Exception as exc:
                if conn_init and conn_init.in_transaction:
                    await conn_init.rollback()
                logger.critical(
                    f"CRITICAL: schema initialisation failed for {self.db_path}: {exc}",
                    exc_info=True,
                )
                raise ToolError(f"Schema initialisation failed: {exc}") from exc
            finally:
                if conn_init:
                    await conn_init.close()

    # ───────── Public helpers ─────────
    @contextlib.asynccontextmanager
    async def transaction(
        self,
        *,
        readonly: bool = False,
        mode: str | None = None,
    ):
        """
        Asynchronous context manager yielding an `aiosqlite.Connection`.

        readonly=True  ➜ file:…?mode=ro&cache=shared URI,   BEGIN DEFERRED;
        readonly=False ➜ normal R/W connection,             BEGIN {mode};
        """
        await self._ensure_schema_for_path()

        uri_mode = f"file:{self.db_path}?mode=ro&cache=shared" if readonly else self.db_path
        conn: Optional[aiosqlite.Connection] = None
        try:
            conn = await aiosqlite.connect(
                uri_mode,
                uri=readonly,
                timeout=agent_memory_config.connection_timeout,
            )
            conn.row_factory = aiosqlite.Row
            await self._configure_connection(conn)

            begin_stmt = "BEGIN DEFERRED;" if readonly else f"BEGIN {mode or 'IMMEDIATE'};"
            await conn.execute(begin_stmt)
            yield conn
            await conn.commit()
        except Exception:
            if conn and conn.in_transaction:
                await conn.rollback()
            raise
        finally:
            if conn:
                await conn.close()

    # Convenience for non-transactional quick reads
    async def __aenter__(self) -> aiosqlite.Connection:
        await self._ensure_schema_for_path()
        self._managed_conn = await aiosqlite.connect(
            self.db_path, timeout=agent_memory_config.connection_timeout
        )
        self._managed_conn.row_factory = aiosqlite.Row
        await self._configure_connection(self._managed_conn)
        return self._managed_conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._managed_conn:
            if self._managed_conn.in_transaction:
                await self._managed_conn.rollback()
            await self._managed_conn.close()
            self._managed_conn = None


# ──────────────────────────────
# SQLite JSON utility functions
# ──────────────────────────────
def _safe_json_loads(text: str | None) -> Any | None:
    """Robust, fast helper hiding JSON errors."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _json_contains(json_text: str | None, search_value: Any) -> bool:
    """True if *search_value* is an element of the JSON‐encoded list `json_text`."""
    data = _safe_json_loads(json_text)
    return isinstance(data, list) and search_value in data


def _json_contains_any(json_text: str | None, search_values_json: str | None) -> bool:
    """True if *any* element of `search_values_json` exists in the JSON list `json_text`."""
    data = _safe_json_loads(json_text)
    values = _safe_json_loads(search_values_json)
    return isinstance(data, list) and isinstance(values, list) and bool(set(values) & set(data))


def _json_contains_all(json_text: str | None, search_values_json: str | None) -> bool:
    """True if *every* element of `search_values_json` exists in the JSON list `json_text`."""
    data = _safe_json_loads(json_text)
    values = _safe_json_loads(search_values_json)
    return isinstance(data, list) and isinstance(values, list) and set(values).issubset(data)


# ───────────────────────────────────────────────────────────────
# Memory-relevance scoring UDF – kept deterministic for FTS use
# ───────────────────────────────────────────────────────────────


def _compute_memory_relevance(
    importance: float,
    confidence: float,
    created_at: int,
    access_count: int,
    last_accessed: int | None,
) -> float:
    """
    Composite relevance score ∈ [0, 10].

    importance     : author-assigned 1–10
    confidence     : model confidence 0–1
    created_at     : Unix time (s)
    access_count   : integer count
    last_accessed  : Unix time (s) or NULL
    """
    now = time.time()
    decay_rate = agent_memory_config.memory_decay_rate  # cached lookup
    created_at = created_at or now
    last_accessed = last_accessed or created_at

    age_hours = (now - created_at) / 3600
    recency_factor = 1.0 / (1.0 + (now - last_accessed) / 86_400)  # daily half-life
    decayed_importance = max(0.0, importance * (1.0 - decay_rate * age_hours))
    usage_boost = min(1.0 + access_count / 10.0, 2.0)

    relevance = decayed_importance * usage_boost * confidence * recency_factor
    return max(0.0, min(relevance, 10.0))


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


@with_tool_metrics
@with_error_handling
async def initialize_memory_system(
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Prepare the Unified Memory System for use.

    Steps
    -----
    1. Ensure the SQLite file exists and the full schema is present.
    2. Open a *read-only* snapshot to prove the DB is readable.
    3. Initialise the EmbeddingService.
    4. Return a status payload.

    Returns
    -------
    {
        success: bool,
        message: str,
        db_path: str,               # absolute
        embedding_service_functional: bool,
        embedding_service_warning: Optional[str],
        processing_time: float      # seconds
    }
    """
    t0 = time.perf_counter()
    logger.info("Initializing Unified Memory System…", emoji_key="rocket")

    db = DBConnection(db_path)

    try:
        # ───────── 1. Schema bootstrap (idempotent) ─────────
        await db._ensure_schema_for_path()

        # ───────── 2. Sanity check – read-only snapshot ─────
        async with db.transaction(readonly=True) as conn:
            await conn.execute_fetchone("SELECT count(*) FROM ums_internal_metadata")

        logger.success("Database schema verified.", emoji_key="database")

        # ───────── 3. Embedding service ─────────────────────
        embedding_service_warning: str | None = None
        try:
            es = get_embedding_service()
            if es.client is None:
                embedding_service_warning = (
                    "EmbeddingService client not available; embeddings disabled."
                )
                logger.error(embedding_service_warning, emoji_key="warning")
                raise ToolError(embedding_service_warning)
            logger.info("EmbeddingService active.", emoji_key="brain")
            embedding_ok = True
        except Exception as exc:
            if not isinstance(exc, ToolError):
                embedding_service_warning = (
                    f"Failed to initialise EmbeddingService: {exc}; embeddings disabled."
                )
                logger.error(embedding_service_warning, emoji_key="error", exc_info=True)
                raise ToolError(embedding_service_warning) from exc
            raise  # propagate pre-built ToolError

        # ───────── 4. Done ──────────────────────────────────
        dt = time.perf_counter() - t0
        logger.success(
            "Unified Memory System ready.",
            emoji_key="white_check_mark",
            time=dt,
        )
        return {
            "success": True,
            "message": "Unified Memory System initialised successfully.",
            "db_path": os.path.abspath(db_path),
            "embedding_service_functional": embedding_ok,
            "embedding_service_warning": embedding_service_warning,
            "processing_time": dt,
        }

    except Exception as exc:
        dt = time.perf_counter() - t0
        logger.error(
            f"Memory system initialisation failed: {exc}",
            emoji_key="x",
            exc_info=True,
            time=dt,
        )
        if isinstance(exc, ToolError):
            raise
        raise ToolError(f"Memory system initialisation failed: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def create_workflow(
    title: str,
    *,
    description: str | None = None,
    goal: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    parent_workflow_id: str | None = None,
    idempotency_key: Optional[str] = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    # --- START: Input Validation & Logging ---
    logger.info(
        f"UMS:create_workflow CALLED. Title: '{title}', Desc: '{str(description)[:50]}...', "
        f"Goal: '{str(goal)[:50]}...', Tags: {tags}, IdempotencyKey: {idempotency_key}"
    )
    if not isinstance(title, str) or not title.strip():
        logger.error("UMS:create_workflow VALIDATION FAILED: Title is empty or not a string.")
        raise ToolInputError("Workflow title must be a non-empty string.", param_name="title")
    # --- END: Input Validation & Logging ---

    now = int(time.time())
    db = DBConnection(db_path)

    def iso(ts):
        return safe_format_timestamp(ts)

    try:
        # Idempotency: Return existing workflow if idempotency_key matches
        if idempotency_key:
            logger.info(f"UMS:create_workflow: Checking idempotency for key '{idempotency_key}'...")
            async with db.transaction(readonly=True) as conn_check:
                existing_workflow_row = await conn_check.execute_fetchone(
                    "SELECT * FROM workflows WHERE idempotency_key = ?", (idempotency_key,)
                )
            if existing_workflow_row:
                existing_workflow_id = existing_workflow_row["workflow_id"]
                logger.info(f"UMS:create_workflow: Idempotency HIT for key '{idempotency_key}'. Returning existing WF ID: {existing_workflow_id}")
                
                # Fetch complete details for the idempotency hit payload
                async with db.transaction(readonly=True) as conn_details:
                    tag_rows = await conn_details.execute_fetchall(
                        "SELECT t.name FROM tags t JOIN workflow_tags wt ON wt.tag_id = t.tag_id WHERE wt.workflow_id = ?",
                        (existing_workflow_id,),
                    )
                    tags_list = [r["name"] for r in tag_rows]
                    
                    chain_row = await conn_details.execute_fetchone(
                        "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? ORDER BY created_at LIMIT 1",
                        (existing_workflow_id,),
                    )
                    primary_chain_id_existing = chain_row["thought_chain_id"] if chain_row else None
                    
                    deserialized_metadata_existing = await MemoryUtils.deserialize(existing_workflow_row.get("metadata"))

                # --- START: Logging for Idempotency Hit Return (COMPLETE) ---
                idempotency_hit_payload = {
                    "workflow_id": existing_workflow_id,
                    "title": existing_workflow_row["title"],
                    "description": existing_workflow_row["description"],
                    "goal": existing_workflow_row["goal"],
                    "status": existing_workflow_row["status"],
                    "created_at": existing_workflow_row["created_at"],
                    "created_at_iso": iso(existing_workflow_row["created_at"]),
                    "updated_at": existing_workflow_row["updated_at"],
                    "updated_at_iso": iso(existing_workflow_row["updated_at"]),
                    "completed_at": existing_workflow_row.get("completed_at"),
                    "completed_at_iso": iso(existing_workflow_row.get("completed_at")) if existing_workflow_row.get("completed_at") else None,
                    "parent_workflow_id": existing_workflow_row.get("parent_workflow_id"),
                    "metadata": deserialized_metadata_existing, # Use deserialized version
                    "last_active": existing_workflow_row.get("last_active"),
                    "last_active_iso": iso(existing_workflow_row.get("last_active")) if existing_workflow_row.get("last_active") else None,
                    "tags": tags_list,
                    "primary_thought_chain_id": primary_chain_id_existing,
                    "idempotency_hit": True,
                    "success": True,
                }
                
                log_payload_idem_str = "Error serializing idempotency payload for logging"
                try:
                    log_payload_idem_str = json.dumps(idempotency_hit_payload, default=str, indent=2)
                except Exception as json_log_idem_e:
                    logger.error(f"UMS:create_workflow (Idempotency Hit): Failed to serialize payload for logging: {json_log_idem_e}")
                    log_payload_idem_str = str(idempotency_hit_payload)

                logger.info(f"UMS:create_workflow (Idempotency Hit) RETURNING PAYLOAD (len: {len(log_payload_idem_str)}):\n{log_payload_idem_str}")
                return idempotency_hit_payload
                # --- END: Logging for Idempotency Hit Return ---

        # --- Main path (no idempotency hit or no key) ---
        workflow_id = MemoryUtils.generate_id()
        chain_id = MemoryUtils.generate_id()
        logger.info(f"UMS:create_workflow: Generated new workflow_id: {workflow_id}, chain_id: {chain_id}")

        async with db.transaction(mode="IMMEDIATE") as conn:
            if parent_workflow_id:
                logger.info(f"UMS:create_workflow: Validating parent_workflow_id: {parent_workflow_id}")
                p = await conn.execute_fetchone(
                    "SELECT 1 FROM workflows WHERE workflow_id=?",
                    (parent_workflow_id,),
                )
                if p is None:
                    logger.error(f"UMS:create_workflow: Parent workflow ID '{parent_workflow_id}' not found.")
                    raise ToolInputError(
                        f"Parent workflow ID '{parent_workflow_id}' not found.",
                        param_name="parent_workflow_id",
                    )

            serialized_metadata = await MemoryUtils.serialize(metadata)
            title_stripped = title.strip() if title else ""
            desc_str = description or ""
            goal_str = goal or ""
            
            logger.info(
                f"UMS:create_workflow: Inserting workflow record. "
                f"Title='{title_stripped}', Desc length={len(desc_str)}, Goal length={len(goal_str)}, "
                f"Metadata length={len(serialized_metadata or '')}"
            )

            await conn.execute(
                """
                INSERT INTO workflows (
                    workflow_id, title, description, goal, status,
                    created_at, updated_at, parent_workflow_id, metadata, last_active,
                    idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    workflow_id,
                    title_stripped, # Use stripped title
                    description,
                    goal,
                    WorkflowStatus.ACTIVE.value,
                    now,
                    now,
                    parent_workflow_id,
                    serialized_metadata,
                    now,
                    idempotency_key,
                ),
            )

            if tags:
                logger.info(f"UMS:create_workflow: Processing tags: {tags}")
                await MemoryUtils.process_tags(conn, workflow_id, tags, entity_type="workflow")

            thought_chain_title = f"Main reasoning for: {title_stripped[:100]}"
            logger.info(f"UMS:create_workflow: Inserting primary thought chain with title: '{thought_chain_title}'")
            await conn.execute(
                "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?,?,?,?)",
                (chain_id, workflow_id, thought_chain_title, now),
            )

            if goal:
                thought_id = MemoryUtils.generate_id()
                seq = await MemoryUtils.get_next_sequence_number(
                    conn, chain_id, "thoughts", "thought_chain_id"
                )
                logger.info(f"UMS:create_workflow: Inserting initial GOAL thought (ID: {thought_id}, Seq: {seq}) for goal text: '{goal_str[:70]}...'")
                await conn.execute(
                    "INSERT INTO thoughts (thought_id, thought_chain_id, thought_type, content, sequence_number, created_at) VALUES (?,?,?,?,?,?)",
                    (thought_id, chain_id, ThoughtType.GOAL.value, goal, seq, now),
                )

            existing_state_row = await conn.execute_fetchone(
                "SELECT 1 FROM cognitive_states WHERE state_id = ?", (workflow_id,)
            )
            if not existing_state_row:
                cognitive_state_title = f"Primary context for workflow: {title_stripped[:100]}"
                logger.info(f"UMS:create_workflow: Inserting initial cognitive_state (ID: {workflow_id}) with title: '{cognitive_state_title}'")
                await conn.execute(
                    """
                    INSERT INTO cognitive_states (
                        state_id, workflow_id, title, working_memory, focus_areas, context_actions, current_goals,
                        created_at, is_latest, focal_memory_id, last_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workflow_id, workflow_id, cognitive_state_title,
                        None, None, None, None, 
                        now, True, None, now,
                    ),
                )
            else:
                logger.warning(f"UMS:create_workflow: Cognitive state with ID {workflow_id} unexpectedly already existed. Not inserting a new one.")
        
        final_payload = {
            "workflow_id": workflow_id,
            "title": title_stripped, # Use stripped title
            "description": description,
            "goal": goal,
            "status": WorkflowStatus.ACTIVE.value,
            "created_at": now,
            "created_at_iso": iso(now),
            "updated_at": now,
            "updated_at_iso": iso(now),
            "tags": tags or [],
            "primary_thought_chain_id": chain_id,
            "idempotency_hit": False,
            "success": True,
        }

        log_payload_str = "Error serializing payload for logging"
        try:
            log_payload_str = json.dumps(final_payload, default=str, indent=2)
        except Exception as json_log_e:
            logger.error(f"UMS:create_workflow: Failed to serialize final_payload for logging: {json_log_e}")
            log_payload_str = str(final_payload)

        logger.info(f"UMS:create_workflow: Successfully created workflow '{_fmt_id(workflow_id)}'.")
        logger.info(f"UMS:create_workflow FINAL RETURN PAYLOAD (len: {len(log_payload_str)}):\n{log_payload_str}")
        
        logger.info(f"UMS:create_workflow Payload Field Lengths: title='{len(title_stripped)}', "
                     f"description='{len(desc_str)}', goal='{len(goal_str)}'")

        return final_payload

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"UMS:create_workflow UNEXPECTED ERROR for title ('{str(title)[:50]}'): {exc}", exc_info=True)
        raise ToolError(f"Failed to create workflow due to unexpected error: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def update_workflow_status(
    workflow_id: str,
    status: str,
    *,
    completion_message: str | None = None,
    update_tags: list[str] | None = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Change the *status* of a workflow, optionally append a completion / reflection
    thought and/or replace its tags. Conditionally triggers a WAL checkpoint.
    """
    try:
        status_enum = WorkflowStatus(status.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid status '{status}'. "
            f"Must be one of: {', '.join(s.value for s in WorkflowStatus)}",
            param_name="status",
        ) from exc
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    now = int(time.time())
    db = DBConnection(db_path)  # DBConnection instance for this specific db_path

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            exists = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if not exists:
                raise ToolInputError(f"Workflow not found: {workflow_id}", param_name="workflow_id")

            cols: dict[str, Any] = {
                "status": status_enum.value,
                "updated_at": now,
                "last_active": now,
            }
            if status_enum in (
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.ABANDONED,
            ):
                cols["completed_at"] = now

            set_sql = ", ".join(f"{k}=?" for k in cols)
            await conn.execute(
                f"UPDATE workflows SET {set_sql} WHERE workflow_id = ?",
                [*cols.values(), workflow_id],
            )

            if completion_message:
                tc_row = await conn.execute_fetchone(
                    "SELECT thought_chain_id "
                    "FROM thought_chains WHERE workflow_id = ? "
                    "ORDER BY created_at LIMIT 1",
                    (workflow_id,),
                )
                if tc_row:
                    chain_id = tc_row["thought_chain_id"]
                    seq_no = await MemoryUtils.get_next_sequence_number(
                        conn, chain_id, "thoughts", "thought_chain_id"
                    )
                    thought_id = MemoryUtils.generate_id()
                    thought_type_enum = (
                        ThoughtType.SUMMARY
                        if status_enum == WorkflowStatus.COMPLETED
                        else ThoughtType.REFLECTION
                    )
                    await conn.execute(
                        "INSERT INTO thoughts "
                        "(thought_id, thought_chain_id, thought_type, content, "
                        " sequence_number, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            thought_id,
                            chain_id,
                            thought_type_enum.value,
                            completion_message,
                            seq_no,
                            now,
                        ),
                    )
                    logger.debug(
                        f"Added {thought_type_enum.value} thought {thought_id} for workflow {workflow_id}"
                    )
                else:
                    logger.warning(
                        f"No thought chain found for workflow {workflow_id}; completion message skipped."
                    )

            if update_tags:
                await MemoryUtils.process_tags(
                    conn, workflow_id, update_tags, entity_type="workflow"
                )
        # Transaction committed successfully here

        result = {
            "workflow_id": workflow_id,
            "status": status_enum.value,
            "updated_at_iso": safe_format_timestamp(now),
            "success": True,
        }
        if "completed_at" in cols:
            result["completed_at_iso"] = safe_format_timestamp(now)

        logger.info(
            f"Workflow {_fmt_id(workflow_id)} status ➜ '{status_enum.value}'",
            emoji_key="arrows_counterclockwise",
        )
        return result

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"update_workflow_status({_fmt_id(workflow_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to update workflow status: {exc}") from exc


# --- 3. Action Tracking Tools ---
@with_tool_metrics
@with_error_handling
async def record_action_start(
    workflow_id: str,
    action_type: str,
    reasoning: str,
    *,
    tool_name: str | None = None,
    tool_args: Dict[str, Any] | None = None,
    title: str | None = None,
    parent_action_id: str | None = None,
    tags: list[str] | None = None,
    related_thought_id: str | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    # ... (validation for action_type, reasoning, tool_name remains the same) ...
    try:
        action_type_enum = ActionType(action_type.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid action_type '{action_type}'. Valid types: {', '.join(t.value for t in ActionType)}",
            param_name="action_type",
        ) from exc
    if not reasoning:
        raise ToolInputError("Reasoning must be a non-empty string.", param_name="reasoning")
    if action_type_enum is ActionType.TOOL_USE and not tool_name:
        raise ToolInputError("tool_name required for 'tool_use' actions.", param_name="tool_name")

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    # Helper to fetch full action details for consistent return on idempotency hit
    async def _fetch_existing_action_details(
        conn_fetch: aiosqlite.Connection, existing_action_id: str
    ) -> Dict[str, Any]:
        action_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM actions WHERE action_id = ?", (existing_action_id,)
        )
        if not action_row:
            raise ToolError(
                f"Failed to re-fetch existing action {existing_action_id} on idempotency hit."
            )

        action_data = dict(action_row)
        tag_rows = await conn_fetch.execute_fetchall(
            "SELECT t.name FROM tags t JOIN action_tags at ON at.tag_id = t.tag_id WHERE at.action_id = ?",
            (existing_action_id,),
        )
        action_data["tags"] = [r["name"] for r in tag_rows]
        action_data["tool_args"] = await MemoryUtils.deserialize(action_data.get("tool_args"))

        # Find linked memory_id if exists (the one created when action originally started)
        linked_mem_row = await conn_fetch.execute_fetchone(
            "SELECT memory_id FROM memories WHERE action_id = ? AND memory_type = ? ORDER BY created_at ASC LIMIT 1",
            (existing_action_id, MemoryType.ACTION_LOG.value),
        )
        linked_memory_id_existing = linked_mem_row["memory_id"] if linked_mem_row else None

        return {
            "action_id": existing_action_id,
            "workflow_id": action_data["workflow_id"],
            "action_type": action_data["action_type"],
            "title": action_data["title"],
            "tool_name": action_data.get("tool_name"),
            "status": action_data[
                "status"
            ],  # Should be IN_PROGRESS or a terminal state if it somehow got completed/failed
            "started_at_unix": action_data["started_at"],
            "started_at_iso": to_iso_z(action_data["started_at"]),
            "sequence_number": action_data["sequence_number"],
            "tags": action_data["tags"],
            "linked_memory_id": linked_memory_id_existing,
            "idempotency_hit": True,
            "success": True,
            "processing_time": time.perf_counter() - t0_perf,
        }

    try:
        if idempotency_key:
            async with db.transaction(readonly=True) as conn_check:
                existing_action_row = await conn_check.execute_fetchone(
                    "SELECT action_id FROM actions WHERE workflow_id = ? AND idempotency_key = ?",
                    (workflow_id, idempotency_key),
                )
            if existing_action_row:
                existing_action_id = existing_action_row["action_id"]
                logger.info(
                    f"Idempotency hit for record_action_start (key='{idempotency_key}'). Returning existing action {_fmt_id(existing_action_id)}."
                )
                async with db.transaction(
                    readonly=True
                ) as conn_details:  # New transaction for fetching details
                    return await _fetch_existing_action_details(conn_details, existing_action_id)

        action_id = MemoryUtils.generate_id()
        memory_id = MemoryUtils.generate_id()  # For the new episodic memory

        async with db.transaction(mode="IMMEDIATE") as conn:
            # ... (existence checks for workflow_id, parent_action_id, related_thought_id remain the same) ...
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
            ):
                raise ToolInputError("Workflow not found.", param_name="workflow_id")
            if parent_action_id and not await conn.execute_fetchone(
                "SELECT 1 FROM actions WHERE action_id=? AND workflow_id=?",
                (parent_action_id, workflow_id),
            ):
                raise ToolInputError(
                    f"Parent action '{parent_action_id}' not in workflow.",
                    param_name="parent_action_id",
                )
            if related_thought_id and not await conn.execute_fetchone(
                "SELECT 1 FROM thoughts t JOIN thought_chains c ON c.thought_chain_id = t.thought_chain_id WHERE t.thought_id=? AND c.workflow_id=?",
                (related_thought_id, workflow_id),
            ):
                raise ToolInputError(
                    "related_thought_id not found in workflow.",
                    param_name="related_thought_id",
                )

            seq_no = await MemoryUtils.get_next_sequence_number(
                conn, workflow_id, "actions", "workflow_id"
            )
            auto_title = title or (
                f"Using {tool_name}"
                if action_type_enum is ActionType.TOOL_USE and tool_name
                else (
                    reasoning.split(".", 1)[0][:50] or f"{action_type_enum.value.title()} #{seq_no}"
                )
            )

            await conn.execute(
                """
                INSERT INTO actions (
                    action_id, workflow_id, parent_action_id, action_type, title, reasoning, 
                    tool_name, tool_args, status, started_at, sequence_number, idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,  # MODIFIED: Added idempotency_key
                (
                    action_id,
                    workflow_id,
                    parent_action_id,
                    action_type_enum.value,
                    auto_title,
                    reasoning,
                    tool_name,
                    await MemoryUtils.serialize(tool_args),
                    ActionStatus.IN_PROGRESS.value,
                    now_unix,
                    seq_no,
                    idempotency_key,  # MODIFIED: Added value
                ),
            )

            if tags:
                await MemoryUtils.process_tags(conn, action_id, tags, entity_type="action")
            if related_thought_id:
                await conn.execute(
                    "UPDATE thoughts SET relevant_action_id=? WHERE thought_id=?",
                    (action_id, related_thought_id),
                )

            mem_content = (
                f"Started action [{seq_no}] '{auto_title}' ({action_type_enum.value}). Reasoning: {reasoning}"
                + (f" Tool: {tool_name}." if tool_name else "")
            )
            final_mem_tags_list = ["action_start", action_type_enum.value]
            if tags:
                final_mem_tags_list.extend(tags)
            mem_tags_json = json.dumps(list(set(final_mem_tags_list)))

            await conn.execute(
                """INSERT INTO memories (memory_id, workflow_id, action_id, content, memory_level, memory_type, importance, confidence, tags, created_at, updated_at, access_count, last_accessed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
                (
                    memory_id,
                    workflow_id,
                    action_id,
                    mem_content,
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
            await conn.execute(
                "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
                (now_unix, now_unix, workflow_id),
            )

        return {
            "action_id": action_id,
            "workflow_id": workflow_id,
            "action_type": action_type_enum.value,
            "title": auto_title,
            "tool_name": tool_name,
            "status": ActionStatus.IN_PROGRESS.value,
            "started_at_unix": now_unix,
            "started_at_iso": to_iso_z(now_unix),
            "sequence_number": seq_no,
            "tags": tags or [],
            "linked_memory_id": memory_id,
            "idempotency_hit": False,  # NEW
            "success": True,
            "processing_time": time.perf_counter() - t0_perf,
        }
    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"record_action_start failed for workflow {workflow_id}: {exc}", exc_info=True)
        raise ToolError(f"Failed to start action: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def record_action_completion(
    action_id: str,
    *,
    status: str = "completed",
    tool_result: Optional[Any] = None,
    summary: Optional[str] = None,
    conclusion_thought: Optional[str] = None,
    conclusion_thought_type: str = "inference",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Mark an action terminal (completed / failed / skipped), persist the tool-result,
    optionally append a concluding thought, and update any linked action-log memory.
    Conditionally triggers a WAL checkpoint.
    """
    start_perf = time.perf_counter()

    try:
        status_enum = ActionStatus(status.lower())
        if status_enum not in (
            ActionStatus.COMPLETED,
            ActionStatus.FAILED,
            ActionStatus.SKIPPED,
        ):
            raise ValueError("Invalid terminal status for action.")  # More specific error message
    except ValueError as e:
        valid_terminal_statuses = ", ".join(
            s.value for s in (ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.SKIPPED)
        )
        raise ToolInputError(
            f"Invalid status '{status}'. Must be one of: {valid_terminal_statuses}",
            param_name="status",
        ) from e

    thought_enum: Optional[ThoughtType] = None
    if conclusion_thought:
        try:
            thought_enum = ThoughtType(conclusion_thought_type.lower())
        except ValueError as e:
            valid_thought_types = ", ".join(t.value for t in ThoughtType)
            raise ToolInputError(
                f"Invalid thought type '{conclusion_thought_type}'. Must be one of: {valid_thought_types}",
                param_name="conclusion_thought_type",
            ) from e

    now = int(time.time())
    conclusion_thought_id: Optional[str] = None
    workflow_id_for_response: Optional[str] = None  # To store workflow_id for the final response

    db = DBConnection(db_path)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            row = await conn.execute_fetchone(
                "SELECT workflow_id, status FROM actions WHERE action_id = ?",
                (action_id,),
            )
            if row is None:
                raise ToolInputError(f"Action not found: {action_id}", param_name="action_id")

            # Capture workflow_id for response and logging outside transaction if needed
            workflow_id_for_response = row["workflow_id"]
            current_action_status_in_db = row["status"]

            if current_action_status_in_db in (
                ActionStatus.COMPLETED.value,
                ActionStatus.FAILED.value,
                ActionStatus.SKIPPED.value,
            ):
                logger.warning(
                    f"record_action_completion: action {action_id} already in terminal state "
                    f"'{current_action_status_in_db}'; overwriting with new status '{status_enum.value}'."
                )

            serialized_tool_result = await MemoryUtils.serialize(tool_result)
            if serialized_tool_result is None and tool_result is not None:
                logger.warning(
                    f"Tool result for action {action_id} could not be serialized, will store as NULL or error marker."
                )
                # Depending on strictness, you might want to store a placeholder like {"error": "serialization_failed"}

            await conn.execute(
                """
                UPDATE actions
                SET status       = ?,
                    completed_at = ?,
                    tool_result  = ?
                WHERE action_id  = ?
                """,
                (
                    status_enum.value,
                    now,
                    serialized_tool_result,
                    action_id,
                ),
            )
            await conn.execute(
                "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
                (now, now, workflow_id_for_response),
            )

            if (
                conclusion_thought and thought_enum and workflow_id_for_response
            ):  # Ensure workflow_id is available
                chain_id = await conn.execute_fetchval(
                    "SELECT thought_chain_id "
                    "FROM thought_chains "
                    "WHERE workflow_id = ? "
                    "ORDER BY created_at "  # Get the primary/first chain
                    "LIMIT 1",
                    (workflow_id_for_response,),
                )
                if chain_id:
                    seq = await MemoryUtils.get_next_sequence_number(
                        conn, chain_id, "thoughts", "thought_chain_id"
                    )
                    conclusion_thought_id = MemoryUtils.generate_id()
                    await conn.execute(
                        """
                        INSERT INTO thoughts (
                            thought_id, thought_chain_id, thought_type, content,
                            sequence_number, created_at, relevant_action_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conclusion_thought_id,
                            chain_id,
                            thought_enum.value,
                            conclusion_thought,
                            seq,
                            now,
                            action_id,
                        ),
                    )
                    logger.debug(
                        f"Added concluding thought {conclusion_thought_id} to chain {chain_id} for action {action_id}"
                    )
                else:
                    logger.warning(
                        f"No thought chain found for workflow {workflow_id_for_response}; "
                        "conclusion thought not recorded for action {action_id}."
                    )

            mem_row = await conn.execute_fetchone(
                "SELECT memory_id, content FROM memories WHERE action_id = ? AND memory_type = ?",
                (action_id, MemoryType.ACTION_LOG.value),
            )
            if mem_row and workflow_id_for_response:  # Ensure workflow_id is available for logging
                mem_id = mem_row["memory_id"]
                parts = [f"Completed ({status_enum.value})."]
                if summary:
                    parts.append(f"Summary: {summary}")

                if tool_result is not None:
                    if isinstance(tool_result, dict):
                        parts.append(f"Result: [Dict with {len(tool_result)} keys]")
                    elif isinstance(tool_result, list):
                        parts.append(f"Result: [List with {len(tool_result)} items]")
                    elif isinstance(tool_result, str):
                        parts.append(
                            f"Result: {tool_result[:97]}{'…' if len(tool_result) > 100 else ''}"
                        )
                    elif tool_result is True:
                        parts.append("Result: Success (True)")
                    elif tool_result is False:
                        parts.append("Result: Failure (False)")
                    else:
                        parts.append(f"Result: [Object {type(tool_result).__name__}]")

                new_content = f"{mem_row['content']} {' '.join(parts)}"
                # Cap new_content length to avoid excessive growth, matching MemoryUtils.serialize limits roughly
                max_len = agent_memory_config.max_text_length
                if len(new_content.encode("utf-8")) > max_len:
                    new_content_bytes = new_content.encode("utf-8")[
                        : max_len - 3
                    ]  # -3 for ellipsis
                    new_content = new_content_bytes.decode("utf-8", errors="replace") + "..."
                    logger.warning(
                        f"Action log memory content for {mem_id} truncated during update."
                    )

                importance_factor = (
                    1.2
                    if status_enum == ActionStatus.FAILED
                    else 0.8
                    if status_enum == ActionStatus.SKIPPED
                    else 1.0
                )
                current_importance_row = await conn.execute_fetchone(
                    "SELECT importance FROM memories WHERE memory_id = ?", (mem_id,)
                )
                new_importance = (
                    current_importance_row["importance"] if current_importance_row else 5.0
                ) * importance_factor
                new_importance = max(0.0, min(10.0, new_importance))  # Clamp importance

                await conn.execute(
                    """
                    UPDATE memories
                    SET content       = ?,
                        importance    = ?, 
                        updated_at    = ?,
                        last_accessed = ? 
                    WHERE memory_id  = ?
                    """,
                    (new_content, new_importance, now, now, mem_id),
                )
                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id_for_response,  # Use the captured workflow_id
                    "update_from_action_completion",
                    mem_id,
                    action_id,
                    {"status": status_enum.value, "summary_added": bool(summary)},
                )
                logger.debug(f"Updated memory {mem_id} for action {action_id}")
            elif workflow_id_for_response:  # mem_row was None
                logger.debug(
                    f"No action_log memory found for action {action_id} in workflow {workflow_id_for_response} to update."
                )
        # Transaction committed successfully here

        return {
            "action_id": action_id,
            "workflow_id": workflow_id_for_response,  # Return the captured workflow_id
            "status": status_enum.value,
            "completed_at_iso": to_iso_z(now),
            "conclusion_thought_id": conclusion_thought_id,
            "success": True,
            "processing_time": time.perf_counter() - start_perf,
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"record_action_completion({action_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to record action completion: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_action_details(
    *,
    action_id: str | None = None,
    action_ids: list[str] | None = None,
    include_dependencies: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Fetch one or many actions (plus optional dependency graph).

    • Raw integer timestamps are preserved; *_iso companions are added.
    • `tool_args` / `tool_result` columns are JSON-deserialised.
    • When *include_dependencies* is True a bidirectional dependency map is returned
      for each action:  `{depends_on: [...], dependent_actions: [...]}`.
    """
    if not action_id and not action_ids:
        raise ToolInputError("Provide action_id or action_ids", param_name="action_id")

    targets: list[str] = [action_id] if action_id else action_ids or []
    if not targets:
        raise ToolInputError("No valid action IDs supplied.", param_name="action_id")

    start = time.perf_counter()
    db = DBConnection(db_path)

    def _add_iso(obj: dict[str, Any], *cols: str) -> None:
        for c in cols:
            if (ts := obj.get(c)) is not None:
                obj[f"{c}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            ph = ", ".join("?" * len(targets))

            # ───────── primary query ─────────
            rows = await conn.execute_fetchall(
                f"""
                SELECT a.*,
                       GROUP_CONCAT(DISTINCT t.name) AS tags_str
                FROM   actions a
                LEFT   JOIN action_tags at ON at.action_id = a.action_id
                LEFT   JOIN tags        t  ON t.tag_id    = at.tag_id
                WHERE  a.action_id IN ({ph})
                GROUP  BY a.action_id
                """,
                targets,
            )

            if not rows:
                raise ToolInputError(
                    f"No actions found for IDs: {', '.join(targets[:5])}"
                    + ("…" if len(targets) > 5 else ""),
                    param_name="action_id",
                )

            # ───────── dependencies (single query) ─────────
            dep_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
            if include_dependencies:
                dep_rows = await conn.execute_fetchall(
                    """
                    SELECT source_action_id, target_action_id, dependency_type
                    FROM   dependencies
                    WHERE  source_action_id IN ({ph})
                       OR  target_action_id IN ({ph})
                    """.format(ph=ph),
                    targets * 2,
                )
                for r in dep_rows:
                    src, tgt, typ = r
                    dep_map.setdefault(src, {}).setdefault("depends_on", []).append(
                        {"action_id": tgt, "type": typ}
                    )
                    dep_map.setdefault(tgt, {}).setdefault("dependent_actions", []).append(
                        {"action_id": src, "type": typ}
                    )

            # ───────── row post-processing ─────────
            actions: list[dict[str, Any]] = []
            for r in rows:
                a = dict(r)
                a["tags"] = r["tags_str"].split(",") if r["tags_str"] else []
                a.pop("tags_str", None)

                # JSON columns
                for col in ("tool_args", "tool_result"):
                    if a.get(col):
                        a[col] = await MemoryUtils.deserialize(a[col])

                # Dependency attach
                if include_dependencies:
                    a["dependencies"] = dep_map.get(
                        a["action_id"],
                        {"depends_on": [], "dependent_actions": []},
                    )

                _add_iso(a, "started_at", "completed_at")
                actions.append(a)

            elapsed = time.perf_counter() - start
            logger.info(
                f"get_action_details: {len(actions)} actions returned in {elapsed:.3f}s",
                emoji_key="bolt",
            )
            return {"actions": actions, "success": True, "processing_time": elapsed}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_action_details failed", exc_info=True)
        raise ToolError(f"Failed to get action details: {exc}") from exc


# ======================================================
# Contextual Summarization
# ======================================================


@with_tool_metrics
@with_error_handling
async def summarize_context_block(
    text_to_summarize: str,
    *,
    target_tokens: int = 500,
    context_type: str = "actions",  # "actions" | "memories" | "thoughts" | …
    workflow_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Compress an arbitrary context blob using the configured LLM provider.

    • Prompt is specialised per *context_type*.
    • Provider/model defaulting mirrors get_workflow_details logic.
    • Logs a `compress_context` entry in *memory_operations* when workflow_id supplied.
    """
    t0 = time.perf_counter()

    if not text_to_summarize:
        raise ToolInputError("Text to summarise cannot be empty.", param_name="text_to_summarize")

    # ───────────────────────── prompt selection ─────────────────────────
    tmpl: str
    if context_type == "actions":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise the following ACTION HISTORY,
retaining IDs and the most salient events.

Focus:
1. State-changing or output-producing actions
2. Failures + error reasons
3. Last 2-3 actions (even if trivial)
4. Actions that created artifacts / memories
5. Sequencing

Aim for ~{target_tokens} tokens.

ACTION HISTORY:
{text}

SUMMARY:
"""
    elif context_type == "memories":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise these MEMORY ENTRIES.

Prioritise:
1. importance > 7
2. confidence > 0.8
3. insights over observations
4. preserve memory IDs
5. linked memories / networks

~{target_tokens} tokens.

MEMORIES:
{text}

SUMMARY:
"""
    elif context_type == "thoughts":
        tmpl = """
You are an expert context summariser for an AI agent. Summarise the following THOUGHT CHAINS.

Prioritise:
1. goals, decisions, conclusions
2. key hypotheses / reflections
3. most recent thoughts
4. preserve thought IDs

~{target_tokens} tokens.

THOUGHTS:
{text}

SUMMARY:
"""
    else:  # generic
        tmpl = """
Summarise the text below for an AI agent. Preserve:
1. recent, goal-relevant info
2. critical state details
3. unique identifiers
4. significant events/insights

Target length: ~{target_tokens} tokens.

TEXT:
{text}

SUMMARY:
"""

    # ───────────────────────── provider / model ─────────────────────────
    cfg = get_config()
    provider_name = provider or cfg.default_provider or LLMGatewayProvider.ANTHROPIC.value
    prov = await get_provider(provider_name)
    if prov is None:
        raise ToolError(f"Provider '{provider_name}' unavailable.")

    model_name = model or prov.get_default_model()
    if model_name is None:  # hard fallback
        fallbacks = {
            LLMGatewayProvider.OPENAI.value: "gpt-3.5-turbo",
            LLMGatewayProvider.ANTHROPIC.value: "claude-3-haiku-20240307",
        }
        model_name = fallbacks.get(provider_name)
        if model_name is None:
            raise ToolError(f"No model specified and no default for provider '{provider_name}'.")

    # ───────────────────────── LLM call ─────────────────────────
    prompt = tmpl.format(text=text_to_summarize, target_tokens=target_tokens).lstrip()
    out = await prov.generate_completion(
        prompt=prompt,
        model=model_name,
        max_tokens=target_tokens + 150,
        temperature=0.2,
    )

    summary = out.text.strip()
    if not summary:
        logger.warning("LLM returned empty summary for context_type='%s'.", context_type)
        summary = ""

    comp_ratio = len(summary) / max(len(text_to_summarize), 1)

    # ───────────────────────── logging (optional) ───────────────────────
    if workflow_id:
        db = DBConnection(db_path)
        async with db.transaction(mode="IMMEDIATE") as conn:  # write txn
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "compress_context",
                memory_id=None,
                action_id=None,
                operation_data={
                    "context_type": context_type,
                    "original_length": len(text_to_summarize),
                    "summary_length": len(summary),
                    "compression_ratio": comp_ratio,
                    "provider": provider_name,
                    "model": model_name,
                },
            )

    # ───────────────────────── return ─────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Summarised {context_type} context ({len(text_to_summarize)}→{len(summary)} chars, ratio {comp_ratio:.2f}) via {provider_name}/{model_name}",
        emoji_key="compression",
        time=elapsed,  # Pass elapsed as 'time' kwarg
    )
    return {
        "summary": summary,
        "context_type": context_type,
        "compression_ratio": comp_ratio,
        "processing_time": elapsed,
        "success": True,
    }


# ======================================================
# 3.5 Action Dependency Tools
# ======================================================

@with_tool_metrics
@with_error_handling
async def add_action_dependency(
    source_action_id: str,
    target_action_id: str,
    *,
    dependency_type: str = "requires",  # requires | informs | blocks …
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Register a directed relationship between two actions.
    WAL checkpointing is handled by a separate periodic mechanism.
    """
    if not source_action_id:
        raise ToolInputError("Source action ID required.", param_name="source_action_id")
    if not target_action_id:
        raise ToolInputError("Target action ID required.", param_name="target_action_id")
    if source_action_id == target_action_id:
        raise ToolInputError("Source and target IDs must differ.", param_name="source_action_id")
    if not dependency_type.strip():
        raise ToolInputError("Dependency type cannot be empty.", param_name="dependency_type")

    t0_perf = time.perf_counter()  # Use perf_counter for more precise duration
    now_unix = int(time.time())
    dependency_id: Optional[int] = None

    db = DBConnection(db_path)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            src = await conn.execute_fetchone(
                "SELECT workflow_id FROM actions WHERE action_id=?", (source_action_id,)
            )
            if src is None:
                raise ToolInputError(
                    f"Source action {_fmt_id(source_action_id)} not found.",
                    param_name="source_action_id",
                )
            tgt = await conn.execute_fetchone(
                "SELECT workflow_id FROM actions WHERE action_id=?", (target_action_id,)
            )
            if tgt is None:
                raise ToolInputError(
                    f"Target action {_fmt_id(target_action_id)} not found.",
                    param_name="target_action_id",
                )
            if src["workflow_id"] != tgt["workflow_id"]:
                raise ToolInputError(
                    "Source and target actions belong to different workflows.",
                    param_name="target_action_id",
                )
            workflow_id = src["workflow_id"]

            async with conn.execute(
                """
                INSERT OR IGNORE INTO dependencies
                    (source_action_id, target_action_id, dependency_type, created_at)
                VALUES (?,?,?,?)
                """,
                (source_action_id, target_action_id, dependency_type, now_unix),
            ) as cur:
                if cur.rowcount:  # If a new row was inserted
                    dependency_id = cur.lastrowid
                else:  # Row already existed, fetch its ID
                    row = await conn.execute_fetchone(
                        """
                        SELECT dependency_id
                        FROM dependencies
                        WHERE source_action_id=? AND target_action_id=? AND dependency_type=?
                        """,
                        (source_action_id, target_action_id, dependency_type),
                    )
                    dependency_id = (
                        row["dependency_id"] if row else None
                    )  # Should exist due to IGNORE logic

            await conn.execute(
                "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
                (now_unix, now_unix, workflow_id),
            )

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "add_dependency",
                None,  # memory_id
                source_action_id,  # Log against source action
                {
                    "target_action_id": target_action_id,
                    "dependency_type": dependency_type,
                    "db_dependency_id": dependency_id,
                },
            )
        # Transaction committed successfully here

        # NO explicit WAL checkpoint call here. Rely on periodic agent-driven or SQLite auto-checkpoint.

        processing_time = time.perf_counter() - t0_perf
        logger.info(
            f"Dependency '{dependency_type}' {_fmt_id(source_action_id)} → {_fmt_id(target_action_id)} "
            f"(id={dependency_id if dependency_id is not None else 'exists'}) created/verified.",  # More precise logging
            emoji_key="link",
            time=processing_time,
        )

        return {
            "source_action_id": source_action_id,
            "target_action_id": target_action_id,
            "dependency_type": dependency_type,
            "dependency_id": dependency_id,  # This will be the ID of the dependency row
            "created_at_unix": now_unix,
            "created_at_iso": to_iso_z(now_unix),
            "success": True,
            "processing_time": processing_time,
        }
    except ToolInputError:
        raise
    except Exception as exc:
        processing_time = time.perf_counter() - t0_perf
        logger.error(
            f"add_action_dependency failed for {source_action_id}->{target_action_id}: {exc}",
            exc_info=True,
            time=processing_time,
        )
        raise ToolError(f"Failed to add action dependency: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_action_dependencies(
    action_id: str,
    *,
    direction: str = "downstream",  # 'downstream' → children, 'upstream' → parents
    dependency_type: str | None = None,  # filter by dependency edge type
    include_details: bool = False,  # include extra cols + ISO timestamps
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return actions directly connected to *action_id* via the *dependencies* table.

    • Uses a read-only snapshot ⇒ no WAL writes.
    • Sequence ordering preserved; timestamps decorated with *_iso when requested.
    """
    if not action_id:
        raise ToolInputError("Action ID required.", param_name="action_id")
    if direction not in ("downstream", "upstream"):
        raise ToolInputError(
            "Direction must be 'downstream' or 'upstream'.", param_name="direction"
        )

    t0 = time.perf_counter()
    db = DBConnection(db_path)

    # Helper: decorate ISO strings when details requested
    def _iso(obj: dict, keys: tuple[str, ...]) -> None:
        for k in keys:
            if (val := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(val)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── confirm root action exists ─────────
            exists = await conn.execute_fetchone(
                "SELECT 1 FROM actions WHERE action_id = ?", (action_id,)
            )
            if exists is None:
                raise ToolInputError(f"Action {action_id} not found.", param_name="action_id")

            # ───────── build query ─────────
            cols = [
                "a.action_id",
                "a.title",
                "dep.dependency_type",
            ]
            if include_details:
                cols += [
                    "a.action_type",
                    "a.status",
                    "a.started_at",
                    "a.completed_at",
                    "a.sequence_number",
                ]

            base = (
                "SELECT {cols} "
                "FROM dependencies dep "
                "JOIN actions a ON {join_cond} "
                "WHERE {where_cond}"
            )
            if direction == "downstream":
                join_cond = "dep.source_action_id = a.action_id"
                where_cond = "dep.target_action_id = ?"
            else:  # upstream
                join_cond = "dep.target_action_id = a.action_id"
                where_cond = "dep.source_action_id = ?"

            sql = base.format(
                cols=", ".join(cols),
                join_cond=join_cond,
                where_cond=where_cond,
            )
            params: list[Any] = [action_id]
            if dependency_type:
                sql += " AND dep.dependency_type = ?"
                params.append(dependency_type)
            sql += " ORDER BY a.sequence_number"

            # ───────── fetch ─────────
            related: list[dict] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    rec = dict(row)
                    if include_details:
                        _iso(rec, ("started_at", "completed_at"))
                    related.append(rec)

        return {
            "action_id": action_id,
            "direction": direction,
            "related_actions": related,
            "success": True,
            "processing_time": time.perf_counter() - t0,
        }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_action_dependencies failed", exc_info=True)
        raise ToolError(f"Failed to get action dependencies: {exc}") from exc


# --- 4. Artifact Tracking Tools ---


@with_tool_metrics
@with_error_handling
async def record_artifact(
    workflow_id: str,
    name: str,
    artifact_type: str,
    *,
    action_id: str | None = None,
    description: str | None = None,
    path: str | None = None,
    content: str | None = None,
    metadata: Dict[str, Any] | None = None,
    is_output: bool = False,
    tags: list[str] | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    # ... (validation logic remains the same) ...
    if not name:
        raise ToolInputError("Artifact name required", param_name="name")
    try:
        art_type_enum = ArtifactType(artifact_type.lower())
    except ValueError as err:
        raise ToolInputError(
            f"Invalid artifact_type '{artifact_type}'. Expected one of {[t.value for t in ArtifactType]}",
            param_name="artifact_type",
        ) from err

    started_perf_counter = time.perf_counter()
    now_ts = int(time.time())
    db = DBConnection(db_path)

    async def _fetch_existing_artifact_details(
        conn_fetch: aiosqlite.Connection, existing_artifact_id: str
    ) -> Dict[str, Any]:
        artifact_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (existing_artifact_id,)
        )
        if not artifact_row:
            raise ToolError(
                f"Failed to re-fetch existing artifact {existing_artifact_id} on idempotency hit."
            )

        art_data = dict(artifact_row)
        tag_rows = await conn_fetch.execute_fetchall(
            "SELECT t.name FROM tags t JOIN artifact_tags att ON att.tag_id = t.tag_id WHERE att.artifact_id = ?",
            (existing_artifact_id,),
        )
        art_data["tags"] = [r["name"] for r in tag_rows]
        art_data["metadata"] = await MemoryUtils.deserialize(art_data.get("metadata"))
        art_data["is_output"] = bool(art_data.get("is_output", False))  # Ensure boolean

        linked_mem_row = await conn_fetch.execute_fetchone(
            "SELECT memory_id FROM memories WHERE artifact_id = ? AND memory_type = ? ORDER BY created_at ASC LIMIT 1",
            (existing_artifact_id, MemoryType.ARTIFACT_CREATION.value),
        )
        linked_memory_id_existing = linked_mem_row["memory_id"] if linked_mem_row else None

        return {
            "artifact_id": existing_artifact_id,
            "linked_memory_id": linked_memory_id_existing,
            "workflow_id": art_data["workflow_id"],
            "name": art_data["name"],
            "artifact_type": art_data["artifact_type"],
            "path": art_data.get("path"),
            "created_at_unix": art_data["created_at"],
            "created_at_iso": to_iso_z(art_data["created_at"]),
            "content_stored_in_db": art_data.get("content") is not None,
            "content_truncated_in_db": False,  # Assume full content if fetched
            "is_output": art_data["is_output"],
            "tags": art_data["tags"],
            "idempotency_hit": True,
            "success": True,
            "processing_time": time.perf_counter() - started_perf_counter,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            # Artifacts are unique by (workflow_id, idempotency_key)
            existing_artifact_row = await conn_check.execute_fetchone(
                "SELECT artifact_id FROM artifacts WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_artifact_row:
            existing_artifact_id = existing_artifact_row["artifact_id"]
            logger.info(
                f"Idempotency hit for record_artifact (key='{idempotency_key}'). Returning existing artifact {_fmt_id(existing_artifact_id)}."
            )
            async with db.transaction(readonly=True) as conn_details:
                return await _fetch_existing_artifact_details(conn_details, existing_artifact_id)

    artifact_id_new = MemoryUtils.generate_id()
    linked_memory_id_new = MemoryUtils.generate_id()
    metadata_json = await MemoryUtils.serialize(metadata)
    db_content_to_store = content  # No truncation as per revised logic

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ... (existence checks for workflow_id, action_id remain the same) ...
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
        ):
            raise ToolInputError(
                f"Workflow {_fmt_id(workflow_id)} not found", param_name="workflow_id"
            )
        if action_id and not await conn.execute_fetchone(
            "SELECT 1 FROM actions WHERE action_id=? AND workflow_id=?", (action_id, workflow_id)
        ):
            raise ToolInputError(
                f"Action {_fmt_id(action_id)} does not belong to workflow {_fmt_id(workflow_id)}",
                param_name="action_id",
            )

        await conn.execute(
            """INSERT INTO artifacts (artifact_id, workflow_id, action_id, artifact_type, name, description, path, content, metadata, created_at, is_output, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",  # MODIFIED: Added idempotency_key
            (
                artifact_id_new,
                workflow_id,
                action_id,
                art_type_enum.value,
                name,
                description,
                path,
                db_content_to_store,
                metadata_json,
                now_ts,
                is_output,
                idempotency_key,
            ),  # MODIFIED: Added value
        )
        # ... (tag processing, workflow update, memory creation, logging remain the same, using artifact_id_new and linked_memory_id_new)
        tag_list_to_process = tags or []
        if tag_list_to_process:
            await MemoryUtils.process_tags(
                conn, artifact_id_new, tag_list_to_process, entity_type="artifact"
            )
        await conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now_ts, now_ts, workflow_id),
        )
        mem_content_parts = [f"Artifact '{name}' ({art_type_enum.value}) created"]
        if action_id:
            mem_content_parts.append(f"in action '{action_id[:8]}…'")
        if description:
            mem_content_parts.append(f"Description: {description[:100]}…")
        if path:
            mem_content_parts.append(f"External path provided: {path}")
        if db_content_to_store is not None:
            mem_content_parts.append(
                f"Content (size: {len(db_content_to_store.encode('utf-8'))} bytes) stored directly in DB."
            )
        else:
            mem_content_parts.append("No direct content stored in DB (may be external via path).")
        if is_output:
            mem_content_parts.append("Marked as workflow output.")
        mem_content = ". ".join(mem_content_parts) + "."
        await conn.execute(
            """INSERT INTO memories (memory_id, workflow_id, action_id, artifact_id, content, memory_level, memory_type, importance, confidence, tags, created_at, updated_at, access_count, last_accessed)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
            (
                linked_memory_id_new,
                workflow_id,
                action_id,
                artifact_id_new,
                mem_content,
                MemoryLevel.EPISODIC.value,
                MemoryType.ARTIFACT_CREATION.value,
                6.0 if is_output else 5.0,
                1.0,
                json.dumps(list(set(["artifact_creation", art_type_enum.value] + (tags or [])))),
                now_ts,
                now_ts,
                0,
            ),
        )
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_artifact",
            None,
            action_id,
            {
                "artifact_id": artifact_id_new,
                "name": name,
                "type": art_type_enum.value,
                "linked_memory_id": linked_memory_id_new,
                "content_stored_directly": (db_content_to_store is not None),
            },
        )
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_from_artifact",
            linked_memory_id_new,
            action_id,
            {"artifact_id": artifact_id_new, "reason": "Artifact recording"},
        )

    elapsed_processing_time = time.perf_counter() - started_perf_counter
    logger.info(
        f"Artifact '{name}' ({_fmt_id(artifact_id_new)}, type: {art_type_enum.value}) recorded. Linked memory: {_fmt_id(linked_memory_id_new)}.",
        emoji_key="page_facing_up",
        time=elapsed_processing_time,
    )
    return {
        "artifact_id": artifact_id_new,
        "linked_memory_id": linked_memory_id_new,
        "workflow_id": workflow_id,
        "name": name,
        "artifact_type": art_type_enum.value,
        "path": path,
        "created_at_unix": now_ts,
        "created_at_iso": to_iso_z(now_ts),
        "content_stored_in_db": bool(db_content_to_store is not None),
        "content_truncated_in_db": False,
        "is_output": is_output,
        "tags": tags or [],
        "idempotency_hit": False,  # NEW
        "success": True,
        "processing_time": elapsed_processing_time,
    }


# --- 5. Thought & Reasoning Tools ---
@with_tool_metrics
@with_error_handling
async def record_thought(
    workflow_id: str,
    content: str,
    *,
    thought_type: str = "inference",
    thought_chain_id: str | None = None,  # If None, will use/create primary chain for workflow_id
    parent_thought_id: str | None = None,
    relevant_action_id: str | None = None,
    relevant_artifact_id: str | None = None,
    relevant_memory_id: str | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
    conn: Optional[Any] = None,  # Allow passing external connection
) -> Dict[str, Any]:
    if not content or not isinstance(content, str):
        raise ToolInputError("Thought content must be a non-empty string", param_name="content")
    try:
        thought_type_enum = ThoughtType(thought_type.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid thought_type '{thought_type}'. Must be one of: {', '.join(t.value for t in ThoughtType)}",
            param_name="thought_type",
        ) from exc

    now_unix = int(time.time())
    t0_perf = time.perf_counter()  # For processing_time if new thought is created
    final_memory_id_for_return: str | None = None  # For the linked memory if created

    # This helper will be used if an idempotency hit occurs for a thought
    async def _fetch_existing_thought_details(
        db_conn_fetch: aiosqlite.Connection,
        existing_thought_id: str,
        chain_id_of_existing_thought: str,
    ) -> Dict[str, Any]:
        thought_row = await db_conn_fetch.execute_fetchone(
            "SELECT * FROM thoughts WHERE thought_id = ?", (existing_thought_id,)
        )
        if not thought_row:
            raise ToolError(
                f"Failed to re-fetch existing thought {existing_thought_id} on idempotency hit."
            )

        thought_data = dict(thought_row)
        # Check if a memory was linked to this existing thought
        linked_mem_row = await db_conn_fetch.execute_fetchone(
            "SELECT memory_id FROM memories WHERE thought_id = ? AND memory_type = ? ORDER BY created_at ASC LIMIT 1",
            (
                existing_thought_id,
                MemoryType.REASONING_STEP.value,
            ),  # Assuming this is the type used for thought-linked memories
        )
        existing_linked_memory_id = linked_mem_row["memory_id"] if linked_mem_row else None

        return {
            "thought_id": existing_thought_id,
            "thought_chain_id": chain_id_of_existing_thought,  # Use the chain_id it belongs to
            "thought_type": thought_data["thought_type"],
            "content": thought_data["content"],  # Return existing content
            "sequence_number": thought_data["sequence_number"],
            "created_at": to_iso_z(thought_data["created_at"]),
            "linked_memory_id": existing_linked_memory_id,
            "idempotency_hit": True,
            "success": True,
            "processing_time": time.perf_counter() - t0_perf,  # Time for this get operation
        }

    async def _tx(
        db_conn: aiosqlite.Connection,
    ) -> tuple[str, int, Optional[str]]:  # Returns chain_id, seq_num, linked_memory_id
        # ... (FK existence checks remain the same) ...
        async def _exists(sql: str, param: Optional[str], pname: str) -> None:
            if not param:
                return
            row = await db_conn.execute_fetchone(sql, (param,))
            if row is None:
                raise ToolInputError(
                    f"{pname.replace('_', ' ').capitalize()} not found: {param}", param_name=pname
                )

        await _exists("SELECT 1 FROM workflows WHERE workflow_id = ?", workflow_id, "workflow_id")
        await _exists(
            "SELECT 1 FROM thoughts WHERE thought_id = ?", parent_thought_id, "parent_thought_id"
        )
        await _exists(
            "SELECT 1 FROM actions WHERE action_id = ?", relevant_action_id, "relevant_action_id"
        )
        await _exists(
            "SELECT 1 FROM artifacts WHERE artifact_id = ?",
            relevant_artifact_id,
            "relevant_artifact_id",
        )
        await _exists(
            "SELECT 1 FROM memories WHERE memory_id = ?", relevant_memory_id, "relevant_memory_id"
        )

        # Resolve target_chain_id (this logic is complex due to auto-creation if conn is None)
        tgt_chain_id_resolved: Optional[str] = thought_chain_id
        if not tgt_chain_id_resolved:  # If no specific chain is provided
            primary_chain_row = await db_conn.execute_fetchone(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id=? ORDER BY created_at LIMIT 1",
                (workflow_id,),
            )
            if primary_chain_row:
                tgt_chain_id_resolved = primary_chain_row["thought_chain_id"]
            elif conn is None:  # Managing transaction locally, can create primary chain
                tgt_chain_id_resolved = MemoryUtils.generate_id()
                await db_conn.execute(
                    "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at) VALUES (?, ?, 'Main reasoning', ?)",
                    (tgt_chain_id_resolved, workflow_id, now_unix),
                )
            else:  # External transaction, cannot auto-create primary chain
                raise ToolError(
                    f"Primary thought chain for workflow {workflow_id} not found; cannot auto-create in external transaction."
                )
        else:  # Explicit chain_id provided, validate it
            chain_valid_row = await db_conn.execute_fetchone(
                "SELECT 1 FROM thought_chains WHERE thought_chain_id=? AND workflow_id=?",
                (tgt_chain_id_resolved, workflow_id),
            )
            if not chain_valid_row:
                raise ToolInputError(
                    f"Provided thought_chain_id {tgt_chain_id_resolved} not found or mismatched workflow.",
                    param_name="thought_chain_id",
                )

        if not tgt_chain_id_resolved:  # Should be caught by above logic
            raise ToolError("Could not determine target thought_chain_id for recording thought.")

        # NEW: Idempotency check for thoughts (scoped to thought_chain_id)
        if idempotency_key:
            existing_thought_row = await db_conn.execute_fetchone(
                "SELECT thought_id FROM thoughts WHERE thought_chain_id = ? AND idempotency_key = ?",
                (tgt_chain_id_resolved, idempotency_key),
            )
            if existing_thought_row:
                # If hit, the calling function will handle fetching full details
                # This inner _tx will signal by returning a specific structure or raising a specific error
                # For now, let's make it return the ID, the main function will re-fetch
                # This changes what _tx returns, so the main function needs to adapt
                raise ToolError(
                    f"IDEMPOTENCY_HIT:{existing_thought_row['thought_id']}:{tgt_chain_id_resolved}"
                )  # Special error format

        # No idempotency hit or no key, proceed to create new thought
        new_thought_id = (
            MemoryUtils.generate_id()
        )  # Always generate a new ID for the *potential* new thought
        seq_num = await MemoryUtils.get_next_sequence_number(
            db_conn, tgt_chain_id_resolved, "thoughts", "thought_chain_id"
        )

        await db_conn.execute(
            """INSERT INTO thoughts (thought_id, thought_chain_id, parent_thought_id, thought_type, content, sequence_number, created_at, relevant_action_id, relevant_artifact_id, relevant_memory_id, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",  # MODIFIED
            (
                new_thought_id,
                tgt_chain_id_resolved,
                parent_thought_id,
                thought_type_enum.value,
                content,
                seq_num,
                now_unix,
                relevant_action_id,
                relevant_artifact_id,
                relevant_memory_id,
                idempotency_key,
            ),  # MODIFIED
        )
        await db_conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now_unix, now_unix, workflow_id),
        )

        # Optional memory promotion (logic remains, using new_thought_id)
        linked_mem_id: Optional[str] = None
        important_types = {
            ThoughtType.GOAL,
            ThoughtType.DECISION,
            ThoughtType.SUMMARY,
            ThoughtType.REFLECTION,
            ThoughtType.HYPOTHESIS,
            ThoughtType.INSIGHT,
        }
        if thought_type_enum in important_types:
            linked_mem_id = MemoryUtils.generate_id()
            await db_conn.execute(
                """INSERT INTO memories (memory_id, workflow_id, thought_id, content, memory_level, memory_type, importance, confidence, tags, created_at, updated_at, access_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,0)""",
                (
                    linked_mem_id,
                    workflow_id,
                    new_thought_id,
                    f"Thought [{seq_num}] ({thought_type_enum.value.capitalize()}): {content}",
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.REASONING_STEP.value,
                    7.5 if thought_type_enum in {ThoughtType.GOAL, ThoughtType.DECISION} else 6.5,
                    1.0,
                    json.dumps(["reasoning", thought_type_enum.value]),
                    now_unix,
                    now_unix,
                ),
            )
            await MemoryUtils._log_memory_operation(
                db_conn,
                workflow_id,
                "create_from_thought",
                linked_mem_id,
                None,
                {"thought_id": new_thought_id},
            )

        # Return new_thought_id along with other details
        return tgt_chain_id_resolved, seq_num, linked_mem_id, new_thought_id

    manage_locally = conn is None
    final_thought_id_to_return: str
    final_chain_id_to_return: str
    final_seq_num_to_return: int
    idempotency_was_hit = False

    # This outer try-except handles the special "IDEMPOTENCY_HIT" error from _tx
    try:
        if manage_locally:
            db_main = DBConnection(db_path)
            async with db_main.transaction(mode="IMMEDIATE") as local_tx_conn:
                chain_id_res, seq_res, mem_id_res, thought_id_res = await _tx(local_tx_conn)
        else:
            if not isinstance(conn, aiosqlite.Connection):
                raise ToolError("Parameter 'conn' is not valid Connection")
            chain_id_res, seq_res, mem_id_res, thought_id_res = await _tx(conn)

        final_thought_id_to_return = thought_id_res
        final_chain_id_to_return = chain_id_res
        final_seq_num_to_return = seq_res
        final_memory_id_for_return = mem_id_res  # From the new thought creation
        idempotency_was_hit = False

    except ToolError as te:
        if str(te).startswith("IDEMPOTENCY_HIT:"):
            parts = str(te).split(":")
            existing_thought_id = parts[1]
            chain_id_of_existing = parts[2]
            logger.info(
                f"Idempotency hit for record_thought (key='{idempotency_key}'). Returning existing thought {_fmt_id(existing_thought_id)} from chain {_fmt_id(chain_id_of_existing)}."
            )
            idempotency_was_hit = True
            # Fetch full details of the existing thought using a new read-only transaction
            db_fetch = DBConnection(
                db_path
            )  # Need a new DBConnection instance for this fetch if manage_locally was true
            async with db_fetch.transaction(readonly=True) as conn_details:
                # We need to get the thought_id for the final return, and also linked_memory_id
                # _fetch_existing_thought_details already does this, but it needs the sequence number
                # and content which are not in the "IDEMPOTENCY_HIT" error.
                # Simpler: The _fetch_existing_thought_details is self-contained.
                # We just need to call it here.
                fetched_details = await _fetch_existing_thought_details(
                    conn_details, existing_thought_id, chain_id_of_existing
                )
                # Now extract values from fetched_details for the final return
                final_thought_id_to_return = fetched_details["thought_id"]
                final_chain_id_to_return = fetched_details["thought_chain_id"]
                final_seq_num_to_return = fetched_details["sequence_number"]
                # Note: created_at is already ISO in fetched_details, content also from there.
                # The 'content' in the return dict should be the existing content.
                content = fetched_details["content"]
                final_memory_id_for_return = fetched_details["linked_memory_id"]
        else:
            raise  # Re-raise other ToolErrors

    # Ensure all required final variables are set
    if (
        "final_thought_id_to_return" not in locals()
        or "final_chain_id_to_return" not in locals()
        or "final_seq_num_to_return" not in locals()
    ):
        raise ToolError(
            "Internal error: Failed to determine final thought details after idempotency check."
        )

    return {
        "thought_id": final_thought_id_to_return,
        "thought_chain_id": final_chain_id_to_return,
        "thought_type": thought_type_enum.value,  # This is from the original call's intent
        "content": content,  # This should be the content of the (new or existing) thought
        "sequence_number": final_seq_num_to_return,
        "created_at": to_iso_z(now_unix)
        if not idempotency_was_hit
        else to_iso_z(time.time()),  # If hit, created_at is 'now' for the get
        "linked_memory_id": final_memory_id_for_return,
        "idempotency_hit": idempotency_was_hit,  # NEW
        "success": True,
        "processing_time": time.perf_counter() - t0_perf,
    }


# --- 6. Core Memory Tools ---
@with_tool_metrics
@with_error_handling
async def store_memory(
    workflow_id: str,
    content: str,
    memory_type: str,
    *,
    memory_level: str = MemoryLevel.EPISODIC.value,
    importance: float = 5.0,
    confidence: float = 1.0,
    description: str | None = None,
    reasoning: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    ttl: int | None = None,
    context_data: dict[str, Any] | None = None,
    generate_embedding: bool = True,
    suggest_links: bool = True,
    link_suggestion_threshold: float = agent_memory_config.similarity_threshold,
    max_suggested_links: int = 3,
    action_id: str | None = None,
    thought_id: str | None = None,
    artifact_id: str | None = None,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    # ... (validation logic remains the same) ...
    if not content:
        raise ToolInputError("Content cannot be empty.", param_name="content")
    try:
        mem_type_enum = MemoryType(memory_type.lower())
    except ValueError as e:
        raise ToolInputError(
            f"Invalid memory_type. Use one of: {', '.join(mt.value for mt in MemoryType)}",
            param_name="memory_type",
        ) from e
    try:
        mem_level_enum = MemoryLevel(memory_level.lower())
    except ValueError as e:
        raise ToolInputError(
            f"Invalid memory_level. Use one of: {', '.join(ml.value for ml in MemoryLevel)}",
            param_name="memory_level",
        ) from e
    if not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0–10.0.", param_name="importance")
    # ... (other validations)

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_memory_details(
        conn_fetch: aiosqlite.Connection, existing_memory_id: str, wf_id: str
    ) -> Dict[str, Any]:
        # This reuses parts of get_memory_by_id logic, simplified
        mem_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM memories WHERE memory_id = ?", (existing_memory_id,)
        )
        if not mem_row:
            raise ToolError(
                f"Failed to re-fetch existing memory {existing_memory_id} on idempotency hit."
            )

        mem_data = dict(mem_row)
        mem_data["tags"] = await MemoryUtils.deserialize(mem_data.get("tags"))
        # Fetch embedding_id if it exists, but don't regenerate embedding on hit
        # Suggested links are not regenerated on hit to avoid re-computation

        return {
            "success": True,
            "idempotency_hit": True,
            "memory_id": existing_memory_id,
            "workflow_id": wf_id,
            "memory_level": mem_data["memory_level"],
            "memory_type": mem_data["memory_type"],
            "content_preview": (
                mem_data["content"][:100] + "…"
                if len(mem_data["content"]) > 100
                else mem_data["content"]
            ),
            "importance": mem_data["importance"],
            "confidence": mem_data["confidence"],
            "created_at": to_iso_z(mem_data["created_at"]),
            "tags": mem_data["tags"],
            "embedding_id": mem_data.get("embedding_id"),
            "linked_action_id": mem_data.get("action_id"),
            "linked_thought_id": mem_data.get("thought_id"),
            "linked_artifact_id": mem_data.get("artifact_id"),
            "suggested_links": [],  # Don't re-suggest on hit
            "processing_time": time.perf_counter() - t0_perf,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_mem_row = await conn_check.execute_fetchone(
                "SELECT memory_id FROM memories WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_mem_row:
            existing_memory_id = existing_mem_row["memory_id"]
            logger.info(
                f"Idempotency hit for store_memory (key='{idempotency_key}'). Returning existing memory {_fmt_id(existing_memory_id)}."
            )
            async with db.transaction(readonly=True) as conn_details:  # Read-only for fetching
                # Since an idempotency hit implies the memory and its embedding (if any) were already processed,
                # we just return its identifier and core metadata.
                # For a more complete return matching a new store, we'd need to fetch links etc.
                # For now, a simpler payload for idempotency hit.
                return await _fetch_existing_memory_details(
                    conn_details, existing_memory_id, workflow_id
                )

    # No idempotency hit or no key, proceed to create new memory
    memory_id_new = MemoryUtils.generate_id()
    # ... (tag normalization, TTL logic remains the same) ...
    base_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
    final_tags = list({*base_tags, mem_type_enum.value, mem_level_enum.value})
    final_tags_json = json.dumps(final_tags)
    if ttl is None:
        ttl = {
            MemoryLevel.WORKING: agent_memory_config.ttl_working,
            MemoryLevel.EPISODIC: agent_memory_config.ttl_episodic,
        }.get(mem_level_enum, 0)
    else:
        ttl = int(ttl)

    embed_id: str | None = None
    suggested_links_new: list[dict[str, Any]] = []

    async with db.transaction(mode="IMMEDIATE") as conn:
        # 1️⃣  FK existence checks
        async def _exists(tbl: str, key: str) -> bool:
            row = await conn.execute_fetchone(
                f"SELECT 1 FROM {tbl} WHERE {tbl[:-1]}_id = ?", (key,)
            )
            return row is not None

        async def _exists(tbl: str, key_val: Optional[str]) -> bool:  # key_val can be None
            if not key_val:
                return True  # If ID not provided, it's not a constraint violation
            row = await conn.execute_fetchone(
                f"SELECT 1 FROM {tbl} WHERE {tbl[:-1]}_id = ? AND workflow_id = ?",
                (key_val, workflow_id),
            )  # Ensure scope
            return row is not None

        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
        ):
            raise ToolInputError("Workflow not found.", param_name="workflow_id")
        if action_id and not await _exists("actions", action_id):
            raise ToolInputError(
                f"Action {action_id} not found in workflow {workflow_id}.", param_name="action_id"
            )
        if thought_id and not await _exists("thoughts", thought_id):
            raise ToolInputError(
                f"Thought {thought_id} not found in workflow {workflow_id}.",
                param_name="thought_id",
            )
        if artifact_id and not await _exists("artifacts", artifact_id):
            raise ToolInputError(
                f"Artifact {artifact_id} not found in workflow {workflow_id}.",
                param_name="artifact_id",
            )

        # 2️⃣  INSERT memory
        await conn.execute(
            """INSERT INTO memories (memory_id, workflow_id, content, memory_level, memory_type, importance, confidence, description, reasoning, source, context, tags, created_at, updated_at, last_accessed, access_count, ttl, action_id, thought_id, artifact_id, embedding_id, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",  # MODIFIED: Added idempotency_key
            (
                memory_id_new,
                workflow_id,
                content,
                mem_level_enum.value,
                mem_type_enum.value,
                importance,
                confidence,
                description or "",
                reasoning or "",
                source or "",
                await MemoryUtils.serialize(context_data) if context_data else "{}",
                final_tags_json,
                now_unix,
                now_unix,
                None,
                0,
                ttl,
                action_id,
                thought_id,
                artifact_id,
                idempotency_key,
            ),  # MODIFIED: Added value
        )

        # 3️⃣  Embedding
        if generate_embedding:
            try:
                embed_id = await _store_embedding(
                    conn, memory_id_new, f"{description}: {content}" if description else content
                )
            except Exception as e_embed:
                logger.error(f"Embedding failed for {memory_id_new}: {e_embed}", exc_info=True)

        # 4️⃣  Link suggestions
        if suggest_links and embed_id and max_suggested_links:
            try:
                sims = await _find_similar_memories(
                    conn=conn,
                    query_text=content,
                    workflow_id=workflow_id,
                    limit=max_suggested_links + 1,
                    threshold=link_suggestion_threshold,
                )
                target_ids = [mid for mid, _ in sims if mid != memory_id_new][:max_suggested_links]
                if target_ids:
                    ph = ",".join("?" * len(target_ids))
                    rows = await conn.execute_fetchall(
                        f"SELECT memory_id, description, memory_type FROM memories WHERE memory_id IN ({ph})",
                        target_ids,
                    )
                    score_map = dict(sims)
                    for row in rows:
                        m_id = row["memory_id"]
                        sim = round(score_map.get(m_id, 0.0), 4)
                        tgt_type = row["memory_type"]
                        link_type = LinkType.RELATED.value
                        if (
                            mem_type_enum.value == tgt_type
                            and mem_level_enum == MemoryLevel.EPISODIC
                        ):
                            link_type = LinkType.SEQUENTIAL.value
                        elif (
                            mem_type_enum == MemoryType.INSIGHT
                            and tgt_type == MemoryType.FACT.value
                        ):
                            link_type = LinkType.GENERALIZES.value
                        suggested_links_new.append(
                            {
                                "target_memory_id": m_id,
                                "target_description": row["description"],
                                "target_type": tgt_type,
                                "similarity": sim,
                                "suggested_link_type": link_type,
                            }
                        )
            except Exception as e_link:
                logger.error(
                    f"Link-suggestion failure for {memory_id_new}: {e_link}", exc_info=True
                )

        # 5️⃣  Touch workflow
        await conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now_unix, now_unix, workflow_id),
        )

        # 6️⃣  Operation log
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create",
            memory_id_new,
            action_id,
            {
                "memory_level": mem_level_enum.value,
                "memory_type": mem_type_enum.value,
                "importance": importance,
                "embedding_generated": bool(embed_id),
                "links_suggested": len(suggested_links_new),
                "tags": final_tags,
            },
        )

    # 7️⃣  Response
    elapsed_time = time.perf_counter() - t0_perf
    logger.info(
        f"Memory {memory_id_new} stored; {len(suggested_links_new)} links suggested.",
        emoji_key="floppy_disk",
        time=elapsed_time,
    )
    return {
        "success": True,
        "idempotency_hit": False,  # NEW
        "memory_id": memory_id_new,
        "workflow_id": workflow_id,
        "memory_level": mem_level_enum.value,
        "memory_type": mem_type_enum.value,
        "content_preview": content[:100] + ("…" if len(content) > 100 else ""),
        "importance": importance,
        "confidence": confidence,
        "created_at": to_iso_z(now_unix),
        "tags": final_tags,
        "embedding_id": embed_id,
        "linked_action_id": action_id,
        "linked_thought_id": thought_id,
        "linked_artifact_id": artifact_id,
        "suggested_links": suggested_links_new,
        "processing_time": elapsed_time,
    }


@with_tool_metrics
@with_error_handling
async def get_memory_by_id(
    memory_id: str,
    *,
    include_links: bool = True,
    include_context: bool = True,
    context_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single memory row and its optional graph / semantic context.

    • Access statistics are updated inside the R/W transaction.
    • TTL-expired rows are deleted atomically and reported as errors.
    • All integer timestamps are preserved; ISO strings are appended as *_iso.
    • Conditionally triggers a WAL checkpoint after successful transaction.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    t0 = time.time()
    db = DBConnection(db_path)
    # This flag will track if any actual database modification occurred (write or delete)
    db_modified_in_transaction = False

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction() as conn:  # R/W IMMEDIATE txn
            mem_row = await conn.execute_fetchone(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            )
            if mem_row is None:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

            mem: Dict[str, Any] = dict(mem_row)

            ttl = mem.get("ttl", 0)
            if ttl and mem["created_at"] + ttl <= int(time.time()):
                logger.warning(f"Memory {memory_id} expired; deleting.", emoji_key="wastebasket")
                await conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                db_modified_in_transaction = True  # DB was modified
                raise ToolError(f"Memory {memory_id} has expired and was deleted.")

            mem["tags"] = await MemoryUtils.deserialize(mem.get("tags"))
            mem["context"] = await MemoryUtils.deserialize(mem.get("context"))

            # These operations modify the database:
            await MemoryUtils._update_memory_access(conn, memory_id)
            await MemoryUtils._log_memory_operation(
                conn, mem["workflow_id"], "access_by_id", memory_id
            )
            db_modified_in_transaction = True  # DB was modified

            if include_links:
                mem["outgoing_links"], mem["incoming_links"] = [], []
                # Using execute_fetchall as these are typically small, bounded queries for a single memory_id
                outgoing_rows = await conn.execute_fetchall(
                    """
                    SELECT ml.*, m.description AS target_description,
                           m.memory_type AS target_type
                    FROM memory_links ml
                    JOIN memories m ON m.memory_id = ml.target_memory_id
                    WHERE ml.source_memory_id = ?
                    """,
                    (memory_id,),
                )
                for r in outgoing_rows:
                    link = dict(r)
                    _add_iso(link, ["created_at"])  # Keep original int, add iso
                    # No need for created_at_unix if _add_iso handles it
                    mem["outgoing_links"].append(
                        {
                            "link_id": link["link_id"],
                            "target_memory_id": link["target_memory_id"],
                            "target_description": link["target_description"],
                            "target_type": link["target_type"],
                            "link_type": link["link_type"],
                            "strength": link["strength"],
                            "description": link["description"],
                            "created_at": link["created_at_iso"],  # Use ISO formatted
                        }
                    )

                incoming_rows = await conn.execute_fetchall(
                    """
                    SELECT ml.*, m.description AS source_description,
                           m.memory_type AS source_type
                    FROM memory_links ml
                    JOIN memories m ON m.memory_id = ml.source_memory_id
                    WHERE ml.target_memory_id = ?
                    """,
                    (memory_id,),
                )
                for r in incoming_rows:
                    link = dict(r)
                    _add_iso(link, ["created_at"])
                    mem["incoming_links"].append(
                        {
                            "link_id": link["link_id"],
                            "source_memory_id": link["source_memory_id"],
                            "source_description": link["source_description"],
                            "source_type": link["source_type"],
                            "link_type": link["link_type"],
                            "strength": link["strength"],
                            "description": link["description"],
                            "created_at": link["created_at_iso"],
                        }
                    )
            else:
                mem["outgoing_links"] = mem["incoming_links"] = []

            mem["semantic_context"] = []
            if include_context and mem.get("embedding_id"):
                query = (mem.get("description", "") or "") + ": " + (mem.get("content", "") or "")
                if query.strip():
                    try:
                        sims = await _find_similar_memories(
                            conn=conn,
                            query_text=query,
                            workflow_id=mem["workflow_id"],
                            limit=context_limit
                            + 1,  # Fetch one extra to ensure we can exclude self
                            threshold=agent_memory_config.similarity_threshold * 0.9,
                        )
                        if sims:
                            ids = [i for i, _ in sims if i != memory_id][:context_limit]
                            if ids:
                                ph = ",".join("?" * len(ids))
                                ctx_rows = await conn.execute_fetchall(
                                    f"""
                                    SELECT memory_id, description, memory_type, importance
                                    FROM memories WHERE memory_id IN ({ph})
                                    """,
                                    ids,
                                )
                                score_map = dict(sims)  # Renamed for clarity
                                for r_ctx in sorted(  # Renamed r to r_ctx
                                    ctx_rows,
                                    key=lambda x: score_map.get(
                                        x["memory_id"], 0.0
                                    ),  # Use .get() for safety
                                    reverse=True,
                                ):
                                    mem["semantic_context"].append(
                                        {
                                            "memory_id": r_ctx["memory_id"],
                                            "description": r_ctx["description"],
                                            "memory_type": r_ctx["memory_type"],
                                            "importance": r_ctx["importance"],
                                            "similarity": round(
                                                score_map.get(r_ctx["memory_id"], 0.0), 4
                                            ),  # Use .get()
                                        }
                                    )
                    except Exception as err:
                        logger.warning(
                            f"Semantic context lookup failed for memory {_fmt_id(memory_id)}: {err}",
                            exc_info=True,
                        )

            _add_iso(mem, ["created_at", "updated_at", "last_accessed"])
            mem["success"] = True
 
        logger.debug(
            f"Skipping WAL checkpoint for get_memory_by_id ({_fmt_id(memory_id)}) as no DB modifications occurred in transaction."
        )

        mem["processing_time"] = time.time() - t0
        logger.info(
            f"Memory {_fmt_id(memory_id)} retrieved (links={include_links}, ctx={include_context}) in {mem['processing_time']:.3f}s",
            emoji_key="inbox_tray",
        )
        return mem

    except ToolError as te:  # Catch ToolError if memory expired and was deleted
        # Log as info because it's an expected outcome for expired memory
        logger.info(f"get_memory_by_id({_fmt_id(memory_id)}): {te}")
        # Re-raise to signal failure to the caller as the memory is gone
        raise
    except ToolInputError:
        raise  # Let the decorator handle this
    except Exception as exc:
        logger.error(f"get_memory_by_id({_fmt_id(memory_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get memory {memory_id}: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def search_semantic_memories(
    query: str,
    *,
    workflow_id: str | None = None,
    limit: int = 5,
    threshold: float = agent_memory_config.similarity_threshold,
    memory_level: str | None = None,
    memory_type: str | None = None,
    include_content: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Run an embedding-based semantic search across memories.

    Side-effects
    ------------
    • bumps `access_count`
    • refreshes `last_accessed`
    • logs the access in `memory_operations`
    """
    # ───────────── validation ─────────────
    if not query:
        raise ToolInputError("Search query required.", param_name="query")
    if limit < 1:
        raise ToolInputError("Limit must be positive integer.", param_name="limit")
    if not 0.0 <= threshold <= 1.0:
        raise ToolInputError("Threshold must be 0.0-1.0.", param_name="threshold")
    if memory_level:
        MemoryLevel(memory_level.lower())  # raises ValueError → ToolInputError upstream
    if memory_type:
        MemoryType(memory_type.lower())

    start = time.time()
    results: list[dict[str, Any]] = []

    db = DBConnection(db_path)

    # one R/W transaction – grab a write lock early so read/updates remain atomic
    async with db.transaction(mode="IMMEDIATE") as conn:
        # ───────────── optional workflow check ─────────────
        if workflow_id:
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        # ───────────── similarity search (UDF-backed helper) ─────────────
        similar: list[tuple[str, float]] = await _find_similar_memories(
            conn=conn,
            query_text=query,
            workflow_id=workflow_id,
            limit=limit,
            threshold=threshold,
            memory_level=memory_level,
            memory_type=memory_type,
        )

        if not similar:
            processing = time.time() - start
            logger.info("Semantic search returned 0 rows.", emoji_key="zzz", time=processing)
            return {
                "memories": [],
                "query": query,
                "workflow_id": workflow_id,
                "success": True,
                "processing_time": processing,
            }

        # ───────────── hydrate memory rows ─────────────
        ids = [mid for mid, _ in similar]
        score_map = dict(similar)
        placeholders = ",".join("?" * len(ids))
        columns = (
            "memory_id, workflow_id, description, memory_type, memory_level, importance, "
            "confidence, created_at, updated_at, last_accessed, access_count, ttl, tags, "
            "action_id, thought_id, artifact_id"
        )
        if include_content:
            columns += ", content"

        rows = await conn.execute_fetchall(
            f"SELECT {columns} FROM memories WHERE memory_id IN ({placeholders})", ids
        )

        # keep original ordering by similarity
        for row in sorted(rows, key=lambda r: score_map[r["memory_id"]], reverse=True):
            m = dict(row)
            m["similarity"] = round(score_map[m["memory_id"]], 4)
            m["tags"] = await MemoryUtils.deserialize(m.get("tags"))
            results.append(m)

        # ───────────── batch touch / log ─────────────
        now_unix = int(time.time())
        upd_params = [(now_unix, mid) for mid in ids]
        log_params = [
            (
                MemoryUtils.generate_id(),
                row["workflow_id"],
                row["memory_id"],
                None,  # action_id
                "semantic_access",
                json.dumps({"query": query[:100], "score": score_map[row["memory_id"]]}),
                now_unix,
            )
            for row in rows
        ]

        await conn.executemany(
            "UPDATE memories SET last_accessed=?, "
            "access_count=COALESCE(access_count,0)+1 WHERE memory_id=?",
            upd_params,
        )
        await conn.executemany(
            "INSERT INTO memory_operations "
            "(operation_log_id, workflow_id, memory_id, action_id, operation, operation_data, timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            log_params,
        )

    # ───────────── decorate timestamps ─────────────
    def _add_iso(d: dict[str, Any], key: str) -> None:
        if (ts := d.get(key)) is not None:
            d[f"{key}_iso"] = safe_format_timestamp(ts)

    for m in results:
        for k in ("created_at", "updated_at", "last_accessed"):
            _add_iso(m, k)

    processing = time.time() - start
    logger.info(
        f"Semantic search → {len(results)} memories for '{query[:50]}…'",
        emoji_key="mag",
        time=processing,
    )

    return {
        "memories": results,
        "query": query,
        "workflow_id": workflow_id,
        "success": True,
        "processing_time": processing,
    }


@with_tool_metrics
@with_error_handling
async def hybrid_search_memories(
    query: str,
    *,
    workflow_id: str | None = None,
    limit: int = 10,
    offset: int = 0,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    memory_level: str | None = None,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    min_importance: float | None = None,
    max_importance: float | None = None,
    min_confidence: float | None = None,
    min_created_at_unix: int | None = None,
    max_created_at_unix: int | None = None,
    include_content: bool = True,
    include_links: bool = False,
    link_direction: str = "outgoing",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Hybrid (semantic + keyword/FTS) memory search with rich filtering.
    Returns ranked memories, preserves raw timestamps, adds *_iso strings.
    """
    t0 = time.time()

    # ───────── validation ─────────
    if not query:
        raise ToolInputError("Query string cannot be empty.", param_name="query")
    if not 0 <= semantic_weight <= 1:
        raise ToolInputError("semantic_weight 0-1", param_name="semantic_weight")
    if not 0 <= keyword_weight <= 1:
        raise ToolInputError("keyword_weight 0-1", param_name="keyword_weight")
    if semantic_weight + keyword_weight == 0:
        raise ToolInputError("At least one weight must be >0", param_name="semantic_weight")
    if limit < 1:
        raise ToolInputError("limit ≥1", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset ≥0", param_name="offset")
    if memory_level:
        MemoryLevel(memory_level.lower())  # raises if invalid
    if memory_type:
        MemoryType(memory_type.lower())
    if (ld := link_direction.lower()) not in {"outgoing", "incoming", "both"}:
        raise ToolInputError("link_direction invalid", param_name="link_direction")

    # weight normalisation
    w_sum = semantic_weight + keyword_weight
    w_sem = semantic_weight / w_sum
    w_kw = keyword_weight / w_sum

    db = DBConnection(db_path)

    # ranking maps
    score_map: dict[str, dict[str, float]] = defaultdict(
        lambda: {"semantic": 0.0, "keyword": 0.0, "hybrid": 0.0}
    )

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ───────── semantic candidates ─────────
        if w_sem:
            try:
                sem_limit = min(max(limit * 10, 100), agent_memory_config.max_semantic_candidates)
                sem_results = await _find_similar_memories(
                    conn=conn,
                    query_text=query,
                    workflow_id=workflow_id,
                    limit=sem_limit,
                    threshold=0.1,
                    memory_level=memory_level,
                    memory_type=memory_type,
                )  # → List[(memory_id, sim_score)]
                for mem_id, score in sem_results:
                    score_map[mem_id]["semantic"] = score
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}", exc_info=True)

        # ───────── keyword candidates (FTS) ─────────
        if w_kw:
            wh, prm = ["1=1"], []  # WHERE fragments / params
            joins = ""
            if workflow_id:
                wh += ["m.workflow_id=?"]
                prm += [workflow_id]
            if memory_level:
                wh += ["m.memory_level=?"]
                prm += [memory_level.lower()]
            if memory_type:
                wh += ["m.memory_type=?"]
                prm += [memory_type.lower()]
            if min_importance is not None:
                wh += ["m.importance>=?"]
                prm += [min_importance]
            if max_importance is not None:
                wh += ["m.importance<=?"]
                prm += [max_importance]
            if min_confidence is not None:
                wh += ["m.confidence>=?"]
                prm += [min_confidence]
            if min_created_at_unix is not None:
                wh += ["m.created_at>=?"]
                prm += [min_created_at_unix]
            if max_created_at_unix is not None:
                wh += ["m.created_at<=?"]
                prm += [max_created_at_unix]

            now = int(time.time())
            wh += ["(m.ttl=0 OR m.created_at+m.ttl>?)"]
            prm += [now]

            if tags:
                tag_json = json.dumps([t.strip().lower() for t in tags if t.strip()])
                wh += ["json_contains_all(m.tags, ?)"]
                prm += [tag_json]

            if query:
                sanitized_fts_term = re.sub(r'[^a-zA-Z0-9\s*+\-"]', "", query).strip()
                if sanitized_fts_term:  # Only add FTS components if there's a searchable term
                    # The 'joins' variable is initialized to "" at the start of the 'if w_kw:' block, so simply appending here is fine.
                    joins += " JOIN memory_fts f ON m.rowid=f.rowid"
                    wh += ["f.memory_fts MATCH ?"]
                    prm += [sanitized_fts_term]
                else:
                    logger.debug(
                        f"FTS part of query sanitized to empty. Original: '{query}'. Skipping FTS MATCH."
                    )

            sql_kw = (
                "SELECT m.memory_id, "
                "compute_memory_relevance(m.importance,m.confidence,m.created_at,"
                "IFNULL(m.access_count,0),m.last_accessed) AS kw_rel "
                "FROM memories m" + joins + (" WHERE " + " AND ".join(wh) if wh else "")
            )

            rows_kw = await conn.execute_fetchall(sql_kw, prm)
            if rows_kw:
                max_rel = max(r["kw_rel"] for r in rows_kw) or 1e-6
                for r in rows_kw:
                    nid = r["memory_id"]
                    score_map[nid]["keyword"] = min(max(r["kw_rel"] / max_rel, 0.0), 1.0)

        # ───────── hybrid score and ranking ─────────
        for sc in score_map.values():
            sc["hybrid"] = sc["semantic"] * w_sem + sc["keyword"] * w_kw

        ranked = sorted(score_map.items(), key=lambda i: i[1]["hybrid"], reverse=True)
        total_considered = len(ranked)
        ranked_page = ranked[offset : offset + limit]
        ids_page = [m_id for m_id, _ in ranked_page]

        # ───────── fetch full rows ─────────
        memories: list[dict[str, Any]] = []
        if ids_page:
            cols = (
                "memory_id, workflow_id, memory_level, memory_type, importance, confidence, "
                "description, reasoning, source, tags, created_at, updated_at, last_accessed, "
                "access_count, ttl, action_id, thought_id, artifact_id"
                + (", content" if include_content else "")
            )
            rows = await conn.execute_fetchall(
                f"SELECT {cols} FROM memories WHERE memory_id IN ({','.join('?' * len(ids_page))})",
                ids_page,
            )
            row_map = {r["memory_id"]: dict(r) for r in rows}

            # ───────── optional links ─────────
            link_map: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
                lambda: {"outgoing": [], "incoming": []}
            )
            if include_links and ids_page:
                ph = ",".join("?" * len(ids_page))
                if ld in {"outgoing", "both"}:
                    async with conn.execute(
                        f"""
                        SELECT ml.*, t.description AS target_description, t.memory_type AS target_type
                        FROM memory_links ml
                        JOIN memories t ON ml.target_memory_id = t.memory_id
                        WHERE ml.source_memory_id IN ({ph})
                        """,
                        ids_page,
                    ) as cur:
                        async for r in cur:
                            link_map[r["source_memory_id"]]["outgoing"].append(dict(r))
                if ld in {"incoming", "both"}:
                    async with conn.execute(
                        f"""
                        SELECT ml.*, s.description AS source_description, s.memory_type AS source_type
                        FROM memory_links ml
                        JOIN memories s ON ml.source_memory_id = s.memory_id
                        WHERE ml.target_memory_id IN ({ph})
                        """,
                        ids_page,
                    ) as cur:
                        async for r in cur:
                            link_map[r["target_memory_id"]]["incoming"].append(dict(r))

            # ───────── build final list & access updates ─────────
            upd_params, log_params = [], []
            ts_now = int(time.time())
            for mem_id in ids_page:
                row = row_map[mem_id]
                sc = score_map[mem_id]
                row.update(
                    hybrid_score=round(sc["hybrid"], 4),
                    semantic_score=round(sc["semantic"], 4),
                    keyword_relevance_score=round(sc["keyword"], 4),
                    tags=await MemoryUtils.deserialize(row.get("tags")),
                )
                if include_links:
                    row["links"] = link_map[mem_id]
                memories.append(row)

                # prepare access update + log
                upd_params.append((ts_now, mem_id))
                log_params.append(
                    (
                        row["workflow_id"],
                        "hybrid_access",
                        mem_id,
                        None,
                        {"query": query[:100], "hybrid_score": row["hybrid_score"]},
                    )
                )

            # batch update & log
            if upd_params:
                await conn.executemany(
                    "UPDATE memories SET last_accessed=?, access_count=COALESCE(access_count,0)+1 "
                    "WHERE memory_id=?",
                    upd_params,
                )
                for p in log_params:
                    await MemoryUtils._log_memory_operation(conn, *p)

    # ───────── timestamp iso decoration ─────────
    def add_iso(d: dict[str, Any], ks: Sequence[str]) -> None:
        for k in ks:
            if (v := d.get(k)) is not None:
                d[f"{k}_iso"] = safe_format_timestamp(v)

    for m in memories:
        add_iso(m, ["created_at", "updated_at", "last_accessed"])
        if include_links:
            for dir_ in ("outgoing", "incoming"):
                for ln in m.get("links", {}).get(dir_, []):
                    add_iso(ln, ["created_at"])

    proc_time = time.time() - t0
    logger.info(
        f"Hybrid search ({query[:40]}…) → {len(memories)} rows in {proc_time:.3f}s",
        emoji_key="sparkles",
    )
    return {
        "memories": memories,
        "total_candidates_considered": total_considered,
        "success": True,
        "processing_time": proc_time,
    }


@with_tool_metrics
@with_error_handling
async def create_memory_link(
    source_memory_id: str,
    target_memory_id: str,
    link_type: str,
    *,
    strength: float = 1.0,
    description: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Create – or replace – a typed link between two memories.
    Conditionally triggers a WAL checkpoint.

    • Enforces UNIQUE(source_memory_id, target_memory_id, link_type) constraint via
      `INSERT OR REPLACE`.
    • Logs the operation in `memory_operations`.
    """
    # ─────── basic validation ───────
    if not source_memory_id:
        raise ToolInputError("Source memory ID required.", param_name="source_memory_id")
    if not target_memory_id:
        raise ToolInputError("Target memory ID required.", param_name="target_memory_id")
    if source_memory_id == target_memory_id:
        raise ToolInputError("Cannot link memory to itself.", param_name="source_memory_id")

    try:
        link_type_enum = LinkType(link_type.lower())
    except ValueError as exc:
        valid = ", ".join(lt.value for lt in LinkType)
        raise ToolInputError(
            f"Invalid link_type. Must be one of: {valid}", param_name="link_type"
        ) from exc

    if not 0.0 <= strength <= 1.0:
        raise ToolInputError("Strength must be 0.0–1.0.", param_name="strength")

    # ─────── prepare ───────
    link_id = MemoryUtils.generate_id()
    now_unix = int(time.time())
    started = time.time()

    db = DBConnection(db_path)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            src = await conn.execute_fetchone(
                "SELECT workflow_id FROM memories WHERE memory_id = ?",
                (source_memory_id,),
            )
            if src is None:
                raise ToolInputError(
                    f"Source memory {source_memory_id} not found.", param_name="source_memory_id"
                )
            workflow_id = src["workflow_id"]

            tgt = await conn.execute_fetchone(
                "SELECT 1 FROM memories WHERE memory_id = ?",
                (target_memory_id,),
            )
            if tgt is None:
                raise ToolInputError(
                    f"Target memory {target_memory_id} not found.", param_name="target_memory_id"
                )

            await conn.execute(
                """
                INSERT OR REPLACE INTO memory_links
                    (link_id, source_memory_id, target_memory_id,
                     link_type, strength, description, created_at)
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

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "link_created",
                source_memory_id,
                None,
                {
                    "target_memory_id": target_memory_id,
                    "link_type": link_type_enum.value,
                    "link_id": link_id,
                    "strength": strength,
                    "description": description or "",
                },
            )

        # ─────── success payload ───────
        elapsed = time.time() - started
        result = {
            "link_id": link_id,
            "source_memory_id": source_memory_id,
            "target_memory_id": target_memory_id,
            "link_type": link_type_enum.value,
            "strength": strength,
            "description": description or "",
            "created_at_unix": now_unix,
            "created_at_iso": to_iso_z(now_unix),
            "success": True,
            "processing_time": elapsed,
        }
        logger.info(
            f"Memory link {link_id} ⟶ {_fmt_id(target_memory_id)} [{link_type_enum.value}] created.",
            emoji_key="link",
            time=elapsed,
        )
        return result

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"create_memory_link failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to create memory link: {exc}") from exc


# --- 7. Core Memory Retrieval ---


@with_tool_metrics
@with_error_handling
async def query_memories(
    *,
    workflow_id: str | None = None,
    memory_level: str | None = None,
    memory_type: str | None = None,
    search_text: str | None = None,
    tags: list[str] | None = None,
    min_importance: float | None = None,
    max_importance: float | None = None,
    min_confidence: float | None = None,
    min_created_at_unix: int | None = None,
    max_created_at_unix: int | None = None,
    sort_by: str = "relevance",  # relevance, importance, created_at, …
    sort_order: str = "DESC",
    include_content: bool = True,
    include_links: bool = False,
    link_direction: str = "outgoing",  # outgoing / incoming / both
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Filter, rank and paginate memories.

    • All rows returned as dicts with raw ints + *_iso timestamp companions.
    • Access statistics are updated atomically inside the same write txn.
    • Deterministic UDFs (`json_contains_all`, `compute_memory_relevance`) used
      for tag filtering and dynamic scoring.
    """
    t0 = time.time()

    # ───────── Validation ─────────
    sort_fields = {
        "relevance",
        "importance",
        "created_at",
        "updated_at",
        "confidence",
        "last_accessed",
        "access_count",
    }
    if sort_by not in sort_fields:
        raise ToolInputError(
            f"sort_by must be one of {', '.join(sort_fields)}", param_name="sort_by"
        )
    if sort_order.upper() not in {"ASC", "DESC"}:
        raise ToolInputError("sort_order must be 'ASC' or 'DESC'", param_name="sort_order")
    if limit < 1:
        raise ToolInputError("limit must be ≥ 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("offset must be ≥ 0", param_name="offset")

    if memory_level:
        MemoryLevel(memory_level.lower())  # raises ValueError if invalid
    if memory_type:
        MemoryType(memory_type.lower())

    # ───────── Query assembly ─────────
    sel_cols = [
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
        sel_cols.append("m.content")

    select_clause = ", ".join(
        sel_cols
        + [
            # relevance is always selected; used both for ordering and caller display
            "compute_memory_relevance("
            "m.importance, m.confidence, m.created_at, "
            "IFNULL(m.access_count,0), m.last_accessed) AS relevance"
        ]
    )

    joins: list[str] = []
    where: list[str] = ["1=1"]
    params: list[Any] = []
    fts_params: list[Any] = []

    # Structured filters
    if workflow_id:
        where.append("m.workflow_id = ?")
        params.append(workflow_id)
    if memory_level:
        where.append("m.memory_level = ?")
        params.append(memory_level.lower())
    if memory_type:
        where.append("m.memory_type = ?")
        params.append(memory_type.lower())
    if min_importance is not None:
        where.append("m.importance >= ?")
        params.append(min_importance)
    if max_importance is not None:
        where.append("m.importance <= ?")
        params.append(max_importance)
    if min_confidence is not None:
        where.append("m.confidence >= ?")
        params.append(min_confidence)
    if min_created_at_unix is not None:
        where.append("m.created_at >= ?")
        params.append(min_created_at_unix)
    if max_created_at_unix is not None:
        where.append("m.created_at <= ?")
        params.append(max_created_at_unix)

    # TTL constraint
    now_int = int(time.time())
    where.append("(m.ttl = 0 OR m.created_at + m.ttl > ?)")
    params.append(now_int)

    # Tag filtering
    if tags:
        tag_list = [t.strip().lower() for t in tags if t.strip()]
        if tag_list:
            where.append("json_contains_all(m.tags, ?)")
            params.append(json.dumps(tag_list))

    # FTS search
    if search_text:
        joins.append("JOIN memory_fts fts ON fts.rowid = m.rowid")
        sanitized = re.sub(r'[^a-zA-Z0-9\s*+\-"]', "", search_text).strip()
        if sanitized:
            where.append("fts.memory_fts MATCH ?")
            fts_params.append(sanitized)

    where_sql = " AND ".join(where)
    join_sql = " ".join(joins)

    # ───────── SQL strings ─────────
    base_from = f"FROM memories m {join_sql} WHERE {where_sql}"
    count_sql = f"SELECT COUNT(*) {base_from}"
    data_sql = f"SELECT {select_clause} {base_from}"

    # Ordering
    if sort_by == "relevance":
        order_sql = "ORDER BY relevance"
    else:
        safe_col = MemoryUtils._validate_sql_identifier(sort_by, "sort_by")
        order_sql = f"ORDER BY m.{safe_col}"
    order_sql += f" {sort_order.upper()}"

    # Pagination
    paginated_sql = f"{data_sql} {order_sql} LIMIT ? OFFSET ?"

    db = DBConnection(db_path)

    # ───────── DB interaction (single write txn) ─────────
    async with db.transaction() as conn:
        # Workflow existence check (cheap, read‐only inside same txn)
        if workflow_id:
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        total_matching = (await conn.execute_fetchone(count_sql, params + fts_params))[0]

        rows = await conn.execute_fetchall(paginated_sql, params + fts_params + [limit, offset])

        memories: list[dict[str, Any]] = []
        now_unix = int(time.time())
        access_updates: list[tuple[int, str]] = []
        op_logs: list[tuple[str, str, str, None, dict[str, Any]]] = []

        for r in rows:
            mem = dict(r)
            mem["tags"] = await MemoryUtils.deserialize(mem.get("tags"))
            # raw timestamps kept; iso added later
            memories.append(mem)

            # prepare access-stat update
            access_updates.append((now_unix, mem["memory_id"]))
            op_logs.append(
                (
                    mem["workflow_id"],
                    "query_access",
                    mem["memory_id"],
                    None,
                    {"query_filters": {"sort": sort_by, "limit": limit}},
                )
            )

        # batch access-stat update
        if access_updates:
            await conn.executemany(
                """
                UPDATE memories
                   SET last_accessed = ?, access_count = COALESCE(access_count,0)+1
                 WHERE memory_id = ?
                """,
                access_updates,
            )
            for log_row in op_logs:
                await MemoryUtils._log_memory_operation(conn, *log_row)

    # ───────── optional linked memories (read-only) ─────────
    if include_links and memories:

        async def _get_links(mid: str) -> dict[str, Any]:
            try:
                data = await get_linked_memories(
                    memory_id=mid,
                    direction=link_direction.lower(),
                    limit=5,
                    include_memory_details=False,
                    db_path=db_path,
                )
                return data.get("links", {})
            except Exception as e:
                logger.warning(f"link fetch failed for {mid}: {e}")
                return {"error": str(e)}

        # gather concurrently – small fan-out
        link_tasks = {
            m["memory_id"]: asyncio.create_task(_get_links(m["memory_id"])) for m in memories
        }
        for m in memories:
            m["links"] = await link_tasks[m["memory_id"]]

    # ───────── timestamp prettification ─────────
    def _iso(obj: dict[str, Any], ks: Sequence[str]) -> None:
        for k in ks:
            if k in obj and obj[k] is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(obj[k])

    for m in memories:
        _iso(m, ["created_at", "updated_at", "last_accessed"])

    elapsed = time.time() - t0
    logger.info(f"query_memories → {len(memories)}/{total_matching} rows in {elapsed:0.2f}s")

    return {
        "memories": memories,
        "total_matching_count": total_matching,
        "success": True,
        "processing_time": elapsed,
    }


# --- 8. Workflow Listing & Details ---


@with_tool_metrics
@with_error_handling
async def list_workflows(
    *,
    status: str | None = None,
    tag: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
    limit: int = 10,
    offset: int = 0,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return a paginated list of workflows with optional filters.

    • Timestamps stay as integers; *_iso companions are added.
    • Tag filter is case-sensitive (unchanged – DB collation may override).
    • `total_count` gives the number of records that *match*, ignoring LIMIT/OFFSET.
    """
    # ──────────── validation ────────────
    if status:
        try:
            WorkflowStatus(status.lower())
        except ValueError as exc:
            raise ToolInputError(f"Invalid status: {status}", param_name="status") from exc

    def _iso_to_ts(iso: str, field: str) -> int:
        try:
            return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
        except ValueError as exc:
            raise ToolInputError(f"Invalid {field} format.", param_name=field) from exc

    after_ts = _iso_to_ts(after_date, "after_date") if after_date else None
    before_ts = _iso_to_ts(before_date, "before_date") if before_date else None

    if limit < 1:
        raise ToolInputError("Limit must be ≥ 1", param_name="limit")
    if offset < 0:
        raise ToolInputError("Offset must be ≥ 0", param_name="offset")

    db = DBConnection(db_path)

    # helper
    def _add_iso(obj: Dict[str, Any], *keys: str) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── build WHERE/JOIN fragments ─────────
            joins: list[str] = []
            where: list[str] = ["1=1"]
            params: list[Any] = []

            if tag:
                joins.append(
                    "JOIN workflow_tags wt ON wt.workflow_id = w.workflow_id "
                    "JOIN tags t ON t.tag_id = wt.tag_id"
                )
                where.append("t.name = ?")
                params.append(tag)
            if status:
                where.append("w.status = ?")
                params.append(status.lower())
            if after_ts is not None:
                where.append("w.created_at >= ?")
                params.append(after_ts)
            if before_ts is not None:
                where.append("w.created_at <= ?")
                params.append(before_ts)

            join_sql = "".join(joins)
            where_sql = " WHERE " + " AND ".join(where)

            # ───────── total count ─────────
            total_sql = (
                "SELECT COUNT(DISTINCT w.workflow_id) FROM workflows w " + join_sql + where_sql
            )
            total_count = await conn.execute_fetchone(total_sql, params) or (0,)
            total_count = total_count[0]

            # ───────── main data query ─────
            data_sql = (
                "SELECT DISTINCT w.workflow_id, w.title, w.description, w.goal, "
                "w.status, w.created_at, w.updated_at, w.completed_at "
                "FROM workflows w " + join_sql + where_sql + " ORDER BY w.updated_at DESC "
                "LIMIT ? OFFSET ?"
            )
            rows = await conn.execute_fetchall(data_sql, params + [limit, offset])

            workflows: list[Dict[str, Any]] = [dict(r) for r in rows]
            wf_ids = [wf["workflow_id"] for wf in workflows]

            # ───────── attach tags ─────────
            if wf_ids:
                placeholders = ",".join("?" * len(wf_ids))
                tag_sql = (
                    f"SELECT wt.workflow_id, t.name "
                    f"FROM workflow_tags wt "
                    f"JOIN tags t ON t.tag_id = wt.tag_id "
                    f"WHERE wt.workflow_id IN ({placeholders})"
                )
                tag_rows = await conn.execute_fetchall(tag_sql, wf_ids)

                tag_map: dict[str, list[str]] = defaultdict(list)
                for r in tag_rows:
                    tag_map[r["workflow_id"]].append(r["name"])

                for wf in workflows:
                    wf["tags"] = tag_map.get(wf["workflow_id"], [])
            else:
                for wf in workflows:
                    wf["tags"] = []

            # ───────── ISO decoration ──────
            for wf in workflows:
                _add_iso(wf, "created_at", "updated_at", "completed_at")

            result = {"workflows": workflows, "total_count": total_count, "success": True}
            logger.info(
                f"list_workflows → {len(workflows)} rows (total={total_count})",
                emoji_key="scroll",
            )
            return result

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"list_workflows failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to list workflows: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_workflow_details(
    workflow_id: str,
    *,
    include_actions: bool = True,
    include_artifacts: bool = True,
    include_thoughts: bool = True,
    include_memories: bool = False, # Default from original
    include_cognitive_states: bool = False, # NEW PARAMETER
    memories_limit: int = 20,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Hydrate a workflow with all optional children, including cognitive states.

    * Raw integer timestamps are preserved.
    * ISO-8601 siblings are added as *_iso.
    * Memory rows include `relevance` from the deterministic UDF.
    * Cognitive states include deserialized JSON fields and ISO timestamps.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0_perf = time.perf_counter() # For more precise processing time
    db = DBConnection(db_path)

    def _add_iso(row: Dict[str, Any], keys: Sequence[str]) -> None:
        """Append *_iso keys in-place."""
        for k in keys:
            if (ts := row.get(k)) is not None:
                row[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        # snapshot read – URI ?mode=ro behind the scenes → zero WAL pressure
        async with db.transaction(readonly=True) as conn:
            # ───────── workflow core ─────────
            wf_row = await conn.execute_fetchone(
                "SELECT * FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
            if wf_row is None:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            details: Dict[str, Any] = dict(wf_row)
            details["metadata"] = await MemoryUtils.deserialize(details.get("metadata"))

            # ───────── tags ─────────
            tag_rows = await conn.execute_fetchall(
                """SELECT t.name
                   FROM tags t
                   JOIN workflow_tags wt ON wt.tag_id = t.tag_id
                   WHERE wt.workflow_id = ?""",
                (workflow_id,),
            )
            details["tags"] = [row["name"] for row in tag_rows]

            # ───────── actions ─────────
            if include_actions:
                details["actions"] = []
                async with conn.execute(
                    """
                    SELECT a.*,
                           GROUP_CONCAT(DISTINCT t.name) AS tags_str
                    FROM   actions a
                           LEFT JOIN action_tags at ON at.action_id = a.action_id
                           LEFT JOIN tags        t  ON t.tag_id      = at.tag_id
                    WHERE  a.workflow_id = ?
                    GROUP  BY a.action_id
                    ORDER  BY a.sequence_number
                    """,
                    (workflow_id,),
                ) as cur:
                    async for row_raw_action in cur:
                        act = dict(row_raw_action)
                        act["tool_args"] = await MemoryUtils.deserialize(act.get("tool_args"))
                        act["tool_result"] = await MemoryUtils.deserialize(act.get("tool_result"))
                        act["tags"] = row_raw_action["tags_str"].split(",") if row_raw_action["tags_str"] else []
                        act.pop("tags_str", None)
                        details["actions"].append(act)

            # ───────── artifacts ─────────
            if include_artifacts:
                details["artifacts"] = []
                async with conn.execute(
                    """
                    SELECT a.*,
                           GROUP_CONCAT(DISTINCT t.name) AS tags_str
                    FROM   artifacts a
                           LEFT JOIN artifact_tags att ON att.artifact_id = a.artifact_id
                           LEFT JOIN tags          t   ON t.tag_id        = att.tag_id
                    WHERE  a.workflow_id = ?
                    GROUP  BY a.artifact_id
                    ORDER  BY a.created_at
                    """,
                    (workflow_id,),
                ) as cur:
                    async for row_raw_artifact in cur:
                        art = dict(row_raw_artifact)
                        art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
                        art["is_output"] = bool(art["is_output"]) # Ensure boolean
                        art["tags"] = row_raw_artifact["tags_str"].split(",") if row_raw_artifact["tags_str"] else []
                        art.pop("tags_str", None)
                        if art.get("content") and len(art["content"]) > 200: # content_preview logic from original
                            art["content_preview"] = art["content"][:197] + "…"
                        details["artifacts"].append(art)

            # ───────── thought chains / thoughts ─────────
            if include_thoughts:
                details["thought_chains"] = []
                async with conn.execute(
                    "SELECT * FROM thought_chains WHERE workflow_id = ? ORDER BY created_at",
                    (workflow_id,),
                ) as chains_cursor:
                    async for chain_raw in chains_cursor:
                        chain_dict = dict(chain_raw)
                        chain_dict["thoughts"] = []
                        async with conn.execute(
                            "SELECT * FROM thoughts "
                            "WHERE thought_chain_id = ? "
                            "ORDER BY sequence_number",
                            (chain_dict["thought_chain_id"],),
                        ) as thoughts_cursor:
                            async for thought_raw in thoughts_cursor:
                                chain_dict["thoughts"].append(dict(thought_raw))
                        details["thought_chains"].append(chain_dict)

            # ───────── memories (scored) ─────────
            if include_memories:
                details["memories_sample"] = []
                async with conn.execute(
                    """
                    SELECT memory_id, content, memory_type, memory_level,
                           importance, confidence, access_count,
                           created_at, last_accessed,
                           compute_memory_relevance(
                               importance, confidence, created_at,
                               IFNULL(access_count,0), last_accessed
                           ) AS relevance
                    FROM   memories
                    WHERE  workflow_id = ?
                    ORDER  BY relevance DESC
                    LIMIT  ?
                    """,
                    (workflow_id, memories_limit),
                ) as mems_cursor:
                    async for row_raw_mem in mems_cursor:
                        mem = dict(row_raw_mem)
                        # 'created_at' is already an integer timestamp from DB
                        # _add_iso will handle creating 'created_at_iso' and 'last_accessed_iso'
                        # The original used 'created_at_unix' as a duplicate; this can be omitted
                        # as 'created_at' already holds the Unix timestamp.
                        if mem.get("content") and len(mem["content"]) > 150: # content_preview logic
                            mem["content_preview"] = mem["content"][:147] + "…"
                        details["memories_sample"].append(mem)
            
            # --- FETCH COGNITIVE STATES (NEW) ---
            if include_cognitive_states:
                details["cognitive_states"] = []
                async with conn.execute(
                    # Fetch ALL cognitive states for the workflow, order by is_latest then created_at
                    # This ensures the 'is_latest=True' one is first if it exists,
                    # otherwise the most recently created one.
                    "SELECT * FROM cognitive_states WHERE workflow_id = ? ORDER BY is_latest DESC, created_at DESC",
                    (workflow_id,),
                ) as cog_states_cursor:
                    async for cs_row_raw in cog_states_cursor:
                        cs_row = dict(cs_row_raw)
                        # Deserialize JSON fields
                        for json_field_cs in ["working_memory", "focus_areas", "context_actions", "current_goals"]:
                            if cs_row.get(json_field_cs): # Check if field exists and is not None
                                cs_row[json_field_cs] = await MemoryUtils.deserialize(cs_row[json_field_cs])
                        
                        # ISO decoration for timestamps
                        _add_iso(cs_row, ["created_at", "last_active"])
                        
                        # Ensure is_latest is boolean
                        cs_row["is_latest"] = bool(cs_row.get("is_latest", False))
                        
                        details["cognitive_states"].append(cs_row)
            # --- END COGNITIVE STATES FETCH ---

            # ───────── ISO decoration for root + children ─────────
            _add_iso(details, ["created_at", "updated_at", "completed_at", "last_active"])
            for act_item in details.get("actions", []): # Use different var name
                _add_iso(act_item, ["started_at", "completed_at"])
            for art_item in details.get("artifacts", []): # Use different var name
                _add_iso(art_item, ["created_at"])
            for ch_item in details.get("thought_chains", []): # Use different var name
                _add_iso(ch_item, ["created_at"])
                for th_item in ch_item.get("thoughts", []): # Use different var name
                    _add_iso(th_item, ["created_at"])
            # ISO decoration for memories_sample was already handled by _add_iso within the loop
            # for memories_sample items in the original code. Re-check this.
            # Original code: _add_iso(mem, ["created_at", "last_accessed"]) for memories. This is correct.
            # Let's ensure it is applied to each memory in memories_sample if that list exists.
            if "memories_sample" in details:
                for mem_item in details["memories_sample"]:
                    _add_iso(mem_item, ["created_at", "last_accessed"])
            
            # ISO decoration for cognitive_states was handled inside its loop.

            details["success"] = True
            processing_time = time.perf_counter() - t0_perf
            details["processing_time_seconds"] = round(processing_time, 4)

            logger.info(
                f"Workflow {workflow_id} hydrated. Actions: {len(details.get('actions', []))}, "
                f"Artifacts: {len(details.get('artifacts', []))}, Thoughts: {sum(len(tc.get('thoughts',[])) for tc in details.get('thought_chains',[]))}, "
                f"Memories Sample: {len(details.get('memories_sample',[])) if include_memories else 'N/A'}, "
                f"Cognitive States: {len(details.get('cognitive_states',[])) if include_cognitive_states else 'N/A'}. "
                f"Time: {processing_time:.3f}s",
                emoji_key="books"
            )
            return details

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_workflow_details({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get workflow details: {exc}") from exc


# --- 9. Action Details ---


@with_tool_metrics
@with_error_handling
async def get_recent_actions(
    workflow_id: str,
    *,
    limit: int = 5,
    action_type: str | None = None,
    status: str | None = None,
    include_tool_results: bool = True,
    include_reasoning: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Return the *latest* `limit` actions for a workflow, optionally filtered.

    • Raw integer timestamps (`started_at`, `completed_at`) are preserved.
      ISO companions are added under *_iso.
    • Supports `action_type`, `status` filters; validates against enums.
    • Optional columns: `reasoning` and `tool_result`.
    """
    # ───────────── validation ─────────────
    if not (isinstance(limit, int) and limit > 0):
        raise ToolInputError("limit must be a positive integer", param_name="limit")

    if action_type:
        try:
            ActionType(action_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid action_type '{action_type}'. Allowed: {[t.value for t in ActionType]}",
                param_name="action_type",
            ) from e

    if status:
        try:
            ActionStatus(status.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid status '{status}'. Allowed: {[s.value for s in ActionStatus]}",
                param_name="status",
            ) from e

    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───── verify workflow & get title ─────
            wf_row = await conn.execute_fetchone(
                "SELECT title FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if wf_row is None:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            workflow_title = wf_row["title"]

            # ───── build dynamic SELECT ─────
            cols = [
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
                "GROUP_CONCAT(t.name) AS tags_str",
            ]
            if include_reasoning:
                cols.append("a.reasoning")
            if include_tool_results:
                cols.append("a.tool_result")

            sql = (
                f"SELECT {', '.join(cols)} "
                "FROM actions a "
                "LEFT JOIN action_tags at ON at.action_id = a.action_id "
                "LEFT JOIN tags t       ON t.tag_id    = at.tag_id "
                "WHERE a.workflow_id = ?"
            )
            params: list[Any] = [workflow_id]

            if action_type:
                sql += " AND a.action_type = ?"
                params.append(action_type.lower())

            if status:
                sql += " AND a.status = ?"
                params.append(status.lower())

            sql += " GROUP BY a.action_id ORDER BY a.sequence_number DESC LIMIT ?"
            params.append(limit)

            # ───── execute & transform ─────
            actions: list[Dict[str, Any]] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    a = dict(row)

                    a["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                    a.pop("tags_str", None)

                    a["tool_args"] = await MemoryUtils.deserialize(a.get("tool_args"))
                    if include_tool_results and "tool_result" in a:
                        a["tool_result"] = await MemoryUtils.deserialize(a.get("tool_result"))

                    _add_iso(a, ["started_at", "completed_at"])
                    actions.append(a)

            logger.info(
                f"Fetched {len(actions)} recent actions for workflow {workflow_id}",
                emoji_key="rewind",
            )
            return {
                "workflow_id": workflow_id,
                "workflow_title": workflow_title,
                "actions": actions,
                "success": True,
            }

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_recent_actions({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get recent actions: {exc}") from exc


# --- 10. Artifact Details ---
@with_tool_metrics
@with_error_handling
async def get_artifacts(
    workflow_id: str,
    *,
    artifact_type: str | None = None,
    tag: str | None = None,
    is_output: bool | None = None,
    include_content: bool = False,
    limit: int = 10,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve artifacts for a workflow with rich filtering.

    • Keeps raw `created_at` integer; adds `created_at_iso`.
    • Content trimmed to a preview unless `include_content=True`.
    """
    if limit < 1:
        raise ToolInputError("limit must be ≥ 1", param_name="limit")

    if artifact_type:
        try:
            ArtifactType(artifact_type.lower())
        except ValueError as e:
            raise ToolInputError(
                f"Invalid artifact_type '{artifact_type}'",
                param_name="artifact_type",
            ) from e

    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ensure workflow exists
            exists = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if exists is None:
                raise ToolInputError(f"Workflow {workflow_id} not found", param_name="workflow_id")

            # dynamic SQL
            select_cols = (
                "a.artifact_id, a.action_id, a.artifact_type, a.name, a.description, "
                "a.path, a.metadata, a.created_at, a.is_output, "
                "GROUP_CONCAT(t.name) AS tags_str"
            )
            if include_content:
                select_cols += ", a.content"

            sql = (
                f"SELECT {select_cols} "
                "FROM artifacts a "
                "LEFT JOIN artifact_tags att ON att.artifact_id = a.artifact_id "
                "LEFT JOIN tags          t   ON t.tag_id       = att.tag_id "
                "WHERE a.workflow_id = ?"
            )
            params: list[Any] = [workflow_id]

            if tag:
                sql += " AND t.name = ?"
                params.append(tag)

            if artifact_type:
                sql += " AND a.artifact_type = ?"
                params.append(artifact_type.lower())

            if is_output is not None:
                sql += " AND a.is_output = ?"
                params.append(1 if is_output else 0)

            sql += " GROUP BY a.artifact_id ORDER BY a.created_at DESC LIMIT ?"
            params.append(limit)

            # fetch + transform
            artifacts: list[Dict[str, Any]] = []
            async with conn.execute(sql, params) as cur:
                async for row in cur:
                    art = dict(row)

                    art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
                    art["is_output"] = bool(art["is_output"])
                    art["tags"] = row["tags_str"].split(",") if row["tags_str"] else []
                    art.pop("tags_str", None)

                    if not include_content and art.get("content"):
                        if len(art["content"]) > 100:
                            art["content_preview"] = art["content"][:97] + "…"
                        art.pop("content", None)

                    _add_iso(art, ["created_at"])
                    artifacts.append(art)

            logger.info(
                f"Fetched {len(artifacts)} artifacts for workflow {workflow_id}",
                emoji_key="open_file_folder",
            )
            return {"workflow_id": workflow_id, "artifacts": artifacts, "success": True}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_artifacts({workflow_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get artifacts: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_artifact_by_id(
    artifact_id: str,
    *,
    include_content: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single artifact (and optionally its content) **and**
    record the access against its linked memory row.

    • Preserves raw integer timestamps; adds `created_at_iso`.
    • Tags are returned as a list.
    • If `include_content` is False the `content` key is removed.
    """
    if not artifact_id:
        raise ToolInputError("Artifact ID required.", param_name="artifact_id")

    db = DBConnection(db_path)
    start_time = time.time()

    # Helper for ISO decoration
    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if ts := obj.get(key):
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    async with db.transaction() as conn:  # R/W – we may update memory row
        # ───────── Artifact row ─────────
        artifact_row = await conn.execute_fetchone(
            """
            SELECT a.*, GROUP_CONCAT(t.name) AS tags_str
            FROM artifacts a
            LEFT JOIN artifact_tags att ON att.artifact_id = a.artifact_id
            LEFT JOIN tags          t    ON t.tag_id       = att.tag_id
            WHERE a.artifact_id = ?
            GROUP BY a.artifact_id
            """,
            (artifact_id,),
        )
        if artifact_row is None:
            raise ToolInputError(f"Artifact {artifact_id} not found.", param_name="artifact_id")

        art = dict(artifact_row)
        art["metadata"] = await MemoryUtils.deserialize(art.get("metadata"))
        art["is_output"] = bool(art["is_output"])
        art["tags"] = art.pop("tags_str").split(",") if artifact_row["tags_str"] else []

        if not include_content:
            art.pop("content", None)

        # ───────── Memory linkage / log ─────────
        mem = await conn.execute_fetchone(
            "SELECT memory_id, workflow_id FROM memories WHERE artifact_id = ?",
            (artifact_id,),
        )
        if mem:
            await MemoryUtils._update_memory_access(conn, mem["memory_id"])
            await MemoryUtils._log_memory_operation(
                conn,
                mem["workflow_id"],
                "access_via_artifact",
                mem["memory_id"],
                None,
                {"artifact_id": artifact_id},
            )

        # ───────── Post-transaction formatting ─────────
        _add_iso(art, "created_at")
        art["success"] = True
        art["processing_time"] = time.time() - start_time
        logger.info(f"Artifact {_fmt_id(artifact_id)} fetched.", emoji_key="page_facing_up")
        return art


# --- 10.5 Goals ---
@with_tool_metrics
@with_error_handling
async def create_goal(
    workflow_id: str,
    description: str,
    *,
    parent_goal_id: str | None = None,
    title: str | None = None,
    priority: int = 3,
    reasoning: str | None = None,
    acceptance_criteria: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    initial_status: str = GoalStatus.ACTIVE.value,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    # ... (validation logic remains the same) ...
    if not description:
        raise ToolInputError("Goal description is required.", param_name="description")
    try:
        status_enum = GoalStatus(initial_status.lower())
    except ValueError as exc:
        raise ToolInputError(
            f"Invalid initial_status '{initial_status}'. Must be one of: {', '.join(gs.value for gs in GoalStatus)}",
            param_name="initial_status",
        ) from exc

    now_unix = int(time.time())
    t0_perf = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_goal_details(
        conn_fetch: aiosqlite.Connection, existing_goal_id: str
    ) -> Dict[str, Any]:
        # Reusing get_goal_details logic, simplified for idempotency return
        row = await conn_fetch.execute_fetchone(
            "SELECT * FROM goals WHERE goal_id = ?", (existing_goal_id,)
        )
        if not row:
            raise ToolError(
                f"Failed to re-fetch existing goal {existing_goal_id} on idempotency hit."
            )

        goal_data = dict(row)
        goal_data["acceptance_criteria"] = await MemoryUtils.deserialize(
            goal_data.get("acceptance_criteria")
        )
        goal_data["metadata"] = await MemoryUtils.deserialize(goal_data.get("metadata"))

        def _add_iso_local(obj: Dict[str, Any], keys: tuple[str, ...]) -> None:
            for k_iso in keys:
                if (ts := obj.get(k_iso)) is not None:
                    obj[f"{k_iso}_iso"] = safe_format_timestamp(ts)

        _add_iso_local(goal_data, ("created_at", "updated_at", "completed_at"))

        return {
            "goal": goal_data,
            "success": True,
            "idempotency_hit": True,
            "processing_time": time.perf_counter() - t0_perf,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_goal_row = await conn_check.execute_fetchone(
                "SELECT goal_id FROM goals WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_goal_row:
            existing_goal_id = existing_goal_row["goal_id"]
            logger.info(
                f"Idempotency hit for create_goal (key='{idempotency_key}'). Returning existing goal {_fmt_id(existing_goal_id)}."
            )
            async with db.transaction(
                readonly=True
            ) as conn_details:  # New transaction for fetching details
                return await _fetch_existing_goal_details(conn_details, existing_goal_id)

    goal_id_new = MemoryUtils.generate_id()

    async with db.transaction(mode="IMMEDIATE") as conn:
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id=?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")
        if parent_goal_id and not await conn.execute_fetchone(
            "SELECT 1 FROM goals WHERE goal_id=? AND workflow_id=?",
            (parent_goal_id, workflow_id),
        ):
            raise ToolInputError(
                f"Parent goal {parent_goal_id} not found in workflow {workflow_id}.",
                param_name="parent_goal_id",
            )

        if parent_goal_id:
            seq_row = await conn.execute_fetchone(
                "SELECT MAX(sequence_number) FROM goals WHERE parent_goal_id=? AND workflow_id=?",
                (parent_goal_id, workflow_id),
            )
        else:
            seq_row = await conn.execute_fetchone(
                "SELECT MAX(sequence_number) FROM goals WHERE workflow_id=? AND parent_goal_id IS NULL",
                (workflow_id,),
            )
        sequence_number = (seq_row[0] if seq_row and seq_row[0] is not None else 0) + 1

        await conn.execute(
            """INSERT INTO goals (goal_id, workflow_id, parent_goal_id, title, description, status, priority, reasoning, acceptance_criteria, metadata, created_at, updated_at, sequence_number, completed_at, idempotency_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
            (
                goal_id_new,
                workflow_id,
                parent_goal_id,
                title,
                description,
                status_enum.value,
                priority,
                reasoning,
                await MemoryUtils.serialize(acceptance_criteria or []),
                await MemoryUtils.serialize(metadata or {}),
                now_unix,
                now_unix,
                sequence_number,
                idempotency_key,
            ),
        )
        created_row_raw = await conn.execute_fetchone(
            "SELECT * FROM goals WHERE goal_id=?", (goal_id_new,)
        )
        if not created_row_raw:
            raise ToolError("Failed to retrieve goal after insert.")
        created_row = dict(created_row_raw)
        created_row["acceptance_criteria"] = await MemoryUtils.deserialize(
            created_row.get("acceptance_criteria")
        )
        created_row["metadata"] = await MemoryUtils.deserialize(created_row.get("metadata"))
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "create_goal",
            None,
            None,
            {
                "goal_id": goal_id_new,
                "title": title or description[:50],
                "parent_goal_id": parent_goal_id,
                "status": status_enum.value,
            },
        )

    # ISO decoration for the new row
    def _add_iso_local(o: dict[str, Any], keys: tuple[str, ...]) -> None:  # Local helper
        for k_iso in keys:
            if (ts := o.get(k_iso)) is not None:
                o[f"{k_iso}_iso"] = safe_format_timestamp(ts)

    _add_iso_local(created_row, ("created_at", "updated_at", "completed_at"))

    duration = time.perf_counter() - t0_perf
    logger.info(
        f"Goal '{title or description[:50]}…' ({goal_id_new}) created in workflow {workflow_id}",
        time=duration,
    )
    return {
        "goal": created_row,
        "success": True,
        "idempotency_hit": False,
        "processing_time": duration,
    } 


@with_tool_metrics
@with_error_handling
async def update_goal_status(
    goal_id: str,
    status: str,
    *,
    reason: str | None = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Change a goal's `status` and return the refreshed record.
    Conditionally triggers a WAL checkpoint after successful update.

    • Raw Unix timestamps are kept; *_iso companions are added.
    • `completed_at` is set for terminal states (completed / failed / abandoned).
    • A memory-operations log row is written inside the same transaction.
    """
    if not goal_id:
        raise ToolInputError("Goal ID is required.", param_name="goal_id")

    try:
        status_enum = GoalStatus(status.lower())
    except ValueError as exc:
        opts = ", ".join(g.value for g in GoalStatus)
        raise ToolInputError(
            f"Invalid goal status '{status}'. Must be one of: {opts}", param_name="status"
        ) from exc

    now = int(time.time())
    t0 = time.time()

    db = DBConnection(db_path)
    upd: Optional[Dict[str, Any]] = None  # Initialize to ensure it's defined
    parent_goal_id: Optional[str] = None
    is_root_finished: bool = False
    workflow_id_for_log: Optional[str] = None

    # helper for ISO decoration
    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k_iso in keys:  # Renamed k to k_iso to avoid conflict
            if (ts := obj.get(k_iso)) is not None:  # Use k_iso
                obj[f"{k_iso}_iso"] = safe_format_timestamp(ts)  # Use k_iso

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            row = await conn.execute_fetchone(
                "SELECT workflow_id, parent_goal_id FROM goals WHERE goal_id=?",
                (goal_id,),
            )
            if row is None:
                raise ToolInputError(f"Goal {goal_id} not found.", param_name="goal_id")

            workflow_id_for_log = row["workflow_id"]  # Capture for logging outside transaction
            parent_goal_id = row["parent_goal_id"]

            set_clauses = ["status=?", "updated_at=?"]
            params: list[Any] = [status_enum.value, now]

            if status_enum in {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED}:
                set_clauses.append("completed_at=?")
                params.append(now)

            params.append(goal_id)

            await conn.execute(
                f"UPDATE goals SET {', '.join(set_clauses)} WHERE goal_id=?",
                params,
            )

            updated_row = await conn.execute_fetchone(
                "SELECT * FROM goals WHERE goal_id=?", (goal_id,)
            )
            if updated_row is None:
                raise ToolError("Failed to retrieve goal after update.")  # Should be very rare

            upd = dict(updated_row)  # Assign to the variable defined outside
            upd["acceptance_criteria"] = await MemoryUtils.deserialize(
                upd.get("acceptance_criteria")
            )
            upd["metadata"] = await MemoryUtils.deserialize(upd.get("metadata"))
            _add_iso(upd, ["created_at", "updated_at", "completed_at"])  # upd is now a dict

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id_for_log,  # Use captured workflow_id
                "update_goal_status",
                None,
                None,
                {
                    "goal_id": goal_id,
                    "new_status": status_enum.value,
                    "reason": reason,
                },
            )

            is_root_finished = parent_goal_id is None and status_enum in {
                GoalStatus.COMPLETED,
                GoalStatus.FAILED,
                GoalStatus.ABANDONED,
            }

        dt = time.time() - t0
        logger.info(f"Goal {_fmt_id(goal_id)} set → {status_enum.value}", time=dt)

        if upd is None:  # Should ideally not be None if transaction succeeded
            raise ToolError("Internal error: Updated goal data not available after transaction.")

        return {
            "updated_goal_details": upd,  # upd is now guaranteed to be a dict
            "parent_goal_id": parent_goal_id,
            "is_root_finished": is_root_finished,
            "success": True,
            "processing_time": dt,
        }

    except ToolInputError:
        # Log before re-raising for better context if needed, or just re-raise
        logger.error(
            f"Input error updating goal {goal_id}: status='{status}'", exc_info=False
        )  # exc_info=False for ToolInputError
        raise
    except Exception as exc:
        # Log with full traceback for unexpected errors
        logger.error(
            f"Unexpected error updating goal {goal_id} (status='{status}'): {exc}", exc_info=True
        )
        raise ToolError(f"Failed to update goal status: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def get_goal_details(
    goal_id: str,
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single goal row and return a richly formatted record.

    • Keeps raw integer timestamps and appends *_iso keys
    • Robust JSON deserialisation for `acceptance_criteria` and `metadata`
    """
    t0 = time.time()

    if not goal_id:
        raise ToolInputError("Goal ID is required.", param_name="goal_id")

    db = DBConnection(db_path)

    def _safe_json(text: str | None, default):
        try:
            return json.loads(text) if text else default
        except (json.JSONDecodeError, TypeError):
            return default

    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if (ts := obj.get(key)) is not None:
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            row = await conn.execute_fetchone(
                """
                SELECT goal_id, workflow_id, parent_goal_id, description, status,
                       created_at, updated_at, completed_at,
                       priority, reasoning, acceptance_criteria, metadata
                FROM goals
                WHERE goal_id = ?
                """,
                (goal_id,),
            )
            if row is None:
                raise ToolInputError(f"Goal '{goal_id}' not found.", param_name="goal_id")

        goal: Dict[str, Any] = dict(row)
        goal["acceptance_criteria"] = _safe_json(goal.get("acceptance_criteria"), [])
        goal["metadata"] = _safe_json(goal.get("metadata"), {})

        # keep raw ints, add ISO companions
        for k in ("created_at", "updated_at", "completed_at"):
            _add_iso(goal, k)

        elapsed = time.time() - t0
        logger.info(f"Goal '{_fmt_id(goal_id)}' loaded.", time=elapsed)

        return {"goal": goal, "success": True, "processing_time": elapsed}

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_goal_details({_fmt_id(goal_id)}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to retrieve goal details: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def create_thought_chain(
    workflow_id: str,
    *,
    title: str,
    initial_thought_content: str | None = None,
    initial_thought_type: str = ThoughtType.GOAL.value,
    idempotency_key: Optional[str] = None,  # NEW
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    if not workflow_id:
        raise ToolInputError("workflow_id is required.", param_name="workflow_id")
    if not title:
        raise ToolInputError("title is required.", param_name="title")
    if initial_thought_content:
        try:
            _ = ThoughtType(initial_thought_type.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"initial_thought_type must be one of {', '.join(t.value for t in ThoughtType)}",
                param_name="initial_thought_type",
            ) from exc

    now = int(time.time())
    start_perf_t = time.perf_counter()
    db = DBConnection(db_path)

    async def _fetch_existing_chain_details(
        conn_fetch: aiosqlite.Connection, existing_chain_id: str, wf_id: str
    ) -> Dict[str, Any]:
        # Reusing get_thought_chain logic, simplified
        chain_row = await conn_fetch.execute_fetchone(
            "SELECT * FROM thought_chains WHERE thought_chain_id = ?", (existing_chain_id,)
        )
        if not chain_row:
            raise ToolError(
                f"Failed to re-fetch existing thought_chain {existing_chain_id} on idempotency hit."
            )

        chain_data = dict(chain_row)
        # Fetch initial thought if it was supposed to be created
        initial_thought_id_existing = None
        if initial_thought_content:  # Check if the original call intended an initial thought
            thought_row = await conn_fetch.execute_fetchone(
                "SELECT thought_id FROM thoughts WHERE thought_chain_id = ? ORDER BY sequence_number ASC LIMIT 1",
                (existing_chain_id,),
            )
            if thought_row:
                initial_thought_id_existing = thought_row["thought_id"]

        return {
            "thought_chain_id": existing_chain_id,
            "workflow_id": wf_id,
            "title": chain_data["title"],
            "created_at_unix": chain_data["created_at"],
            "created_at_iso": safe_format_timestamp(chain_data["created_at"]),
            "initial_thought_id": initial_thought_id_existing,
            "idempotency_hit": True,
            "success": True,
            "processing_time": time.perf_counter() - start_perf_t,
        }

    if idempotency_key:
        async with db.transaction(readonly=True) as conn_check:
            existing_chain_row = await conn_check.execute_fetchone(
                "SELECT thought_chain_id FROM thought_chains WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            )
        if existing_chain_row:
            existing_chain_id = existing_chain_row["thought_chain_id"]
            logger.info(
                f"Idempotency hit for create_thought_chain (key='{idempotency_key}'). Returning existing chain {_fmt_id(existing_chain_id)}."
            )
            async with db.transaction(readonly=True) as conn_details:
                return await _fetch_existing_chain_details(
                    conn_details, existing_chain_id, workflow_id
                )

    chain_id_new = MemoryUtils.generate_id()
    initial_th_id_new: str | None = None

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ... (workflow existence check remains) ...
        if not await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ):
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        await conn.execute(
            "INSERT INTO thought_chains (thought_chain_id, workflow_id, title, created_at, idempotency_key) VALUES (?,?,?,?,?)",  # MODIFIED
            (chain_id_new, workflow_id, title, now, idempotency_key),  # MODIFIED
        )
        await conn.execute(
            "UPDATE workflows SET updated_at=?, last_active=? WHERE workflow_id=?",
            (now, now, workflow_id),
        )

        if initial_thought_content:
            # Here, record_thought is called. We assume the agent might generate a *new* idempotency key
            # for this specific thought if it wants that thought to be idempotent independently.
            # If not, it's created as a new thought within this chain.
            th_res = await record_thought(
                workflow_id=workflow_id,
                content=initial_thought_content,
                thought_type=initial_thought_type,
                thought_chain_id=chain_id_new,
                db_path=db_path,
                conn=conn,
                idempotency_key=None,  # Explicitly None for default behavior
            )
            if not th_res.get("success"):
                raise ToolError(th_res.get("error", "Failed to create initial thought"))
            initial_th_id_new = th_res["thought_id"]

    result = {
        "thought_chain_id": chain_id_new,
        "workflow_id": workflow_id,
        "title": title,
        "created_at_unix": now,
        "created_at_iso": safe_format_timestamp(now),
        "initial_thought_id": initial_th_id_new,
        "idempotency_hit": False,  # NEW
        "success": True,
        "processing_time": round(time.perf_counter() - start_perf_t, 4),
    }
    logger.info(
        f"New thought-chain {_fmt_id(chain_id_new)} for workflow {_fmt_id(workflow_id)}{' with seed ' + _fmt_id(initial_th_id_new) if initial_th_id_new else ''}.",
        emoji_key="thought_balloon",
    )
    return result


@with_tool_metrics
@with_error_handling
async def get_thought_chain(
    thought_chain_id: str,
    *,
    include_thoughts: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch a single thought–chain plus (optionally) its ordered thoughts.

    • Raw integer timestamps are preserved.
    • ISO companions are added as *_iso for human consumption.
    """
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")

    db = DBConnection(db_path)

    def _add_iso(obj: Dict[str, Any], key: str) -> None:
        if (ts := obj.get(key)) is not None:
            obj[f"{key}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(readonly=True) as conn:
            # ───────── Chain row ─────────
            row = await conn.execute_fetchone(
                "SELECT * FROM thought_chains WHERE thought_chain_id = ?",
                (thought_chain_id,),
            )
            if row is None:
                raise ToolInputError(
                    f"Thought chain {thought_chain_id} not found.",
                    param_name="thought_chain_id",
                )

            chain: Dict[str, Any] = dict(row)
            _add_iso(chain, "created_at")

            # ───────── Thoughts ─────────
            chain["thoughts"] = []
            if include_thoughts:
                async with conn.execute(
                    """
                    SELECT *
                    FROM thoughts
                    WHERE thought_chain_id = ?
                    ORDER BY sequence_number
                    """,
                    (thought_chain_id,),
                ) as cur:
                    async for t in cur:
                        th = dict(t)
                        _add_iso(th, "created_at")
                        chain["thoughts"].append(th)

            chain["success"] = True
            logger.info(
                f"Retrieved thought chain {thought_chain_id} ({len(chain['thoughts'])} thoughts)",
                emoji_key="left_speech_bubble",
            )
            return chain

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"get_thought_chain({thought_chain_id}) failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to get thought chain: {exc}") from exc


# ======================================================
# Helper Function for Working Memory Management
# ======================================================


async def _add_to_active_memories(
    conn: aiosqlite.Connection,
    context_id: str,
    memory_id: str,
) -> bool:
    """
    Add *memory_id* to the working-memory list for *context_id*,
    enforcing `agent_memory_config.max_working_memory_size`.

    • Uses the `compute_memory_relevance` UDF in SQL to find the
      least-relevant memory, avoiding Python-side sorting.
    • No cursors are leaked: every `conn.execute` is wrapped in
      an async-with or uses `execute_fetch*` helpers.
    • All existing logging, MemoryUtils serialisation, and operation
      logging behaviour is preserved.
    """
    try:
        # ───────────────────────── fetch current state ─────────────────────────
        row = await conn.execute_fetchone(
            "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        )
        if row is None:
            logger.warning(f"Context {context_id} not found when adding {memory_id}.")
            return False

        workflow_id = row["workflow_id"]
        current_ids = await MemoryUtils.deserialize(row["working_memory"]) or []

        # ───────────────────────── fast exits ─────────────────────────
        if memory_id in current_ids:
            logger.debug(f"{memory_id} already present in context {context_id}.")
            return True

        mem_exists = await conn.execute_fetchone(
            "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
        )
        if mem_exists is None:
            logger.warning(f"Memory {memory_id} missing; cannot add to context {context_id}.")
            return False

        # ───────────────────────── enforce capacity ─────────────────────────
        removed_id: str | None = None
        limit = agent_memory_config.max_working_memory_size

        if len(current_ids) >= limit and current_ids:
            # Use SQL to pick the least-relevant memory directly.
            placeholders = ",".join("?" * len(current_ids))
            least_row = await conn.execute_fetchone(
                f"""
                SELECT memory_id
                FROM memories
                WHERE memory_id IN ({placeholders})
                ORDER BY compute_memory_relevance(
                    importance, confidence, created_at,
                    IFNULL(access_count,0), last_accessed
                ) ASC
                LIMIT 1
                """,
                current_ids,
            )
            if least_row is None:
                logger.warning(
                    f"Could not evaluate relevance for context {context_id}; aborting add."
                )
                return False

            removed_id = least_row["memory_id"]
            if removed_id in current_ids:
                current_ids.remove(removed_id)

                await MemoryUtils._log_memory_operation(
                    conn,
                    workflow_id,
                    "remove_from_working",
                    removed_id,
                    None,
                    {
                        "context_id": context_id,
                        "reason": "working_memory_limit",
                    },
                )
                logger.debug(f"Removed {removed_id} from context {context_id} (capacity).")

        # ───────────────────────── append new memory ─────────────────────────
        current_ids.append(memory_id)

        await conn.execute(
            "UPDATE cognitive_states SET working_memory = ?, last_active = ? WHERE state_id = ?",
            (
                await MemoryUtils.serialize(current_ids),
                int(time.time()),
                context_id,
            ),
        )

        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "add_to_working",
            memory_id,
            None,
            {"context_id": context_id},
        )

        logger.debug(
            f"Added {memory_id} to working memory for context {context_id}; "
            f"size={len(current_ids)}/{limit}"
        )
        return True

    except Exception as exc:
        logger.error(
            f"_add_to_active_memories({context_id}, {memory_id}) failed: {exc}",
            exc_info=True,
        )
        return False


# --- 12. Working Memory Management ---
@with_tool_metrics
@with_error_handling
async def get_working_memory(
    context_id: str,
    *,
    include_content: bool = True,
    include_links: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, any]:
    """
    Return the current working-memory set for *context_id*.

    • Keeps every existing feature (content toggle, link toggle, access logging).
    • Uses `DBConnection.transaction(mode='IMMEDIATE')` for a short, single write-lock.
    • Does **not** overwrite integer timestamps; adds ISO companions instead.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")

    t0 = time.time()
    result: Dict[str, any] = dict(
        context_id=context_id,
        workflow_id=None,
        focal_memory_id=None,
        working_memories=[],
        success=True,
        processing_time=0.0,
    )

    db = DBConnection(db_path)

    # small helper
    def _add_iso(obj: Dict[str, any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    try:
        async with db.transaction(mode="IMMEDIATE") as conn:
            # ───────── 1. fetch cognitive state ─────────
            state_row = await conn.execute_fetchone(
                "SELECT * FROM cognitive_states WHERE state_id=?", (context_id,)
            )
            if state_row is None:
                result["processing_time"] = time.time() - t0
                logger.warning("Context %s not found; returning empty set.", context_id)
                return result

            wf_id = state_row["workflow_id"]
            result["workflow_id"] = wf_id
            result["focal_memory_id"] = state_row["focal_memory_id"]

            mem_ids: List[str] = await MemoryUtils.deserialize(state_row["working_memory"]) or []

            if not (mem_ids and wf_id):
                result["processing_time"] = time.time() - t0
                return result  # nothing to do

            # ───────── 2. fetch memory rows ─────────
            cols = [
                "memory_id",
                "workflow_id",
                "description",
                "memory_type",
                "memory_level",
                "importance",
                "confidence",
                "created_at",
                "updated_at",
                "last_accessed",
                "tags",
                "action_id",
                "thought_id",
                "artifact_id",
                "reasoning",
                "source",
                "context",
                "access_count",
                "ttl",
                "embedding_id",
            ]
            if include_content:
                cols.append("content")

            placeholders = ",".join("?" * len(mem_ids))
            mem_rows = await conn.execute_fetchall(
                f"SELECT {', '.join(cols)} FROM memories "
                f"WHERE memory_id IN ({placeholders}) AND workflow_id=?",
                (*mem_ids, wf_id),
            )

            mem_map: Dict[str, Dict[str, any]] = {}
            for r in mem_rows:
                m = dict(r)
                # JSON fields
                m["tags"] = await MemoryUtils.deserialize(m.get("tags"))
                m["context"] = await MemoryUtils.deserialize(m.get("context"))
                # preview
                if include_content and m.get("content") and len(m["content"]) > 150:
                    m["content_preview"] = m["content"][:147] + "…"
                # ISO companions
                _add_iso(m, ["created_at", "updated_at", "last_accessed"])
                mem_map[m["memory_id"]] = m

            # ───────── 3. links (optional) ─────────
            if include_links and mem_map:
                link_ph = ",".join("?" * len(mem_ids))
                link_rows = await conn.execute_fetchall(
                    f"""
                    SELECT ml.source_memory_id, ml.target_memory_id, ml.link_type,
                           ml.strength, ml.description AS link_description, ml.link_id,
                           ml.created_at,
                           tm.description AS target_description, tm.memory_type AS target_type
                    FROM memory_links ml
                    JOIN memories tm ON tm.memory_id = ml.target_memory_id
                    WHERE ml.source_memory_id IN ({link_ph}) AND tm.workflow_id = ?
                    """,
                    (*mem_ids, wf_id),
                )

                by_src: defaultdict[str, List[dict]] = defaultdict(list)
                for lr in link_rows:
                    row = dict(lr)
                    _add_iso(row, ["created_at"])
                    by_src[row["source_memory_id"]].append(row)

                for mid in mem_map:
                    mem_map[mid]["links"] = {"outgoing": by_src.get(mid, [])}

            # ───────── 4. ordered list + access logging ─────────
            ordered: List[Dict[str, any]] = []
            upd_params: List[Tuple[int, str, str]] = []  # (ts, mem_id, wf_id)
            now_ts = int(time.time())
            for mid in mem_ids:
                if mid in mem_map:
                    ordered.append(mem_map[mid])
                    upd_params.append((now_ts, mid, wf_id))

            if upd_params:
                await conn.executemany(
                    """
                    UPDATE memories
                    SET last_accessed=?, access_count=COALESCE(access_count,0)+1
                    WHERE memory_id=? AND workflow_id=?""",
                    upd_params,
                )
                # log each access
                for _, mem_id, wf in upd_params:
                    await MemoryUtils._log_memory_operation(
                        conn,
                        wf,
                        "access_working",
                        mem_id,
                        None,
                        {"context_id": context_id},
                    )

            result["working_memories"] = ordered

        result["processing_time"] = time.time() - t0
        logger.info(
            f"Working memory for {context_id} returned ({len(ordered)} items).",
            emoji_key="brain",
            time=result["processing_time"],
        )
        return result

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_working_memory(%s) failed: %s", context_id, exc, exc_info=True)
        raise ToolError(f"Failed to get working memory: {exc}") from exc


@with_tool_metrics
@with_error_handling
async def focus_memory(
    memory_id: str,
    context_id: str,
    *,
    add_to_working: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Make *memory_id* the focal memory for *context_id*.

    • Verifies that both rows exist and belong to the same workflow.
    • Optionally pushes the memory into the context’s working-memory list.
    • Updates `last_active` and logs the operation.
    • Leaves a clear audit trail in `memory_operations`.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")

    t0 = time.time()
    added_to_wm = False

    db = DBConnection(db_path)
    try:
        # one R/W IMMEDIATE transaction (default) → fast, WAL-friendly
        async with db.transaction() as conn:
            # ───────── validate memory row ─────────
            async with conn.execute(
                "SELECT workflow_id FROM memories WHERE memory_id = ?",
                (memory_id,),
            ) as cur:
                mem_row = await cur.fetchone()
            if mem_row is None:
                raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")
            mem_wf = mem_row["workflow_id"]

            # ───────── validate context row ───────
            async with conn.execute(
                "SELECT workflow_id FROM cognitive_states WHERE state_id = ?",
                (context_id,),
            ) as cur:
                ctx_row = await cur.fetchone()
            if ctx_row is None:
                raise ToolInputError(f"Context {context_id} not found.", param_name="context_id")
            ctx_wf = ctx_row["workflow_id"]

            if mem_wf != ctx_wf:
                raise ToolInputError(
                    f"Memory {_fmt_id(memory_id)} belongs to workflow {_fmt_id(mem_wf)}, "
                    f"not {_fmt_id(ctx_wf)} of context {context_id}"
                )

            # ───────── optionally push to WM ───────
            if add_to_working:
                added_to_wm = await _add_to_active_memories(conn, context_id, memory_id)
                if not added_to_wm:
                    logger.warning(
                        f"focus_memory: could not add {_fmt_id(memory_id)} to working-memory set."
                    )

            # ───────── set focal memory ────────────
            now = int(time.time())
            await conn.execute(
                "UPDATE cognitive_states "
                "SET focal_memory_id = ?, last_active = ? "
                "WHERE state_id = ?",
                (memory_id, now, context_id),
            )

            # ───────── audit log ───────────────────
            await MemoryUtils._log_memory_operation(
                conn,
                mem_wf,
                "focus",
                memory_id,
                None,
                {"context_id": context_id},
            )
            # commit happens on context-manager exit

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error(f"focus_memory failed: {exc}", exc_info=True)
        raise ToolError(f"Unable to focus memory: {exc}") from exc

    duration = time.time() - t0
    logger.info(
        f"Memory {_fmt_id(memory_id)} is now focus for context {context_id} "
        f"(added_to_WM={added_to_wm})",
        emoji_key="target",
    )
    return {
        "context_id": context_id,
        "focused_memory_id": memory_id,
        "workflow_id": mem_wf,
        "added_to_working": added_to_wm,
        "success": True,
        "processing_time": duration,
    }


@with_tool_metrics
@with_error_handling
async def optimize_working_memory(
    context_id: str,
    *,
    target_size: int = agent_memory_config.max_working_memory_size,
    strategy: str = "balanced",  # balanced | importance | recency | diversity
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if not isinstance(target_size, int) or target_size < 0:
        raise ToolInputError("Target size must be a non-negative integer.", param_name="target_size")
    strategies = {"balanced", "importance", "recency", "diversity"}
    if strategy not in strategies:
        raise ToolInputError(
            f"Strategy must be one of: {', '.join(sorted(strategies))}", param_name="strategy"
        )

    t0 = time.time()
    db = DBConnection(db_path)

    # ---- Phase 1: Ensure context exists (strict: fail if missing) ----
    async with db.transaction(readonly=True) as conn:
        state = await conn.execute_fetchone(
            "SELECT workflow_id, working_memory FROM cognitive_states WHERE state_id = ?",
            (context_id,),
        )
        if state is None:
            logger.error(
                "INVARIANT VIOLATION: Context '%s' not found in cognitive_states! This indicates a workflow/context provisioning bug.",
                context_id
            )
            raise ToolError(f"Context '{context_id}' not found in cognitive_states. This is a fatal agent/UMS contract violation.")
        workflow_id = state["workflow_id"]
        wm_ids: list[str] = await MemoryUtils.deserialize(state["working_memory"]) or []
        before_count = len(wm_ids)

    # If the context is empty, there is nothing to optimize
    if before_count == 0:
        after = {
            "context_id": context_id,
            "workflow_id": workflow_id,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": 0,
            "after_count": 0,
            "removed_count": 0,
            "retained_memories": [],
            "removed_memories": [],
            "success": True,
            "processing_time": time.time() - t0,
        }
        return after

    # ---- Phase 2: Early exit if already <= target_size ----
    if before_count <= target_size:
        await _log_optimization_event(
            db,
            workflow_id,
            op="calculate_wm_optimization_skipped",
            payload={
                "context_id": context_id,
                "strategy": strategy,
                "target_size": target_size,
                "before_count": before_count,
                "reason": "already_optimal_size",
            },
        )
        return {
            "context_id": context_id,
            "workflow_id": workflow_id,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": before_count,
            "after_count": before_count,
            "removed_count": 0,
            "retained_memories": wm_ids,
            "removed_memories": [],
            "success": True,
            "processing_time": time.time() - t0,
        }

    # ---- Phase 3: Fetch memory rows needed for scoring ----
    async with db.transaction(readonly=True) as conn:
        placeholders = ", ".join("?" * len(wm_ids))
        mem_rows = await conn.execute_fetchall(
            f"""
            SELECT memory_id,
                   memory_type,
                   importance,
                   confidence,
                   created_at,
                   last_accessed,
                   access_count
            FROM memories
            WHERE memory_id IN ({placeholders})
              AND workflow_id = ?
            """,
            (*wm_ids, workflow_id),
        )

    if not mem_rows:
        await _log_optimization_event(
            db,
            workflow_id,
            op="calculate_wm_optimization_failed_fetch",
            payload={
                "context_id": context_id,
                "strategy": strategy,
                "target_size": target_size,
                "before_count": before_count,
                "reason": "failed_to_fetch_memory_details",
            },
        )
        return {
            "context_id": context_id,
            "workflow_id": workflow_id,
            "strategy_used": strategy,
            "target_size": target_size,
            "before_count": before_count,
            "after_count": 0,
            "removed_count": before_count,
            "retained_memories": [],
            "removed_memories": wm_ids,
            "success": True,
            "processing_time": time.time() - t0,
        }

    # ---- Phase 4: Score & select memories to retain ----
    now = int(time.time())
    scored: list[dict] = []
    for row in mem_rows:
        rel = _compute_memory_relevance(
            row["importance"],
            row["confidence"],
            row["created_at"],
            row["access_count"],
            row["last_accessed"],
        )
        recency = 1.0 / (1.0 + (now - (row["last_accessed"] or row["created_at"])) / 86_400)
        if strategy == "balanced":
            score = rel
        elif strategy == "importance":
            score = row["importance"] * 0.6 + row["confidence"] * 0.2 + rel * 0.1 + recency * 0.1
        elif strategy == "recency":
            score = recency * 0.5 + min(1.0, row["access_count"] / 5.0) * 0.2 + rel * 0.3
        else:  # diversity
            score = rel
        scored.append({"id": row["memory_id"], "score": score, "type": row["memory_type"]})

    retained_ids: list[str] = []
    if strategy == "diversity":
        from collections import defaultdict
        buckets: dict[str, list[dict]] = defaultdict(list)
        for m in scored:
            buckets[m["type"]].append(m)
        for lst in buckets.values():
            lst.sort(key=lambda d: d["score"], reverse=True)
        iters = {t: iter(lst) for t, lst in buckets.items()}
        active = list(iters.keys())
        while len(retained_ids) < target_size and active:
            t = active.pop(0)
            try:
                retained_ids.append(next(iters[t])["id"])
                active.append(t)
            except StopIteration:
                pass
    else:
        scored.sort(key=lambda d: d["score"], reverse=True)
        retained_ids = [m["id"] for m in scored[:target_size]]

    removed_ids = list(set(wm_ids) - set(retained_ids))

    # ---- Phase 5: Write new working memory set if changed ----
    if set(retained_ids) != set(wm_ids):
        async with db.transaction(mode="IMMEDIATE") as conn:
            await conn.execute(
                "UPDATE cognitive_states SET working_memory = ?, last_active = ? WHERE state_id = ?",
                (json.dumps(retained_ids), int(time.time()), context_id),
            )

    # ---- Phase 6: Log optimization ----
    await _log_optimization_event(
        db,
        workflow_id,
        op="calculate_wm_optimization",
        payload={
            "context_id": context_id,
            "strategy": strategy,
            "target_size": target_size,
            "before_count": before_count,
            "after_count": len(retained_ids),
            "removed_count": len(removed_ids),
            "retained_ids_sample": retained_ids[:5],
            "removed_ids_sample": removed_ids[:5],
        },
    )

    return {
        "context_id": context_id,
        "workflow_id": workflow_id,
        "strategy_used": strategy,
        "target_size": target_size,
        "before_count": before_count,
        "after_count": len(retained_ids),
        "removed_count": len(removed_ids),
        "retained_memories": retained_ids,
        "removed_memories": removed_ids,
        "success": True,
        "processing_time": time.time() - t0,
    }



# ───────────────────── helper : single logging call ─────────────────────
async def _log_optimization_event(
    db: DBConnection,
    workflow_id: str | None,
    *,
    op: str,
    payload: dict,
) -> None:
    """
    Write a single memory_operations row and run a PASSIVE checkpoint.
    Never raises: logging failure must not break the caller.
    """
    if workflow_id is None:
        return
    try:
        async with db.transaction() as conn:
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                op,
                None,
                None,
                payload,
            )
    except Exception as exc:
        logger.error(f"Unable to log WM optimisation event '{op}': {exc}", exc_info=True)


# --- 13. Cognitive State Persistence ---
@with_tool_metrics
@with_error_handling
async def save_cognitive_state(
    workflow_id: str,
    title: str,
    working_memory_ids: list[str],
    *,
    focus_area_ids: list[str] | None = None,
    context_action_ids: list[str] | None = None,
    current_goal_thought_ids: list[str] | None = None,
    db_path: str = agent_memory_config.db_path,
) -> dict[str, Any]:
    """
    Persist the agent’s *latest* cognitive snapshot.

    Validation guarantees referenced memories / actions / thoughts belong
    to the same workflow. Older states for the same workflow are automatically
    marked `is_latest = False`.
    """
    if not title:
        raise ToolInputError("State title required.", param_name="title")
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    state_id = MemoryUtils.generate_id() # Each save creates a new state_id
    now_unix = int(time.time())
    t0 = time.time()

    # Gather all memory IDs mentioned in focus_areas or working_memory for validation
    all_memory_ids_in_state: set[str] = set(working_memory_ids or [])
    if focus_area_ids:
        all_memory_ids_in_state.update(focus_area_ids)

    # Gather action IDs and thought IDs for validation
    all_action_ids_in_state: set[str] = set(context_action_ids or [])
    all_thought_ids_in_state: set[str] = set(current_goal_thought_ids or [])

    db = DBConnection(db_path)

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ───── 1. Validate Workflow Existence ─────
        workflow_exists_row = await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
        )
        if not workflow_exists_row:
            raise ToolInputError(
                f"Workflow {_fmt_id(workflow_id)} not found. Cannot save cognitive state.",
                param_name="workflow_id",
            )

        # ───── 2. Validate Memory Ownership ─────
        if all_memory_ids_in_state:
            # Query for memories that are in the set AND belong to the workflow_id
            placeholders_mem = ",".join("?" * len(all_memory_ids_in_state))
            valid_mem_rows = await conn.execute_fetchall(
                f"SELECT memory_id FROM memories "
                f"WHERE memory_id IN ({placeholders_mem}) AND workflow_id = ?",
                (*all_memory_ids_in_state, workflow_id),
            )
            valid_mem_ids_found = {r["memory_id"] for r in valid_mem_rows}
            missing_or_mismatched_mem_ids = all_memory_ids_in_state - valid_mem_ids_found
            if missing_or_mismatched_mem_ids:
                sample_missing_mem = ", ".join(map(_fmt_id, list(missing_or_mismatched_mem_ids)[:5])) + (
                    "…" if len(missing_or_mismatched_mem_ids) > 5 else ""
                )
                raise ToolInputError(
                    f"Memory IDs not found or do not belong to workflow '{_fmt_id(workflow_id)}': {sample_missing_mem}",
                    param_name="working_memory_ids/focus_area_ids",
                )

        # ───── 3. Validate Action Ownership ─────
        if all_action_ids_in_state:
            placeholders_act = ",".join("?" * len(all_action_ids_in_state))
            valid_act_rows = await conn.execute_fetchall(
                f"SELECT action_id FROM actions "
                f"WHERE action_id IN ({placeholders_act}) AND workflow_id = ?",
                (*all_action_ids_in_state, workflow_id),
            )
            valid_act_ids_found = {r["action_id"] for r in valid_act_rows}
            missing_or_mismatched_act_ids = all_action_ids_in_state - valid_act_ids_found
            if missing_or_mismatched_act_ids:
                sample_missing_act = ", ".join(map(_fmt_id, list(missing_or_mismatched_act_ids)[:5])) + (
                    "…" if len(missing_or_mismatched_act_ids) > 5 else ""
                )
                raise ToolInputError(
                    f"Action IDs not found or do not belong to workflow '{_fmt_id(workflow_id)}': {sample_missing_act}",
                    param_name="context_action_ids",
                )

        # ───── 4. Validate Thought Ownership ─────
        if all_thought_ids_in_state:
            placeholders_th = ",".join("?" * len(all_thought_ids_in_state))
            valid_th_rows = await conn.execute_fetchall(
                f"""
                SELECT t.thought_id
                FROM thoughts t
                JOIN thought_chains tc ON t.thought_chain_id = tc.thought_chain_id
                WHERE t.thought_id IN ({placeholders_th}) AND tc.workflow_id = ?
                """,
                (*all_thought_ids_in_state, workflow_id),
            )
            valid_th_ids_found = {r["thought_id"] for r in valid_th_rows}
            missing_or_mismatched_th_ids = all_thought_ids_in_state - valid_th_ids_found
            if missing_or_mismatched_th_ids:
                sample_missing_th = ", ".join(map(_fmt_id, list(missing_or_mismatched_th_ids)[:5])) + (
                    "…" if len(missing_or_mismatched_th_ids) > 5 else ""
                )
                raise ToolInputError(
                    f"Thought IDs not found or do not belong to workflow '{_fmt_id(workflow_id)}': {sample_missing_th}",
                    param_name="current_goal_thought_ids",
                )

        # ───── 5. Write New Cognitive State ─────
        # Mark all existing states for this workflow_id as not the latest
        await conn.execute(
            "UPDATE cognitive_states SET is_latest = 0 WHERE workflow_id = ?",
            (workflow_id,),
        )

        # Serialize ID lists to JSON (sorted for deterministic storage if that matters)
        wm_json = await MemoryUtils.serialize(sorted(list(working_memory_ids or [])))
        fa_json = await MemoryUtils.serialize(sorted(list(focus_area_ids or [])))
        ca_json = await MemoryUtils.serialize(sorted(list(context_action_ids or [])))
        cg_json = await MemoryUtils.serialize(sorted(list(current_goal_thought_ids or [])))

        # Determine focal memory ID: first from focus_areas, then working_memory, else None
        focal_memory_id_selected = (focus_area_ids or working_memory_ids or [None])[0]

        # Insert the new cognitive state, marked as the latest
        await conn.execute(
            """
            INSERT INTO cognitive_states (
                state_id, workflow_id, title,
                working_memory, focus_areas, context_actions, current_goals,
                created_at, is_latest, focal_memory_id, last_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state_id,
                workflow_id,
                title,
                wm_json,
                fa_json,
                ca_json,
                cg_json,
                now_unix,  # created_at
                1,  # is_latest = True
                focal_memory_id_selected,
                now_unix,  # last_active
            ),
        )

        # Update parent workflow's timestamps
        await conn.execute(
            "UPDATE workflows SET updated_at = ?, last_active = ? WHERE workflow_id = ?",
            (now_unix, now_unix, workflow_id),
        )

        # Log the operation
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "save_state",
            memory_id=None,  # Not directly related to one memory, but a state
            action_id=None,
            operation_data={
                "state_id": state_id,
                "title": title,
                "working_memory_count": len(working_memory_ids or []),
                "focus_areas_count": len(focus_area_ids or []),
                "context_actions_count": len(context_action_ids or []),
                "current_goals_count": len(current_goal_thought_ids or []),
                "focal_memory_id_used": _fmt_id(focal_memory_id_selected) if focal_memory_id_selected else "None",
            },
        )

    processing_time_seconds = time.time() - t0
    result = {
        "state_id": state_id,
        "workflow_id": workflow_id,  # Explicitly include workflow_id as requested
        "title": title,
        "created_at": to_iso_z(now_unix), # Use consistent ISO Z format
        "processing_time": processing_time_seconds,
        "success": True,
    }
    logger.info(
        f"Saved cognitive state '{title}' (ID: {_fmt_id(state_id)}) for workflow '{_fmt_id(workflow_id)}'.",
        emoji_key="save", # Assuming 'save' is a valid emoji key in your logger
        time=processing_time_seconds, # Pass processing time to logger if it supports it
    )
    return result


@with_tool_metrics
@with_error_handling
async def load_cognitive_state(
    workflow_id: str,
    *,
    state_id: str | None = None,  # None → latest
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Retrieve a saved cognitive-state snapshot.

    • Uses a *read-only* snapshot transaction (no WAL churn).
    • If *state_id* is omitted, picks `is_latest=1` else the newest `created_at`.
    • Logs the load in *memory_operations* (separate write txn).
    • Returns `created_at_unix` + `created_at_iso`.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0 = time.time()
    db = DBConnection(db_path)

    # ────────────────────────── read phase (snapshot, RO) ──────────────────────────
    async with db.transaction(readonly=True) as conn:
        # Ensure workflow exists
        exists = await conn.execute_fetchone(
            "SELECT 1 FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )
        if not exists:
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        sql = ["SELECT * FROM cognitive_states WHERE workflow_id = ?"]
        params: list[Any] = [workflow_id]

        if state_id:
            sql.append("AND state_id = ?")
            params.append(state_id)
        else:
            sql.append("ORDER BY is_latest DESC, created_at DESC LIMIT 1")

        row = await conn.execute_fetchone(" ".join(sql), tuple(params))

    # ────────────────────────── not found → graceful empty payload ─────────────────
    if row is None:
        msg = (
            f"State {state_id} not found."
            if state_id
            else f"No states found for workflow {workflow_id}."
        )
        logger.warning(f"load_cognitive_state: {msg}")
        return {
            "state_id": None,
            "workflow_id": workflow_id,
            "title": None,
            "working_memory_ids": [],
            "focus_areas": [],
            "context_action_ids": [],
            "current_goals": [],
            "created_at_unix": None,
            "created_at_iso": None,
            "focal_memory_id": None,
            "success": True,
            "message": msg,
            "processing_time": time.time() - t0,
        }

    state = dict(row)  # still holds raw ints
    created_ts = state["created_at"]

    # ────────────────────────── write phase – log operation ───────────────────────
    async with db.transaction() as wconn:
        await MemoryUtils._log_memory_operation(
            wconn,
            workflow_id,
            "load_state",
            None,
            None,
            {"state_id": state["state_id"], "title": state["title"]},
        )

    # ────────────────────────── build response ────────────────────────────────────
    result: Dict[str, Any] = {
        "state_id": state["state_id"],
        "workflow_id": state["workflow_id"],
        "title": state["title"],
        "working_memory_ids": await MemoryUtils.deserialize(state.get("working_memory")) or [],
        "focus_areas": await MemoryUtils.deserialize(state.get("focus_areas")) or [],
        "context_action_ids": await MemoryUtils.deserialize(state.get("context_actions")) or [],
        "current_goals": await MemoryUtils.deserialize(state.get("current_goals")) or [],
        "created_at_unix": created_ts,
        "created_at_iso": safe_format_timestamp(created_ts) if created_ts else None,
        "focal_memory_id": state.get("focal_memory_id"),
        "success": True,
        "processing_time": time.time() - t0,
    }

    logger.info(
        f"Loaded cognitive state '{result['title']}' "
        f"({_fmt_id(result['state_id'])}) for workflow {_fmt_id(workflow_id)}",
        emoji_key="inbox_tray",
        time=result["processing_time"],
    )
    return result


# --- 14. Comprehensive Context Retrieval ---
@with_tool_metrics
@with_error_handling
async def get_workflow_context(
    workflow_id: str,
    *,
    recent_actions_limit: int = 10,
    important_memories_limit: int = 5,
    key_thoughts_limit: int = 5,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Compact snapshot of a workflow’s “working set”:
        latest cognitive state, most-recent actions, top memories, key thoughts.
    """
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    t0 = time.time()
    db = DBConnection(db_path)

    # tiny helper – decorate ISO only when we already return the int
    def _add_iso(obj: Dict[str, Any], k: str) -> None:
        if k in obj and obj[k] is not None:
            obj[f"{k}_iso"] = safe_format_timestamp(obj[k])

    try:
        # ───────── core info & thoughts use a single read-only snapshot ─────────
        async with db.transaction(readonly=True) as conn:
            wf = await conn.execute_fetchone(
                "SELECT title, goal, status FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
            if wf is None:
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            ctx: Dict[str, Any] = {
                "workflow_id": workflow_id,
                "workflow_title": wf["title"],
                "workflow_goal": wf["goal"],
                "workflow_status": wf["status"],
            }

            # ───────── latest cognitive state ─────────
            try:
                latest_state = await load_cognitive_state(
                    workflow_id=workflow_id, state_id=None, db_path=db_path
                )
                latest_state.pop("success", None)
                latest_state.pop("processing_time", None)
                ctx["latest_cognitive_state"] = latest_state
            except ToolInputError:
                ctx["latest_cognitive_state"] = None
            except Exception as exc:
                logger.warning("load_cognitive_state failed: %s", exc)
                ctx["latest_cognitive_state"] = {"error": str(exc)}

            # ───────── recent actions (truncated) ─────────
            try:
                ra = await get_recent_actions(
                    workflow_id=workflow_id,
                    limit=recent_actions_limit,
                    include_reasoning=False,
                    include_tool_results=False,
                    db_path=db_path,
                )
                ctx["recent_actions"] = ra.get("actions", [])
            except Exception as exc:
                logger.warning("get_recent_actions failed: %s", exc)
                ctx["recent_actions"] = [{"error": str(exc)}]

            # ───────── important memories (summary) ─────────
            try:
                mems = await query_memories(
                    workflow_id=workflow_id,
                    limit=important_memories_limit,
                    sort_by="importance",
                    sort_order="DESC",
                    include_content=False,
                    db_path=db_path,
                )
                ctx["important_memories"] = [
                    {
                        "memory_id": m["memory_id"],
                        "description": m.get("description"),
                        "memory_type": m.get("memory_type"),
                        "importance": m.get("importance"),
                    }
                    for m in mems.get("memories", [])
                ]
            except Exception as exc:
                logger.warning("query_memories failed: %s", exc)
                ctx["important_memories"] = [{"error": str(exc)}]

            # ───────── key thoughts ─────────
            try:
                chain_id_row = await conn.execute_fetchone(
                    "SELECT thought_chain_id "
                    "FROM thought_chains WHERE workflow_id = ? "
                    "ORDER BY created_at LIMIT 1",
                    (workflow_id,),
                )
                if chain_id_row:
                    thought_rows = await conn.execute_fetchall(
                        """
                        SELECT thought_type, content, sequence_number, created_at
                        FROM thoughts
                        WHERE thought_chain_id = ?
                          AND thought_type IN (?, ?, ?, ?)
                        ORDER BY sequence_number DESC
                        LIMIT ?
                        """,
                        (
                            chain_id_row["thought_chain_id"],
                            ThoughtType.GOAL.value,
                            ThoughtType.DECISION.value,
                            ThoughtType.SUMMARY.value,
                            ThoughtType.REFLECTION.value,
                            key_thoughts_limit,
                        ),
                    )
                    ctx["key_thoughts"] = [dict(r) for r in thought_rows]
                    for th in ctx["key_thoughts"]:
                        _add_iso(th, "created_at")
                else:
                    ctx["key_thoughts"] = []
            except Exception as exc:
                logger.warning("thought fetch failed: %s", exc)
                ctx["key_thoughts"] = [{"error": str(exc)}]

        ctx["success"] = True
        ctx["processing_time"] = time.time() - t0
        logger.info(
            f"Context summary for {workflow_id} ready",  # Message ends here
            time=ctx["processing_time"],  # Pass processing_time as 'time' kwarg
            # Add emoji_key if one is typically used here
        )
        return ctx

    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("get_workflow_context(%s) failed: %s", workflow_id, exc, exc_info=True)
        raise ToolError(f"Failed to get workflow context: {exc}") from exc


# --- Helper: Scoring for Focus ---


def _calculate_focus_score_internal_ums(
    memory: Dict[str, Any],
    recent_action_ids: list[str],
    now_unix: int,
) -> float:
    """
    Internal scoring function mirrored from the agent-side logic.

    Weighting:
        • 60 %  = decayed relevance (importance, confidence, recency, usage)
        • 30 %  = direct linkage to a recently executed action
        • 10 %  = heuristics for type / level
    """
    base_relevance = _compute_memory_relevance(
        memory.get("importance", 5.0),
        memory.get("confidence", 1.0),
        memory.get("created_at", now_unix),
        memory.get("access_count", 0),
        memory.get("last_accessed"),
    )

    score = base_relevance * 0.6

    if memory.get("action_id") in recent_action_ids:
        score += 3.0  # strong boost for immediate contextuality

    if memory.get("memory_type") in {
        MemoryType.QUESTION.value,
        MemoryType.PLAN.value,
        MemoryType.INSIGHT.value,
    }:
        score += 1.5

    lvl = memory.get("memory_level")
    if lvl == MemoryLevel.SEMANTIC.value:
        score += 0.5
    elif lvl == MemoryLevel.PROCEDURAL.value:
        score += 0.7

    return max(score, 0.0)


# --- Tool: Auto Update Focus ---
@with_tool_metrics
@with_error_handling
async def auto_update_focus(
    context_id: str,
    *,
    recent_actions_count: int = 3,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Strict version: Enforces that a valid cognitive_state row exists for context_id.
    Fails with a clear error if not, rather than auto-creating a row.
    """
    if not context_id:
        raise ToolInputError("Context ID required.", param_name="context_id")
    if recent_actions_count < 0:
        raise ToolInputError("Recent actions count must be ≥ 0.", param_name="recent_actions_count")

    t0 = time.time()
    db = DBConnection(db_path)

    async with db.transaction(readonly=True) as conn:
        state_row = await conn.execute_fetchone(
            """
            SELECT workflow_id,
                   focal_memory_id,
                   working_memory
            FROM   cognitive_states
            WHERE  state_id = ?
            """,
            (context_id,),
        )
        if state_row is None:
            logger.error("INVARIANT VIOLATION: Context %s not found in cognitive_states! This should never happen—fix agent/UMS code.", context_id)
            raise ToolError(f"Context '{context_id}' not found in cognitive_states. This indicates a serious bug in agent/UMS integration.")

        workflow_id: str = state_row["workflow_id"]
        prev_focal_id: str | None = state_row["focal_memory_id"]
        working_ids: list[str] = await MemoryUtils.deserialize(state_row["working_memory"]) or []

        if not working_ids:
            return {
                "context_id": context_id,
                "workflow_id": workflow_id,
                "previous_focal_memory_id": prev_focal_id,
                "new_focal_memory_id": None,
                "focus_changed": False,
                "reason": "Working memory empty.",
                "success": True,
                "processing_time": time.time() - t0,
            }

        placeholders = ",".join("?" * len(working_ids))
        mem_rows = await conn.execute_fetchall(
            f"""
            SELECT memory_id, action_id, memory_type, memory_level,
                   importance, confidence, created_at,
                   last_accessed, access_count
            FROM   memories
            WHERE  memory_id IN ({placeholders})
              AND  workflow_id = ?
            """,
            working_ids + [workflow_id],
        )

        recent_action_ids: list[str] = []
        if recent_actions_count:
            rows = await conn.execute_fetchall(
                """
                SELECT action_id
                FROM   actions
                WHERE  workflow_id = ?
                ORDER  BY sequence_number DESC
                LIMIT  ?
                """,
                (workflow_id, recent_actions_count),
            )
            recent_action_ids = [r["action_id"] for r in rows]

        now_unix = int(time.time())
        best_id: str | None = None
        best_score = -1.0
        for mem in mem_rows:
            s = _calculate_focus_score_internal_ums(mem, recent_action_ids, now_unix)
            logger.debug(
                "Focus-score %s → %.2f (context %s)", _fmt_id(mem["memory_id"]), s, context_id
            )
            if s > best_score:
                best_score = s
                best_id = mem["memory_id"]

    focus_changed = False
    reason: str
    if best_id and best_id != prev_focal_id:
        async with db.transaction(mode="IMMEDIATE") as conn:
            await conn.execute(
                """
                UPDATE cognitive_states
                SET    focal_memory_id = ?, last_active = ?
                WHERE  state_id = ?
                """,
                (best_id, now_unix, context_id),
            )
            focus_changed = True
            reason = f"Shifted focus to {_fmt_id(best_id)} – highest score {best_score:.2f}."

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "auto_focus_shift",
                best_id,
                None,
                {
                    "context_id": context_id,
                    "previous_focus": prev_focal_id,
                    "score": best_score,
                },
            )
    else:
        reason = (
            "No suitable candidate."
            if best_id is None
            else f"Focus remains on {_fmt_id(prev_focal_id)} (score {best_score:.2f})."
        )
        async with db.transaction(mode="IMMEDIATE") as conn:
            await conn.execute(
                "UPDATE cognitive_states SET last_active = ? WHERE state_id = ?",
                (now_unix, context_id),
            )

    return {
        "context_id": context_id,
        "workflow_id": workflow_id,
        "previous_focal_memory_id": prev_focal_id,
        "new_focal_memory_id": best_id,
        "focus_changed": focus_changed,
        "reason": reason,
        "success": True,
        "processing_time": time.time() - t0,
    }


# --- Tool: Promote Memory Level ---
@with_tool_metrics
@with_error_handling
async def promote_memory_level(
    memory_id: str,
    *,
    target_level: str | None = None,
    min_access_count_episodic: int = 5,
    min_confidence_episodic: float = 0.8,
    min_access_count_semantic: int = 10,
    min_confidence_semantic: float = 0.9,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Attempt to promote a memory’s cognitive level (episodic → semantic → procedural).

    • Read phase runs in a *read-only* transaction – zero WAL churn.
    • Write phase (if any) uses its own IMMEDIATE transaction.
    • Keeps full parity with prior behaviour: same criteria, same logging,
      same WAL checkpoint, same return structure.
    """
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")

    # ───────── validate explicit target (optional) ─────────
    explicit_target: MemoryLevel | None = None
    if target_level:
        try:
            explicit_target = MemoryLevel(target_level.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"Invalid target_level. Use one of: {', '.join(ml.value for ml in MemoryLevel)}",
                param_name="target_level",
            ) from exc

    db = DBConnection(db_path)
    start = time.time()

    # ───────── 1. fetch current memory row (RO) ─────────
    async with db.transaction(readonly=True) as conn:
        row = await conn.execute_fetchone(
            """
            SELECT workflow_id, memory_level, memory_type,
                   access_count, confidence, importance
            FROM   memories
            WHERE  memory_id = ?
            """,
            (memory_id,),
        )

    if row is None:
        raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

    current_level = MemoryLevel(row["memory_level"])
    mem_type = MemoryType(row["memory_type"])
    access_count = row["access_count"] or 0
    confidence = row["confidence"] or 0.0
    workflow_id = row["workflow_id"]

    # ───────── 2. decide promotion eligibility ─────────
    # default ladder
    auto_next = (
        MemoryLevel.SEMANTIC
        if current_level == MemoryLevel.EPISODIC
        else MemoryLevel.PROCEDURAL
        if current_level == MemoryLevel.SEMANTIC
        and mem_type in (MemoryType.PROCEDURE, MemoryType.SKILL)
        else None
    )
    candidate = explicit_target or auto_next

    promoted = False
    new_level = current_level
    reason = "Criteria not met or level already maximal."

    if candidate and candidate.value > current_level.value:
        if candidate == MemoryLevel.SEMANTIC:
            ok = access_count >= min_access_count_episodic and confidence >= min_confidence_episodic
            reason = (
                f"access_count {access_count}/{min_access_count_episodic}, "
                f"confidence {confidence:.2f}/{min_confidence_episodic}"
            )
        elif candidate == MemoryLevel.PROCEDURAL:
            ok = (
                mem_type in (MemoryType.PROCEDURE, MemoryType.SKILL)
                and access_count >= min_access_count_semantic
                and confidence >= min_confidence_semantic
            )
            reason = (
                f"type {mem_type.value}, "
                f"access_count {access_count}/{min_access_count_semantic}, "
                f"confidence {confidence:.2f}/{min_confidence_semantic}"
            )
        else:
            ok = False

        if ok:
            promoted = True
            new_level = candidate
            reason = f"Promoted: {reason}"
        else:
            reason = f"Not promoted: {reason}"
    elif candidate:
        reason = f"Already at or above {candidate.value}."

    # ───────── 3. apply update (RW) ─────────
    if promoted:
        async with db.transaction(mode="IMMEDIATE") as conn:
            now = int(time.time())
            await conn.execute(
                "UPDATE memories SET memory_level=?, updated_at=? WHERE memory_id=?",
                (new_level.value, now, memory_id),
            )
            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "promote_level",
                memory_id,
                None,
                {
                    "previous_level": current_level.value,
                    "new_level": new_level.value,
                    "reason": reason,
                },
            )

        logger.info(f"{memory_id}: {current_level.value} → {new_level.value}", emoji_key="arrow_up")
    else:
        logger.info(f"{memory_id} not promoted – {reason}")

    # ───────── 4. return ─────────
    return {
        "memory_id": memory_id,
        "promoted": promoted,
        "previous_level": current_level.value,
        "new_level": new_level.value if promoted else None,
        "reason": reason,
        "success": True,
        "processing_time": time.time() - start,
    }


# --- 15. Memory Update ---
@with_tool_metrics
@with_error_handling
async def update_memory(
    memory_id: str,
    *,
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    description: str | None = None,
    reasoning: str | None = None,
    tags: list[str] | None = None,
    ttl: int | None = None,
    memory_level: str | None = None,
    regenerate_embedding: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Patch any subset of a memory row and (optionally) regenerate its embedding.

    • All validations unchanged.
    • Uses `DBConnection.transaction(mode='IMMEDIATE')` for early write-lock.
    • Embedding regeneration works exactly as before.
    • WAL checkpoint now invoked with explicit db_path.
    """
    # ───────── validations ─────────
    if not memory_id:
        raise ToolInputError("Memory ID required.", param_name="memory_id")
    if importance is not None and not 1.0 <= importance <= 10.0:
        raise ToolInputError("Importance must be 1.0-10.0.", param_name="importance")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ToolInputError("Confidence must be 0.0-1.0.", param_name="confidence")

    # Normalise tags (empty list → clear; None → untouched)
    final_tags_json: str | None = None
    if tags is not None:
        final_tags_json = json.dumps(
            sorted({str(t).strip().lower() for t in tags if str(t).strip()})
        )

    # Validate / map memory_level
    memory_level_value: str | None = None
    if memory_level:
        try:
            memory_level_value = MemoryLevel(memory_level.lower()).value
        except ValueError as e:
            valid_levels = ", ".join(ml.value for ml in MemoryLevel)
            raise ToolInputError(
                f"Invalid memory_level. Must be one of: {valid_levels}",
                param_name="memory_level",
            ) from e

    # ───────── dynamic SET clause assembly ─────────
    update_clauses, params, touched = [], [], []  # sql pieces, values, audit list

    def _add(field: str, value: Any | None) -> None:
        if value is not None:
            update_clauses.append(f"{field} = ?")
            params.append(value)
            touched.append(field)

    _add("content", content)
    _add("importance", importance)
    _add("confidence", confidence)
    _add("description", description)
    _add("reasoning", reasoning)
    if final_tags_json is not None:
        _add("tags", final_tags_json)
    _add("ttl", ttl)
    if memory_level_value is not None:
        _add("memory_level", memory_level_value)

    if not update_clauses and not regenerate_embedding:
        raise ToolInputError(
            "No fields provided to update and regenerate_embedding is False.",
            param_name="content",
        )

    now_ts = int(time.time())
    if update_clauses:
        update_clauses.append("updated_at = ?")
        params.append(now_ts)

    embedding_regenerated = False
    new_embedding_id: str | None = None
    start_time = time.time()

    db = DBConnection(db_path)

    async with db.transaction(mode="IMMEDIATE") as conn:
        # ─── ensure memory exists and capture current text ───
        mem_row = await conn.execute_fetchone(
            "SELECT workflow_id, description, content FROM memories WHERE memory_id = ?",
            (memory_id,),
        )
        if mem_row is None:
            raise ToolInputError(f"Memory {memory_id} not found.", param_name="memory_id")

        workflow_id = mem_row["workflow_id"]
        db_desc, db_content = mem_row["description"], mem_row["content"]

        # ─── apply UPDATE if needed ───
        if update_clauses:
            await conn.execute(
                f"UPDATE memories SET {', '.join(update_clauses)} WHERE memory_id = ?",
                (*params, memory_id),
            )

        # ─── embedding regeneration ───
        if regenerate_embedding:
            eff_desc = description if "description" in touched else db_desc
            eff_content = content if "content" in touched else db_content
            text_for_embed = f"{eff_desc}: {eff_content}" if eff_desc else eff_content

            if text_for_embed:
                try:
                    new_embedding_id = await _store_embedding(conn, memory_id, text_for_embed)
                    if new_embedding_id:
                        embedding_regenerated = True
                        # ensure `updated_at` is bumped even if SQL part untouched
                        if not update_clauses:
                            await conn.execute(
                                "UPDATE memories SET updated_at = ? WHERE memory_id = ?",
                                (now_ts, memory_id),
                            )
                        logger.info(
                            f"Embedding regenerated (id={new_embedding_id}) for {memory_id}",
                            emoji_key="brain",
                        )
                except Exception as e:
                    logger.error(
                        f"Embedding regeneration error for {memory_id}: {e}",
                        exc_info=True,
                    )

        # ─── operation log ───
        log_payload = {"updated_fields": touched}
        if embedding_regenerated:
            log_payload |= {
                "embedding_regenerated": True,
                "new_embedding_id": new_embedding_id,
            }
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "update_memory",
            memory_id,
            None,
            log_payload,
        )

    proc_time = time.time() - start_time
    logger.info(
        f"Memory {memory_id} patched (fields: {touched or 'none'}, "
        f"embedding:{'yes' if embedding_regenerated else 'no'})",
        emoji_key="pencil2",
        time=proc_time,
    )

    return {
        "memory_id": memory_id,
        "updated_fields": touched,
        "embedding_regenerated": embedding_regenerated,
        "new_embedding_id": new_embedding_id,
        "updated_at": to_iso_z(now_ts),
        "processing_time": proc_time,
        "success": True,
    }


# ======================================================
# Linked Memories Retrieval
# ======================================================


@with_tool_metrics
@with_error_handling
async def get_linked_memories(
    memory_id: str,
    *,
    direction: str = "both",  # "outgoing", "incoming", or "both"
    link_type: str | None = None,  # optional filter
    limit: int = 10,
    include_memory_details: bool = True,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Fetch links touching *memory_id*.

    Return shape
    ------------
    {
        memory_id,
        links: {outgoing: [...], incoming: [...]},
        processing_time,
        success
    }
    Each link row retains integer timestamps and adds `created_at_iso`.
    Embedded memories (when requested) keep ints and add *_iso, with `tags`
    JSON-decoded.
    """
    t0 = time.time()

    if not memory_id:
        raise ToolInputError("Memory ID is required", param_name="memory_id")

    direction = direction.lower()
    if direction not in {"outgoing", "incoming", "both"}:
        raise ToolInputError(
            "Direction must be one of: outgoing, incoming, both", param_name="direction"
        )

    if link_type:
        try:
            LinkType(link_type.lower())
        except ValueError as e:
            allowed = ", ".join(lt.value for lt in LinkType)
            raise ToolInputError(
                f"Invalid link_type: choose from {allowed}", param_name="link_type"
            ) from e

    if limit <= 0:
        raise ToolInputError("Limit must be a positive integer.", param_name="limit")

    payload: Dict[str, Any] = {
        "memory_id": memory_id,
        "links": {"outgoing": [], "incoming": []},
        "success": True,
        "processing_time": 0.0,
    }

    def _add_iso(obj: Dict[str, Any], keys: Sequence[str]) -> None:
        for k in keys:
            if (ts := obj.get(k)) is not None:
                obj[f"{k}_iso"] = safe_format_timestamp(ts)

    db = DBConnection(db_path)
    async with db.transaction() as conn:  # R/W IMMEDIATE txn
        # Make sure the source memory exists; grab workflow for logging
        src_row = await conn.execute_fetchone(
            "SELECT workflow_id FROM memories WHERE memory_id = ?", (memory_id,)
        )
        if src_row is None:
            raise ToolInputError(f"Memory {memory_id} not found", param_name="memory_id")
        workflow_id = src_row["workflow_id"]

        # ---------- helper to pull memory details ----------
        async def _fetch_mem(mid: str) -> Dict[str, Any] | None:
            if not include_memory_details:
                return None
            m = await conn.execute_fetchone(
                """
                SELECT memory_id, memory_level, memory_type, importance, confidence,
                       description, created_at, updated_at, tags
                FROM memories WHERE memory_id = ?
                """,
                (mid,),
            )
            if m:
                md = dict(m)
                _add_iso(md, ["created_at", "updated_at"])
                md["tags"] = await MemoryUtils.deserialize(md.get("tags"))
                return md
            return None

        # ---------- outgoing ----------
        if direction in {"outgoing", "both"}:
            q = (
                "SELECT ml.*, m.memory_type AS target_type, m.description AS target_description "
                "FROM memory_links ml "
                "JOIN memories m ON m.memory_id = ml.target_memory_id "
                "WHERE ml.source_memory_id = ?"
            )
            params: list[Any] = [memory_id]
            if link_type:
                q += " AND ml.link_type = ?"
                params.append(link_type.lower())
            q += " ORDER BY ml.created_at DESC LIMIT ?"
            params.append(limit)

            for row in await conn.execute_fetchall(q, params):
                link = dict(row)
                _add_iso(link, ["created_at"])
                if include_memory_details and (tm := await _fetch_mem(link["target_memory_id"])):
                    link["target_memory"] = tm
                payload["links"]["outgoing"].append(link)

        # ---------- incoming ----------
        if direction in {"incoming", "both"}:
            q = (
                "SELECT ml.*, m.memory_type AS source_type, m.description AS source_description "
                "FROM memory_links ml "
                "JOIN memories m ON m.memory_id = ml.source_memory_id "
                "WHERE ml.target_memory_id = ?"
            )
            params: list[Any] = [memory_id]
            if link_type:
                q += " AND ml.link_type = ?"
                params.append(link_type.lower())
            q += " ORDER BY ml.created_at DESC LIMIT ?"
            params.append(limit)

            for row in await conn.execute_fetchall(q, params):
                link = dict(row)
                _add_iso(link, ["created_at"])
                if include_memory_details and (sm := await _fetch_mem(link["source_memory_id"])):
                    link["source_memory"] = sm
                payload["links"]["incoming"].append(link)

        # ---------- stats + logging ----------
        await MemoryUtils._update_memory_access(conn, memory_id)
        await MemoryUtils._log_memory_operation(
            conn,
            workflow_id,
            "access_links",
            memory_id,
            None,
            {"direction": direction, "link_type_filter": link_type},
        )

    payload["processing_time"] = time.time() - t0
    logger.info(
        f"Linked-memory query ({direction}) for {memory_id}: "
        f"{len(payload['links']['outgoing'])} out, {len(payload['links']['incoming'])} in.",
        emoji_key="link",
    )
    return payload


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
    *,
    workflow_id: str | None = None,
    target_memories: list[str] | None = None,
    consolidation_type: str = "summary",
    query_filter: dict[str, Any] | None = None,
    max_source_memories: int = 20,
    prompt_override: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    store_result: bool = True,
    store_as_level: str = MemoryLevel.SEMANTIC.value,
    store_as_type: str | None = None,
    max_tokens: int = 1_000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Synthesise multiple memories into *summary/insight/procedural/question* content.

    Reads use a **read-only** snapshot (no WAL growth); write-backs occur in a
    single IMMEDIATE transaction.  All original features are preserved.
    """
    t0 = time.time()
    valid_types = {"summary", "insight", "procedural", "question"}
    if consolidation_type not in valid_types:
        raise ToolInputError("Invalid consolidation_type.", param_name="consolidation_type")

    # ───────────────────────── source selection ─────────────────────────
    db = DBConnection(db_path)
    source_rows: list[dict[str, Any]] = []
    source_ids: list[str] = []
    effective_wf = workflow_id  # may be inferred

    async with db.transaction(readonly=True) as conn:
        if target_memories:
            if len(target_memories) < 2:
                raise ToolInputError(
                    "At least two target_memories required.", param_name="target_memories"
                )

            placeholders = ",".join("?" * len(target_memories))
            rows = await conn.execute_fetchall(
                f"SELECT * FROM memories WHERE memory_id IN ({placeholders})",
                target_memories,
            )
            found = {r["memory_id"]: dict(r) for r in rows}

            if not rows:
                raise ToolInputError("No target_memories found.", param_name="target_memories")
            effective_wf = effective_wf or rows[0]["workflow_id"]

            issues = []
            for mid in target_memories:
                if mid not in found:
                    issues.append(f"'{mid}' not found")
                elif found[mid]["workflow_id"] != effective_wf:
                    issues.append(f"'{mid}' not in workflow '{effective_wf}'")
                else:
                    source_rows.append(found[mid])
            if issues:
                raise ToolInputError(" ; ".join(issues), param_name="target_memories")

        elif query_filter:
            if not effective_wf:
                raise ToolInputError(
                    "workflow_id required with query_filter.", param_name="workflow_id"
                )

            where, params = ["workflow_id = ?"], [effective_wf]
            for k, v in query_filter.items():
                match k:
                    case "memory_level" | "memory_type" | "source" if v:
                        where.append(f"{k} = ?")
                        params.append(str(v).lower())
                    case "min_importance" if v is not None:
                        where.append("importance >= ?")
                        params.append(float(v))
                    case "min_confidence" if v is not None:
                        where.append("confidence >= ?")
                        params.append(float(v))
                    case _:
                        logger.debug(f"Ignoring unsupported filter key '{k}'")

            nowu = int(time.time())
            where.append("(ttl = 0 OR created_at + ttl > ?)")
            params.append(nowu)
            params.append(max_source_memories)

            sql = (
                f"SELECT * FROM memories WHERE {' AND '.join(where)} "
                "ORDER BY importance DESC, created_at DESC LIMIT ?"
            )
            source_rows = [dict(r) async for r in conn.execute(sql, params)]
        else:
            if not effective_wf:
                raise ToolInputError("workflow_id required.", param_name="workflow_id")

            nowu = int(time.time())
            sql = (
                "SELECT * FROM memories WHERE workflow_id = ? "
                "AND (ttl = 0 OR created_at + ttl > ?) "
                "ORDER BY importance DESC, created_at DESC LIMIT ?"
            )
            source_rows = [
                dict(r) async for r in conn.execute(sql, (effective_wf, nowu, max_source_memories))
            ]

        source_ids = [r["memory_id"] for r in source_rows]

    if len(source_rows) < 2:
        raise ToolError("Need ≥2 source memories to consolidate.")
    if not effective_wf:
        raise ToolError("Unable to determine workflow_id.")

    # ───────────────────────── LLM call ─────────────────────────
    prompt = prompt_override or _generate_consolidation_prompt(source_rows, consolidation_type)
    cfg = get_config()
    provider_name = provider or cfg.default_provider or LLMGatewayProvider.OPENAI.value
    provider_inst = await get_provider(provider_name)
    if not provider_inst:
        raise ToolError(f"LLM provider '{provider_name}' unavailable.")
    model_name = model or provider_inst.get_default_model()

    try:
        llm_resp = await provider_inst.generate_completion(
            prompt=prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.6,
        )
        consolidated = llm_resp.text.strip()
    except Exception as e:
        logger.error("LLM error in consolidation.", exc_info=True)
        raise ToolError(f"LLM error: {e}") from e

    # ───────────────────────── store + log (write tx) ─────────────────
    stored_id: str | None = None
    async with db.transaction() as wconn:
        if store_result and consolidated:
            mtype = (
                store_as_type
                or {
                    "summary": MemoryType.SUMMARY.value,
                    "insight": MemoryType.INSIGHT.value,
                    "procedural": MemoryType.PROCEDURE.value,
                    "question": MemoryType.QUESTION.value,
                }[consolidation_type]
            )

            try:
                mlevel = MemoryLevel(store_as_level.lower())
            except ValueError:
                mlevel = MemoryLevel.SEMANTIC

            # derive importance / confidence
            src_imp = [r.get("importance", 5.0) for r in source_rows]
            src_conf = [r.get("confidence", 0.5) for r in source_rows]
            imp = min(max(max(src_imp) + 0.5, 0), 10)
            conf = max(0.1, min(sum(src_conf) / len(src_conf), 1.0))

            res = await store_memory(
                workflow_id=effective_wf,
                content=consolidated,
                memory_type=mtype,
                memory_level=mlevel.value,
                importance=round(imp, 2),
                confidence=round(conf, 3),
                description=f"Consolidated {consolidation_type} from {len(source_ids)} memories.",
                source=f"consolidation_{consolidation_type}",
                tags=["consolidated", consolidation_type, mtype, mlevel.value],
                context_data={
                    "source_memories": source_ids,
                    "consolidation_type": consolidation_type,
                },
                generate_embedding=True,
                suggest_links=True,
                db_path=db_path,
            )
            stored_id = res.get("memory_id")

            # create links (fire-and-forget)
            if stored_id:
                await asyncio.gather(
                    *(
                        create_memory_link(
                            source_memory_id=stored_id,
                            target_memory_id=sid,
                            link_type=LinkType.GENERALIZES.value,
                            description=f"Source for {consolidation_type}",
                            db_path=db_path,
                        )
                        for sid in source_ids
                    ),
                    return_exceptions=True,
                )

        # operation log
        await MemoryUtils._log_memory_operation(
            wconn,
            effective_wf,
            "consolidate",
            None,
            None,
            {
                "consolidation_type": consolidation_type,
                "source_count": len(source_ids),
                "llm_provider": provider_name,
                "llm_model": model_name or "provider_default",
                "stored_memory_id": stored_id,
                "content_length": len(consolidated),
            },
        )

    elapsed = time.time() - t0
    logger.info(
        f"Consolidated {len(source_ids)} memories → {stored_id or 'not stored'} in {elapsed:.2f}s",
        emoji_key="sparkles",
    )
    return {
        "consolidated_content": consolidated or "LLM produced no content.",
        "consolidation_type": consolidation_type,
        "source_memory_ids": source_ids,
        "workflow_id": effective_wf,
        "stored_memory_id": stored_id,
        "success": True,
        "processing_time": elapsed,
    }


@with_tool_metrics
@with_error_handling
async def generate_reflection(
    workflow_id: str,
    *,
    reflection_type: str = "summary",  # summary | progress | gaps | strengths | plan
    recent_ops_limit: int = 30,
    provider: str = LLMGatewayProvider.OPENAI.value,
    model: str | None = None,
    max_tokens: int = 1_000,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Run a meta-cognitive LLM pass over recent workflow activity and persist the result.
    """
    t0 = time.time()
    if reflection_type not in {"summary", "progress", "gaps", "strengths", "plan"}:
        raise ToolInputError(
            "Invalid reflection_type. Must be one of: summary, progress, gaps, strengths, plan",
            param_name="reflection_type",
        )

    db = DBConnection(db_path)

    # ───────────────────────── 1. fetch context (read-only) ─────────────────────────
    ops: list[dict[str, Any]]
    mem_ids: set[str] = set()
    mem_meta: dict[str, dict[str, Any]] = {}

    async with db.transaction(readonly=True) as conn:
        wf_row = await conn.execute_fetchone(
            "SELECT title, description FROM workflows WHERE workflow_id=?",
            (workflow_id,),
        )
        if wf_row is None:
            raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

        wf_title, wf_desc = wf_row["title"], wf_row["description"]

        raw_ops_rows = await conn.execute_fetchall(
            "SELECT * FROM memory_operations WHERE workflow_id=? ORDER BY timestamp DESC LIMIT ?",
            (workflow_id, recent_ops_limit),
        )
        # Ensure ops is a list of dicts for consistent access
        ops = [dict(row) for row in raw_ops_rows]

        if not ops:
            raise ToolError("No memory_operations to analyse for this workflow.")

        for op in ops:  # Now op is a dictionary
            if op.get(
                "memory_id"
            ):  # Use .get() for safety, though direct access would also work now
                mem_ids.add(op["memory_id"])

        if mem_ids:
            placeholders = ",".join("?" * len(mem_ids))
            async with conn.execute(
                f"""
                SELECT memory_id, description, memory_type
                FROM memories
                WHERE memory_id IN ({placeholders})
                """,
                list(mem_ids),
            ) as cur:
                async for row in cur:
                    mem_meta[row["memory_id"]] = dict(row)

    # ───────────────────────── 2. build prompt ─────────────────────────
    prompt = _generate_reflection_prompt(wf_title, wf_desc, ops, mem_meta, reflection_type)

    # ───────────────────────── 3. call LLM ────────────────────────────
    provider_cfg = get_config()
    provider_name = provider or provider_cfg.default_provider or LLMGatewayProvider.OPENAI.value
    prov = await get_provider(provider_name)
    if prov is None:
        raise ToolError(f"Could not initialise LLM provider '{provider_name}'.")

    model_name = model or prov.get_default_model()
    try:
        llm_out = await prov.generate_completion(
            prompt=prompt, model=model_name, max_tokens=max_tokens, temperature=0.7
        )
    except Exception as llm_err:
        logger.error("LLM failure during reflection", exc_info=True)
        raise ToolError(f"Reflection failed (LLM): {llm_err}") from llm_err

    content = llm_out.text.strip()
    if not content:
        raise ToolError("LLM returned empty reflection.")

    # ───────────────────────── 4. persist reflection (write txn) ─────────────────────
    refl_id = MemoryUtils.generate_id()
    now = int(time.time())
    title = (content.split("\n", 1)[0].lstrip("# ").strip() or reflection_type.title())[:100]

    async with db.transaction() as wconn:
        await wconn.execute(
            """
            INSERT INTO reflections
                  (reflection_id, workflow_id, title, content, reflection_type,
                   created_at, referenced_memories)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                refl_id,
                workflow_id,
                title,
                content,
                reflection_type,
                now,
                json.dumps(list(mem_ids)),
            ),
        )
        await MemoryUtils._log_memory_operation(
            wconn,
            workflow_id,
            "reflect",
            None,  # memory_id
            None,  # action_id
            {
                "reflection_id": refl_id,
                "reflection_type": reflection_type,
                "ops_analyzed": len(ops),
                "title": title,
            },
        )

    elapsed = time.time() - t0
    logger.info(
        f"Reflection '{title}' ({_fmt_id(refl_id)}) generated for workflow {_fmt_id(workflow_id)} "
        f"in {elapsed:0.2f}s",
        emoji_key="mirror",
    )
    return {
        "reflection_id": refl_id,
        "workflow_id": workflow_id,
        "reflection_type": reflection_type,
        "title": title,
        "content": content,
        "operations_analyzed": len(ops),
        "processing_time": elapsed,
        "success": True,
    }


# ======================================================
# Text Summarization (using LLM)
# ======================================================


@with_tool_metrics
@with_error_handling
async def summarize_text(
    text_to_summarize: str,
    *,
    target_tokens: int = 500,
    prompt_template: str | None = None,
    provider: str = "openai",
    model: str | None = None,
    workflow_id: str | None = None,
    record_summary: bool = False,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Summarise *text_to_summarize* with an LLM and (optionally) store the result as a memory.

    Returned keys
    -------------
    summary, original_length, summary_length,
    stored_memory_id, success, processing_time
    """
    t0 = time.time()

    # ───────── basic validation ─────────
    if not text_to_summarize:
        raise ToolInputError("text_to_summarize cannot be empty", param_name="text_to_summarize")
    if record_summary and not workflow_id:
        raise ToolInputError(
            "workflow_id required when record_summary=True", param_name="workflow_id"
        )

    target_tokens = max(50, min(2_000, target_tokens))

    # ───────── provider / model resolution ─────────
    cfg = get_config()
    provider_key = provider or cfg.default_provider or LLMGatewayProvider.OPENAI.value
    default_models = {
        LLMGatewayProvider.OPENAI.value: "gpt-4.1-mini",
        LLMGatewayProvider.ANTHROPIC.value: "claude-3-5-haiku-20241022",
    }
    model_name = model or default_models.get(provider_key)
    if model_name is None:
        prov_tmp = await get_provider(provider_key)
        model_name = prov_tmp.get_default_model()

    # ───────── default prompt (rich version) ─────────
    if prompt_template is None:
        prompt_template = (
            "You are an expert technical writer and editor.\n"
            "Your task is to produce a **concise, accurate, well-structured summary** "
            "of the text provided below.\n\n"
            "**Requirements**\n"
            "1. Length ≈ {target_tokens} tokens (±10 %).\n"
            "2. Preserve all critical facts, numbers, names, and causal relationships.\n"
            "3. Omit anecdotes, filler, or rhetorical questions unless essential to meaning.\n"
            "4. Write in clear, neutral, third-person prose; bullet lists are welcome where helpful.\n"
            "5. Do **not** add opinions or external knowledge.\n\n"
            "---\n"
            "### ORIGINAL TEXT\n"
            "{text_to_summarize}\n"
            "---\n"
            "### SUMMARY\n"
        )

    # ───────── LLM invocation ─────────
    try:
        prov = await get_provider(provider_key)
        prompt = prompt_template.format(
            text_to_summarize=text_to_summarize,
            target_tokens=target_tokens,
        )
        result = await prov.generate_completion(
            prompt=prompt,
            model=model_name,
            max_tokens=target_tokens + 100,
            temperature=0.3,
        )
        summary = result.text.strip()
        if not summary:
            raise ToolError("LLM returned empty summary.")
    except Exception as exc:
        logger.error(f"LLM summarisation failed: {exc}", exc_info=True)
        raise ToolError(f"Failed to generate summary: {exc}") from exc

    stored_memory_id: str | None = None

    # ───────── optional persistence ─────────
    if record_summary:
        db = DBConnection(db_path)
        async with db.transaction() as conn:
            if not await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ):
                raise ToolInputError(f"Workflow {workflow_id} not found.", param_name="workflow_id")

            now = int(time.time())
            stored_memory_id = MemoryUtils.generate_id()
            tags_json = json.dumps(
                list(
                    {
                        "summary",
                        "automated",
                        "text_summary",
                        MemoryLevel.SEMANTIC.value,
                        MemoryType.SUMMARY.value,
                    }
                )
            )

            await conn.execute(
                """
                INSERT INTO memories (
                    memory_id, workflow_id, content,
                    memory_level, memory_type,
                    importance, confidence,
                    description, source, tags,
                    created_at, updated_at, access_count
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    stored_memory_id,
                    workflow_id,
                    summary,
                    MemoryLevel.SEMANTIC.value,
                    MemoryType.SUMMARY.value,
                    6.0,
                    0.85,
                    f"Summary of {len(text_to_summarize)}-character text",
                    "summarize_text_tool",
                    tags_json,
                    now,
                    now,
                    0,
                ),
            )

            await MemoryUtils._log_memory_operation(
                conn,
                workflow_id,
                "create_summary_memory",
                stored_memory_id,
                None,
                {"original_length": len(text_to_summarize), "summary_length": len(summary)},
            )

    elapsed = time.time() - t0
    logger.info(
        f"Summarised {len(text_to_summarize)} chars → {len(summary)} chars "
        f"({'stored' if stored_memory_id else 'not stored'})",
        emoji_key="scissors",
        time=elapsed,
    )

    return {
        "summary": summary,
        "original_length": len(text_to_summarize),
        "summary_length": len(summary),
        "stored_memory_id": stored_memory_id,
        "success": True,
        "processing_time": elapsed,
    }


# --- 17. Maintenance ---


@with_tool_metrics
@with_error_handling
async def delete_expired_memories(
    *,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Purge memories whose *ttl* has elapsed and write an operation-log entry
    for each affected workflow.

    Returns
    -------
    {
        deleted_count      : int,
        workflows_affected : list[str],
        success            : True,
        processing_time    : float   # seconds
    }
    """
    t0 = time.time()
    db = DBConnection(db_path)
    now_ts = int(time.time())

    deleted_ids: list[str] = []
    wf_affected: set[str] = set()

    # ─────────────────────────────────────────────────────────
    # single IMMEDIATE transaction → short WAL, no lost rows
    # ─────────────────────────────────────────────────────────
    async with db.transaction(mode="IMMEDIATE") as conn:
        rows = await conn.execute_fetchall(
            """
            SELECT memory_id, workflow_id
            FROM   memories
            WHERE  ttl > 0
              AND  created_at + ttl < ?
            """,
            (now_ts,),
        )

        if not rows:
            return {
                "deleted_count": 0,
                "workflows_affected": [],
                "success": True,
                "processing_time": time.time() - t0,
            }

        deleted_ids = [r["memory_id"] for r in rows]
        wf_affected = {r["workflow_id"] for r in rows}

        # batch delete (SQLITE_MAX_VARIABLE_NUMBER ≈ 999)
        BATCH = 500
        for i in range(0, len(deleted_ids), BATCH):
            batch = deleted_ids[i : i + BATCH]
            ph = ",".join("?" * len(batch))
            await conn.execute(f"DELETE FROM memories WHERE memory_id IN ({ph})", batch)

        # per-workflow operation log
        for wf in wf_affected:
            expired_here = sum(r["workflow_id"] == wf for r in rows)
            await MemoryUtils._log_memory_operation(
                conn,
                wf,
                "expire_batch",
                None,  # memory_id
                None,  # action_id
                {
                    "expired_count_in_workflow": expired_here,
                    "total_expired_this_run": len(deleted_ids),
                },
            )
    # ────────────── transaction committed ──────────────

    dt = time.time() - t0
    logger.success(
        f"Expired-memory sweep removed {len(deleted_ids)} rows "
        f"across {len(wf_affected)} workflows.",
        emoji_key="wastebasket",
        time=dt,
    )
    return {
        "deleted_count": len(deleted_ids),
        "workflows_affected": sorted(wf_affected),
        "success": True,
        "processing_time": dt,
    }


@with_tool_metrics
@with_error_handling
async def get_rich_context_package(
    workflow_id: str,
    context_id: Optional[str] = None,
    current_plan_step_description: Optional[str] = None,
    focal_memory_id_hint: Optional[str] = None,
    fetch_limits: Optional[Dict[str, int]] = None,
    show_limits: Optional[Dict[str, int]] = None,
    include_core_context: bool = True,
    include_working_memory: bool = True,
    include_proactive_memories: bool = True,
    include_relevant_procedures: bool = True,
    include_contextual_links: bool = True,
    compression_token_threshold: Optional[int] = None,
    compression_target_tokens: Optional[int] = None,
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Assemble a *rich* context package for the agent master-loop (AML).

    All existing functionality is preserved — limits, optional sections,
    UMS-side compression, error tracking, focal-memory propagation, etc.
    The only behavioural change is stricter read-only validation via the
    revamped `DBConnection.transaction(readonly=True)` (zero WAL churn).
    """
    start_time = time.time()
    retrieval_ts = datetime.now(timezone.utc).isoformat()

    assembled: Dict[str, Any] = {"retrieval_timestamp_ums_package": retrieval_ts}
    errors: List[str] = []
    focal_mem_id_for_links: Optional[str] = focal_memory_id_hint

    fetch_limits = fetch_limits or {}
    show_limits = show_limits or {}

    lim_actions = fetch_limits.get("recent_actions", UMS_PKG_DEFAULT_FETCH_RECENT_ACTIONS)
    lim_imp_mems = fetch_limits.get("important_memories", UMS_PKG_DEFAULT_FETCH_IMPORTANT_MEMORIES)
    lim_key_thts = fetch_limits.get("key_thoughts", UMS_PKG_DEFAULT_FETCH_KEY_THOUGHTS)
    lim_proactive = fetch_limits.get("proactive_memories", UMS_PKG_DEFAULT_FETCH_PROACTIVE)
    lim_procedural = fetch_limits.get("procedural_memories", UMS_PKG_DEFAULT_FETCH_PROCEDURAL)
    lim_links = fetch_limits.get("link_traversal", UMS_PKG_DEFAULT_FETCH_LINKS)

    lim_show_links_summary = show_limits.get("link_traversal", UMS_PKG_DEFAULT_SHOW_LINKS_SUMMARY)

    # ─────────────────────────── 0. Validation ───────────────────────────
    db = DBConnection(db_path)
    try:
        async with db.transaction(readonly=True) as conn:
            row = await conn.execute_fetchone(
                "SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)
            )
            if row is None:
                raise ToolInputError(
                    f"Target workflow_id '{workflow_id}' not found in UMS.",
                    param_name="workflow_id",
                )
    except ToolInputError:
        raise
    except Exception as exc:
        logger.error("UMS Package: workflow validation failed", exc_info=True)
        return {
            "success": False,
            "error": f"Workflow ID validation failed: {exc}",
            "processing_time": time.time() - start_time,
        }

    # ─────────────────────────── 1. Core context ─────────────────────────
    if include_core_context:
        try:
            core_res = await get_workflow_context(
                workflow_id=workflow_id,
                recent_actions_limit=lim_actions,
                important_memories_limit=lim_imp_mems,
                key_thoughts_limit=lim_key_thts,
                db_path=db_path,
            )
            if core_res.get("success"):
                assembled["core_context"] = {
                    **{
                        k: v for k, v in core_res.items() if k not in ("success", "processing_time")
                    },
                    "retrieved_at": retrieval_ts,
                }
            else:
                msg = f"Core context retrieval failed: {core_res.get('error')}"
                errors.append(f"UMS Package: {msg}")
                logger.warning(msg)
        except Exception as exc:
            msg = f"Exception in get_workflow_context: {exc}"
            errors.append(f"UMS Package: {msg}")
            logger.error(msg, exc_info=True)

    # ─────────────────────────── 2. Working memory ───────────────────────
    if include_working_memory and context_id:
        try:
            wm_res = await get_working_memory(
                context_id=context_id,
                include_content=False,
                include_links=False,
                db_path=db_path,
            )
            if wm_res.get("success"):
                assembled["current_working_memory"] = {
                    **{k: v for k, v in wm_res.items() if k not in ("success", "processing_time")},
                    "retrieved_at": retrieval_ts,
                }
                focal_mem_id_for_links = wm_res.get("focal_memory_id") or focal_mem_id_for_links
            else:
                msg = f"Working memory retrieval failed: {wm_res.get('error')}"
                errors.append(f"UMS Package: {msg}")
                logger.warning(msg)
        except Exception as exc:
            msg = f"Exception in get_working_memory: {exc}"
            errors.append(f"UMS Package: {msg}")
            logger.error(msg, exc_info=True)

    # ─────────────────────────── 3. Proactive / procedural ───────────────
    search_source = current_plan_step_description or "current agent objectives"

    if include_proactive_memories:
        try:
            q = f"Information relevant to current task or goal: {search_source}"
            pr_res = await hybrid_search_memories(
                query=q,
                workflow_id=workflow_id,
                limit=lim_proactive,
                include_content=False,
                semantic_weight=0.7,
                keyword_weight=0.3,
                db_path=db_path,
            )
            if pr_res.get("success"):
                assembled["proactive_memories"] = {
                    "retrieved_at": retrieval_ts,
                    "query_used": q,
                    "memories": pr_res.get("memories", []),
                }
            else:
                errors.append(f"UMS Package: Proactive search failed: {pr_res.get('error')}")
        except Exception as exc:
            errors.append(f"UMS Package: Proactive search exception: {exc}")
            logger.error("Proactive search error", exc_info=True)

    if include_relevant_procedures:
        try:
            q = f"How to accomplish, perform, or execute: {search_source}"
            proc_res = await hybrid_search_memories(
                query=q,
                workflow_id=workflow_id,
                limit=lim_procedural,
                memory_level=MemoryLevel.PROCEDURAL.value,
                include_content=False,
                db_path=db_path,
            )
            if proc_res.get("success"):
                assembled["relevant_procedures"] = {
                    "retrieved_at": retrieval_ts,
                    "query_used": q,
                    "procedures": proc_res.get("memories", []),
                }
            else:
                errors.append(f"UMS Package: Procedural search failed: {proc_res.get('error')}")
        except Exception as exc:
            errors.append(f"UMS Package: Procedural search exception: {exc}")
            logger.error("Procedural search error", exc_info=True)

    # ─────────────────────────── 4. Contextual links ─────────────────────
    if include_contextual_links:
        link_seed = focal_mem_id_for_links
        if link_seed is None:
            imp_mems = assembled.get("core_context", {}).get("important_memories", [])
            if imp_mems:
                link_seed = imp_mems[0].get("memory_id")

        if link_seed:
            try:
                link_res = await get_linked_memories(
                    memory_id=link_seed,
                    direction="both",
                    limit=lim_links,
                    include_memory_details=False,
                    db_path=db_path,
                )
                if link_res.get("success"):
                    payload = link_res["links"]
                    asm = {
                        "source_memory_id": link_seed,
                        "outgoing_count": len(payload.get("outgoing", [])),
                        "incoming_count": len(payload.get("incoming", [])),
                        "top_outgoing_links_summary": [
                            {
                                "target_memory_id": _fmt_id(link["target_memory_id"]),
                                "link_type": link["link_type"],
                                "description": (link.get("description") or "")[:70] + "…",
                            }
                            for link in payload.get("outgoing", [])[:lim_show_links_summary]
                        ],
                        "top_incoming_links_summary": [
                            {
                                "source_memory_id": _fmt_id(link["source_memory_id"]),
                                "link_type": link["link_type"],
                                "description": (link.get("description") or "")[:70] + "…",
                            }
                            for link in payload.get("incoming", [])[:lim_show_links_summary]
                        ],
                    }
                    assembled["contextual_links"] = {"retrieved_at": retrieval_ts, "summary": asm}
                else:
                    errors.append(f"UMS Package: Link retrieval failed: {link_res.get('error')}")
            except Exception as exc:
                errors.append(f"UMS Package: Link retrieval exception: {exc}")
                logger.error("Link retrieval error", exc_info=True)

    # ─────────────────────────── 5. Compression ─────────────────────────
    if compression_token_threshold is not None and compression_target_tokens is not None:
        try:
            pkg_json = json.dumps(assembled, default=str)
            tok_est = count_tokens(pkg_json)
            if tok_est > compression_token_threshold:
                logger.info(
                    f"Context {workflow_id} at {tok_est} tokens exceeds {compression_token_threshold}; compressing."
                )
                # heuristic selection of a large list to summarise
                cand = {
                    "core_context.recent_actions": assembled.get("core_context", {}).get(
                        "recent_actions"
                    ),
                    "core_context.important_memories": assembled.get("core_context", {}).get(
                        "important_memories"
                    ),
                    "proactive_memories.memories": assembled.get("proactive_memories", {}).get(
                        "memories"
                    ),
                    "relevant_procedures.procedures": assembled.get("relevant_procedures", {}).get(
                        "procedures"
                    ),
                }
                target_key, target_txt, max_tok = None, "", 0
                thresh = compression_target_tokens * 0.5

                for k, v in cand.items():
                    if v and isinstance(v, list) and len(v) > 3:
                        s = json.dumps(v, default=str)
                        t = count_tokens(s)
                        if t > thresh and t > max_tok:
                            target_key, target_txt, max_tok = k, s, t

                if target_key:
                    sum_res = await summarize_text(
                        text_to_summarize=target_txt,
                        target_tokens=int(compression_target_tokens * 0.6),
                        context_type=f"ums_package_component:{target_key}",
                        workflow_id=workflow_id,
                        record_summary=False,
                        db_path=db_path,
                    )
                    if sum_res.get("success") and sum_res.get("summary"):
                        # replace component with marker + preview
                        keys = target_key.split(".")
                        ref = assembled
                        for k in keys[:-1]:
                            ref = ref.setdefault(k, {})
                        original_rt = ref.get(keys[-1], {}).get("retrieved_at", retrieval_ts)
                        ref[keys[-1]] = {
                            "retrieved_at": original_rt,
                            "_ums_compressed_": True,
                            "original_token_estimate": max_tok,
                            "summary_preview": sum_res["summary"][:150] + "…",
                        }
                        assembled.setdefault("ums_compression_details", {})[target_key] = {
                            "summary_content": sum_res["summary"],
                            "retrieved_at": retrieval_ts,
                        }
                        logger.info(f"Compressed component '{target_key}' in context package.")
                    else:
                        errors.append(
                            f"UMS Package: Compression of '{target_key}' failed: {sum_res.get('error')}"
                        )
            else:
                logger.debug(f"Context {workflow_id} size {tok_est} within threshold.")
        except Exception as exc:
            errors.append(f"UMS Package: Compression exception: {exc}")
            logger.error("Compression error", exc_info=True)

    # ─────────────────────────── 6. Final wrap ──────────────────────────
    resp = {
        "success": not errors,
        "context_package": assembled,
        "errors": errors or None,
        "processing_time": time.time() - start_time,
    }

    if errors:
        logger.warning(f"get_rich_context_package for {workflow_id} completed with errors.")
    else:
        logger.info(f"get_rich_context_package for {workflow_id} succeeded.")
    return resp


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
    report_format: str = "markdown",  # markdown | html | json | mermaid
    include_details: bool = True,
    include_thoughts: bool = True,
    include_artifacts: bool = True,
    style: Optional[str] = "professional",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Generates a comprehensive report for the specified workflow.

    All functional behaviour (formats, styles, helper calls, error handling,
    timing metadata, HTML assembly, etc.) is preserved 100 %.
    The only revision is the read-only DB access pattern, now routed through
    the new DBConnection.read-only snapshot to guarantee zero WAL impact.
    """
    # -------- Validation (unchanged) -----------------
    if not workflow_id:
        raise ToolInputError("Workflow ID required.", param_name="workflow_id")

    valid_formats = ["markdown", "html", "json", "mermaid"]
    report_format_lower = (report_format or "markdown").lower()
    if report_format_lower not in valid_formats:
        raise ToolInputError(
            f"Invalid format '{report_format}'. Must be one of: {valid_formats}",
            param_name="report_format",
        )

    valid_styles = ["professional", "concise", "narrative", "technical"]
    style_lower = (style or "professional").lower()
    if report_format_lower in ["markdown", "html"] and style_lower not in valid_styles:
        raise ToolInputError(
            f"Invalid style '{style}'. Must be one of: {valid_styles}",
            param_name="style",
        )

    start_time = time.time()

    try:
        # -------- READ-ONLY data hydration ------------
        # We open a read-only snapshot explicitly; this guarantees that the
        # report never performs writes nor triggers WAL checkpoints.
        async with DBConnection(db_path).transaction(readonly=True) as _:
            workflow_data = await get_workflow_details(
                workflow_id=workflow_id,
                include_actions=True,
                include_artifacts=include_artifacts,
                include_thoughts=include_thoughts,
                include_memories=False,
                db_path=db_path,  # propagated unchanged
            )

        if not workflow_data.get("success"):
            raise ToolError(
                f"Failed to retrieve workflow details for report generation (ID: {workflow_id})."
            )

        # -------- Report generation (unchanged) -------
        report_content = None
        markdown_report_content = None

        if report_format_lower in ["markdown", "html"]:
            # Style-specific markdown
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
            else:  # professional
                markdown_report_content = await _generate_professional_report(
                    workflow_data, include_details
                )

            if report_format_lower == "markdown":
                report_content = markdown_report_content
            else:  # html
                try:
                    html_body = markdown.markdown(
                        markdown_report_content,
                        extensions=["tables", "fenced_code", "codehilite"],
                    )
                    pygments_css = ""
                    try:
                        formatter = HtmlFormatter(style="default")
                        pygments_css = f"<style>{formatter.get_style_defs('.codehilite')}</style>"
                    except Exception as css_err:
                        logger.warning(f"Failed to generate Pygments CSS: {css_err}")
                        pygments_css = "<!-- Pygments CSS generation failed -->"
                    report_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Workflow Report: {workflow_data.get("title", "Untitled")}</title>
    {pygments_css}
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        pre code {{ display: block; padding: 10px; background-color: #f5f5f5;
                   border: 1px solid #ddd; border-radius: 4px; }}
        .codehilite pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
                except Exception as md_err:
                    logger.error("Markdown → HTML conversion failed", exc_info=True)
                    raise ToolError(f"Failed to convert report to HTML: {md_err}") from md_err

        elif report_format_lower == "json":
            try:
                clean_data = {
                    k: v
                    for k, v in workflow_data.items()
                    if k not in ["success", "processing_time"]
                }
                report_content = json.dumps(clean_data, indent=2, ensure_ascii=False)
            except Exception as json_err:
                logger.error("JSON serialization failed for report", exc_info=True)
                raise ToolError(
                    f"Failed to serialize workflow data to JSON: {json_err}"
                ) from json_err

        else:  # mermaid
            report_content = await _generate_mermaid_diagram(workflow_data)

        if report_content is None:
            raise ToolError(
                f"Report content generation failed unexpectedly for format '{report_format_lower}' and style '{style_lower}'."
            )

        # -------- Assemble result ---------------------
        result = {
            "workflow_id": workflow_id,
            "title": workflow_data.get("title", "Workflow Report"),
            "report": report_content,
            "format": report_format_lower,
            "style_used": style_lower if report_format_lower in ["markdown", "html"] else None,
            "generated_at": to_iso_z(datetime.now(timezone.utc).timestamp()),
            "success": True,
            "processing_time": time.time() - start_time,
        }
        logger.info(
            f"Generated {report_format_lower} report (style: {style_lower if report_format_lower in ['markdown', 'html'] else 'N/A'}) for workflow {workflow_id}",
            emoji_key="newspaper",
        )
        return result

    except (ToolInputError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report for {workflow_id}: {e}", exc_info=True)
        raise ToolError(
            f"Failed to generate workflow report due to an unexpected error: {e}"
        ) from e


# --- 20. Visualization ---
@with_tool_metrics
@with_error_handling
async def visualize_reasoning_chain(
    thought_chain_id: str,
    output_format: str = "mermaid",  # 'mermaid' | 'json'
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Produce a Mermaid diagram or hierarchical JSON for a single thought-chain.

    Functionality unchanged; the only difference is that data retrieval now
    runs inside a **read-only snapshot** using the new DBConnection helper,
    ensuring zero WAL interaction.
    """
    # ───────────────── input validation ─────────────────
    if not thought_chain_id:
        raise ToolInputError("Thought chain ID required.", param_name="thought_chain_id")

    valid_formats = {"mermaid", "json"}
    output_format_lc = (output_format or "mermaid").lower()
    if output_format_lc not in valid_formats:
        raise ToolInputError(
            f"Invalid format. Use one of: {sorted(valid_formats)}",
            param_name="output_format",
        )

    t0 = time.time()

    try:
        # ─────────────── read-only data fetch ───────────────
        async with DBConnection(db_path).transaction(readonly=True):
            thought_chain_data = await get_thought_chain(thought_chain_id, db_path=db_path)

        if not thought_chain_data.get("success"):
            raise ToolError(
                f"Failed to retrieve thought chain {thought_chain_id} for visualization."
            )

        # ─────────────── generate visualisation ─────────────
        visual: str | None = None

        if output_format_lc == "mermaid":
            visual = await _generate_thought_chain_mermaid(thought_chain_data)

        else:  # json
            structured = {
                k: v for k, v in thought_chain_data.items() if k not in {"success", "thoughts"}
            }
            child_map: Dict[str | None, list[dict]] = defaultdict(list)
            for th in thought_chain_data.get("thoughts", []):
                child_map[th.get("parent_thought_id")].append(th)

            def build(nodes: list[dict]) -> list[dict]:
                tree = []
                for node in nodes:
                    n = dict(node)
                    kids = child_map.get(node["thought_id"])
                    if kids:
                        n["children"] = build(kids)
                    tree.append(n)
                return tree

            structured["thought_tree"] = build(child_map.get(None, []))
            visual = json.dumps(structured, indent=2)

        if visual is None:
            raise ToolError(
                f"Failed to generate visualization content for format '{output_format_lc}'."
            )

        result = {
            "thought_chain_id": thought_chain_id,
            "title": thought_chain_data.get("title", "Thought Chain"),
            "visualization": visual,
            "format": output_format_lc,
            "success": True,
            "processing_time": time.time() - t0,
        }
        logger.info(
            f"Generated {output_format_lc} visualization for thought chain {thought_chain_id}",
            emoji_key="projector",
        )
        return result

    except (ToolInputError, ToolError):
        raise
    except Exception as exc:
        logger.error(f"Error visualizing thought chain {thought_chain_id}: {exc}", exc_info=True)
        raise ToolError(f"Failed to visualize thought chain: {exc}") from exc


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
    depth: int = 1,
    max_nodes: int = 30,
    memory_level: Optional[str] = None,
    memory_type: Optional[str] = None,
    output_format: str = "mermaid",
    db_path: str = agent_memory_config.db_path,
) -> Dict[str, Any]:
    """
    Generates a Mermaid diagram of the memory graph.  All original behaviour,
    parameters, and return shape are preserved exactly; the sole change is that
    the query block is now executed inside a read-only snapshot transaction
    (`DBConnection.transaction(readonly=True)`) so the visualisation never
    interferes with WAL housekeeping.
    """
    # ------------- validation (unchanged) -------------
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
    selected_memory_ids: set[str] = set()
    memories_data: dict[str, Any] = {}
    links_data: list[dict[str, Any]] = []

    try:
        # ---------- read-only snapshot ----------
        async with DBConnection(db_path).transaction(readonly=True) as conn:
            # --- 1. Initial memory selection (unchanged) ---
            target_workflow_id = workflow_id
            if center_memory_id:
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

                queue, visited = {center_memory_id}, set()
                for current_depth in range(depth + 1):
                    if len(selected_memory_ids) >= max_nodes:
                        break
                    next_queue: set[str] = set()
                    ids_to_query = list(queue - visited)
                    if not ids_to_query:
                        break
                    visited.update(ids_to_query)
                    for node_id in ids_to_query:
                        if len(selected_memory_ids) < max_nodes:
                            selected_memory_ids.add(node_id)
                        else:
                            break
                    if current_depth < depth and len(selected_memory_ids) < max_nodes:
                        placeholders = ", ".join("?" * len(ids_to_query))
                        neighbor_query = (
                            f"SELECT target_memory_id AS neighbor_id FROM memory_links "
                            f"WHERE source_memory_id IN ({placeholders}) "
                            f"UNION "
                            f"SELECT source_memory_id AS neighbor_id FROM memory_links "
                            f"WHERE target_memory_id IN ({placeholders})"
                        )
                        async with conn.execute(neighbor_query, ids_to_query * 2) as cursor:
                            async for row in cursor:
                                if row["neighbor_id"] not in visited:
                                    next_queue.add(row["neighbor_id"])
                    queue = next_queue
            else:
                if not target_workflow_id:
                    raise ToolInputError(
                        "Workflow ID is required when not specifying a center memory.",
                        param_name="workflow_id",
                    )
                filter_where = ["workflow_id = ?"]
                params: list[Any] = [target_workflow_id]
                if memory_level:
                    filter_where.append("memory_level = ?")
                    params.append(memory_level.lower())
                if memory_type:
                    filter_where.append("memory_type = ?")
                    params.append(memory_type.lower())
                now_unix = int(time.time())
                filter_where.append("(ttl = 0 OR created_at + ttl > ?)")
                params.append(now_unix)
                where_sql = " AND ".join(filter_where)
                query = (
                    "SELECT memory_id "
                    "FROM memories "
                    f"WHERE {where_sql} "
                    "ORDER BY compute_memory_relevance("
                    "    importance, confidence, created_at, access_count, last_accessed"
                    ") DESC "
                    "LIMIT ?"
                )
                params.append(max_nodes)
                async with conn.execute(query, params) as cursor:
                    selected_memory_ids = {row["memory_id"] for row in await cursor.fetchall()}

            # --- early-exit branches & rest of logic (unchanged) ---
            if not selected_memory_ids:
                logger.info("No memories selected for visualization based on criteria.")
                return {
                    "workflow_id": target_workflow_id,
                    "center_memory_id": center_memory_id,
                    "visualization": "```mermaid\ngraph TD\n    NoNodes[No memories found]\n```",
                    "node_count": 0,
                    "link_count": 0,
                    "format": "mermaid",
                    "success": True,
                    "processing_time": time.time() - start_time,
                }

            fetch_ids = list(selected_memory_ids)
            placeholders = ", ".join("?" * len(fetch_ids))
            details_query = f"SELECT * FROM memories WHERE memory_id IN ({placeholders})"
            async with conn.execute(details_query, fetch_ids) as cursor:
                async for row in cursor:
                    mem_data = dict(row)
                    if center_memory_id:
                        ok = True
                        if memory_level and mem_data.get("memory_level") != memory_level.lower():
                            ok = False
                        if memory_type and mem_data.get("memory_type") != memory_type.lower():
                            ok = False
                        if ok or mem_data["memory_id"] == center_memory_id:
                            memories_data[mem_data["memory_id"]] = mem_data
                    else:
                        memories_data[mem_data["memory_id"]] = mem_data
            final_selected_ids = set(memories_data)

            if center_memory_id and center_memory_id not in final_selected_ids:
                if center_memory_id in memories_data:
                    final_selected_ids.add(center_memory_id)

            if not final_selected_ids:
                logger.info("No memories remained after applying filters for visualization.")
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

            final_ids_list = list(final_selected_ids)
            placeholders = ", ".join("?" * len(final_ids_list))
            links_query = (
                "SELECT * FROM memory_links "
                f"WHERE source_memory_id IN ({placeholders}) "
                f"AND target_memory_id IN ({placeholders})"
            )
            async with conn.execute(links_query, final_ids_list * 2) as cursor:
                links_data = [dict(row) for row in await cursor.fetchall()]

        # --- diagram generation & return (unchanged) ---
        mermaid_string = await _generate_memory_network_mermaid(
            list(memories_data.values()), links_data, center_memory_id
        )
        processing_time = time.time() - start_time
        node_count, link_count = len(memories_data), len(links_data)
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
        raise ToolError(f"Failed to visualize memory network: {e}") from e


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
