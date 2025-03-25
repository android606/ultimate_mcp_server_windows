"""Constants used throughout the LLM Gateway."""
from enum import Enum
from typing import Dict


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


class TaskType(str, Enum):
    """Types of tasks that can be performed by LLMs."""
    COMPLETION = "completion"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    QA = "qa"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Cost estimates for model pricing (in dollars per million tokens)
COST_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3-mini": {"input": 1.10, "output": 4.40},
    
    # Claude models
    "claude-3-7-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.0},
    
    # DeepSeek models
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    
    # Gemini models
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.35, "output": 1.05},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0},
    "gemini-2.5-pro-exp-03-25": {"input": 0.0, "output": 0.5},
}


# Default models by provider
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-haiku-latest",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-2.0-flash-lite"
}


# Emoji mapping by log type and action
EMOJI_MAP = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "critical": "🔥",
    
    # Component-specific emojis
    "server": "🖥️",
    "cache": "💾",
    "provider": "🔌",
    "request": "📤",
    "response": "📥",
    "processing": "⚙️",
    "model": "🧠",
    "config": "🔧",
    "token": "🔢",
    "cost": "💰",
    "time": "⏱️",
    
    # Task-specific emojis
    "completion": "✍️",
    "summarization": "📝",
    "extraction": "🔍",
    "generation": "🎨",
    "analysis": "📊",
    "classification": "🏷️",
    "translation": "🔄",
    "qa": "❓",
    
    # Provider-specific emojis
    Provider.OPENAI: "🟢",
    Provider.ANTHROPIC: "🟣",
    Provider.DEEPSEEK: "🟠", 
    Provider.GEMINI: "🔵"
}