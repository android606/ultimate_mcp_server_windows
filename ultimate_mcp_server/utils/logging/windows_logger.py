"""
Windows-safe logger that filters out emojis and replaces them with ASCII equivalents.

This module provides a Logger subclass specifically designed for Windows environments
where Unicode emoji characters can cause subprocess encoding issues.
"""
import re
from typing import Any, Dict, Optional

from .logger import Logger


class WindowsLogger(Logger):
    """Windows-safe logger that automatically converts emojis to ASCII equivalents."""
    
    # Emoji to ASCII mapping for common log emojis
    EMOJI_TO_ASCII = {
        # Log levels
        "ℹ️": "[INFO]",
        "🔍": "[DEBUG]", 
        "⚠️": "[WARN]",
        "❌": "[ERROR]",
        "🚨": "[CRIT]",
        "✅": "[OK]",
        "📍": "[TRACE]",
        
        # Status
        "🔄": "[RUN]",
        "⏳": "[WAIT]",
        "🏁": "[DONE]",
        "👎": "[FAIL]",
        "🚀": "[START]",
        "🛑": "[STOP]",
        "🔁": "[RESTART]",
        "📥": "[LOAD]",
        "📤": "[SAVE]",
        "🚫": "[CANCEL]",
        "⏱️": "[TIME]",
        "⏭️": "[SKIP]",
        
        # Operations
        "➡️": "[REQ]",
        "⬅️": "[RESP]", 
        "⚙️": "[PROC]",
        "🔒": "[AUTH]",
        "🔑": "[AUTHZ]",
        "✔️": "[VALID]",
        "🔌": "[CONN]",
        "📝": "[UPD]",
        
        # Components
        "☁️": "[PROV]",
        "🔀": "[ROUTE]",
        "📦": "[CACHE]",
        "🌐": "[API]",
        "📡": "[MCP]",
        "🔧": "[UTIL]",
        
        # Results
        "🎯": "[FOUND]",
        "◐": "[PARTIAL]",
        "❓": "[?]",
        
        # System
        "🔆": "[STARTUP]",
        "🔅": "[SHUTDOWN]",
        "🧱": "[DEP]",
        "🏷️": "[VER]",
        "🆕": "[UPDATE]",
        "⛔": "[ERR]",
        
        # User interaction
        "⌨️": "[INPUT]",
        "📺": "[OUTPUT]",
        "💡": "[HINT]",
        "📋": "[EXAMPLE]",
        "💬": "[ANSWER]",
        
        # Time
        "📅": "[SCHED]",
        "⏰": "[DELAY]",
        "⌛": "[OVER]",
        
        # Provider specific
        "🟢": "[OPENAI]",
        "🟣": "[ANTHROPIC]",
        "🟠": "[DEEPSEEK]",
        "🔵": "[GEMINI]",
        "🌐": "[OPENROUTER]",
        "🦙": "[OLLAMA]",
        "⚡": "[GROK]",
        "🤝": "[TOGETHER]",
        
        # Common symbols that might cause issues
        "🔥": "[FIRE]",
        "💻": "[COMP]",
        "🐍": "[PYTHON]",
        "📋": "[PROC]",
        "🌟": "[STAR]",
        "💰": "[COST]",
        "🧠": "[MODEL]",
        "🛠️": "[TOOL]",
        "📄": "[DOC]",
        "🗃️": "[DB]",
        "🗄️": "[DB]",
    }
    
    def __init__(self, name: str, component: Optional[str] = None):
        """Initialize Windows-safe logger.
        
        Args:
            name: Logger name
            component: Default component name
        """
        super().__init__(name, component)
        
    def _filter_emojis(self, message: str) -> str:
        """Filter emojis from a message and replace with ASCII equivalents.
        
        Args:
            message: Original message that might contain emojis
            
        Returns:
            Message with emojis replaced by ASCII equivalents
        """
        if not message:
            return message
            
        # Replace known emojis with ASCII equivalents
        filtered_message = message
        for emoji, ascii_equiv in self.EMOJI_TO_ASCII.items():
            filtered_message = filtered_message.replace(emoji, ascii_equiv)
        
        # Remove any remaining emoji characters using regex
        # This regex matches most Unicode emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002700-\U000027BF"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U0000FE00-\U0000FE0F"  # variation selectors
            "]+", 
            flags=re.UNICODE
        )
        
        # Replace any remaining emojis with [?]
        filtered_message = emoji_pattern.sub("[?]", filtered_message)
        
        return filtered_message
    
    def _log(
        self,
        level: str,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        emoji: Optional[str] = None,
        emoji_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_detailed_formatter: bool = False,
        exception_info: Optional[Any] = None,
        stack_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Override _log to filter emojis from message and emoji parameters.
        
        Args:
            Same as parent Logger._log method
        """
        # Filter emojis from the message
        filtered_message = self._filter_emojis(message)
        
        # Filter emoji parameter if provided
        filtered_emoji = None
        if emoji:
            filtered_emoji = self.EMOJI_TO_ASCII.get(emoji, "[?]")
        
        # Call parent _log with filtered content
        super()._log(
            level=level,
            message=filtered_message,
            component=component,
            operation=operation,
            emoji=filtered_emoji,
            emoji_key=emoji_key,
            context=context,
            use_detailed_formatter=use_detailed_formatter,
            exception_info=exception_info,
            stack_info=stack_info,
            extra=extra,
        )


def create_windows_logger(name: str, component: Optional[str] = None) -> WindowsLogger:
    """Create a Windows-safe logger instance.
    
    Args:
        name: Logger name
        component: Default component name
        
    Returns:
        WindowsLogger instance
    """
    return WindowsLogger(name, component) 