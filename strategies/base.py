from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
try:
    from ..llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient


class BaseStrategy(ABC):
    """Base class for all prompt improvement strategies."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the strategy with an optional LLM client.
        
        Args:
            llm_client: Optional LLMClient instance. If None, creates a new one.
        """
        self.llm_client = llm_client or LLMClient()
    
    @abstractmethod
    def improve(self, prompt: str, **kwargs) -> str:
        """
        Improve a prompt by applying the strategy.
        
        Args:
            prompt: The original prompt to improve
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            The improved prompt
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy."""
        pass

