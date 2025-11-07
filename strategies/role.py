try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import Optional


class RoleStrategy(BaseStrategy):
    """Apply role prompting by structuring prompts with role context."""
    
    def __init__(self, role: Optional[str] = None, llm_client=None):
        """
        Initialize Role Strategy.
        
        Args:
            role: The role to assign (e.g., "senior software engineer", "university professor").
                  If None, a generic expert role will be used.
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.role = role
    
    def improve(self, prompt: str, role: Optional[str] = None, **kwargs) -> str:
        """
        Improve prompt by adding role context.
        
        Args:
            prompt: Original prompt
            role: Optional role override
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with role context
        """
        effective_role = role or self.role or "an expert in the relevant field"
        return f"You are {effective_role}. Provide clear, professional, and contextually appropriate responses.\n\n{prompt}"
    
    def get_strategy_name(self) -> str:
        return "Role Prompting"

