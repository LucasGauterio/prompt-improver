try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import Optional
from langchain_core.prompts import PromptTemplate


class SelfConsistencyStrategy(BaseStrategy):
    """Apply Self-Consistency by structuring prompts to generate multiple reasoning paths."""
    
    def __init__(self, num_paths: int = 3, llm_client=None):
        """
        Initialize Self-Consistency Strategy.
        
        Args:
            num_paths: Number of reasoning paths to generate (default: 3)
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.num_paths = num_paths
    
    def improve(self, prompt: str, num_paths: Optional[int] = None, **kwargs) -> str:
        """
        Improve prompt by adding Self-Consistency structure.
        
        Args:
            prompt: Original prompt
            num_paths: Number of reasoning paths (overrides self.num_paths)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with Self-Consistency structure
        """
        effective_num_paths = num_paths or self.num_paths
        self_consistency_template = PromptTemplate(
            input_variables=["prompt", "num_paths"],
            template="""Task: {prompt}

Instructions:
1. Generate {num_paths} different independent reasoning paths to solve this task
2. Each path should be thorough and complete
3. After generating all paths, compare them and identify the most consistent answer
4. Explain why the chosen answer is the most consistent across all paths

Reasoning Paths:"""
        )
        return self_consistency_template.format(
            prompt=prompt,
            num_paths=effective_num_paths
        )
    
    def get_strategy_name(self) -> str:
        return "Self-Consistency"

