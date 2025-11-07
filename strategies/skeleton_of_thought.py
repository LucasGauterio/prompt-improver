try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import Optional
from langchain_core.prompts import PromptTemplate


class SkeletonOfThoughtStrategy(BaseStrategy):
    """Apply Skeleton of Thought by structuring prompts with two-phase approach."""
    
    def __init__(self, num_points: int = 5, llm_client=None):
        """
        Initialize Skeleton of Thought Strategy.
        
        Args:
            num_points: Number of skeleton points to generate (default: 5)
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.num_points = num_points
    
    def improve(self, prompt: str, num_points: Optional[int] = None, **kwargs) -> str:
        """
        Improve prompt by adding Skeleton of Thought structure.
        
        Args:
            prompt: Original prompt
            num_points: Number of skeleton points (overrides self.num_points)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with SoT structure
        """
        effective_num_points = num_points or self.num_points
        sot_template = PromptTemplate(
            input_variables=["prompt", "num_points"],
            template="""Task: {prompt}

Step 1 - Generate Skeleton:
Create {num_points} concise bullet points or section headers that outline the main points. Do not expand yet.

Step 2 - Expand Skeleton:
For each bullet point or section header from Step 1, expand it into a clear and detailed explanation with examples and technical details.

Skeleton Generation:"""
        )
        return sot_template.format(
            prompt=prompt,
            num_points=effective_num_points
        )
    
    def get_strategy_name(self) -> str:
        return "Skeleton of Thought"

