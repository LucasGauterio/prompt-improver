try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import Optional
from langchain_core.prompts import PromptTemplate


class TreeOfThoughtStrategy(BaseStrategy):
    """Apply Tree of Thought by structuring prompts to explore multiple solution branches."""
    
    def __init__(self, num_branches: int = 3, llm_client=None):
        """
        Initialize Tree of Thought Strategy.
        
        Args:
            num_branches: Number of branches to explore (default: 3)
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.num_branches = num_branches
    
    def improve(self, prompt: str, num_branches: Optional[int] = None, **kwargs) -> str:
        """
        Improve prompt by adding Tree of Thought structure.
        
        Args:
            prompt: Original prompt
            num_branches: Number of branches to explore (overrides self.num_branches)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with ToT structure
        """
        effective_num_branches = num_branches or self.num_branches
        tot_template = PromptTemplate(
            input_variables=["prompt", "num_branches"],
            template="""Task: {prompt}

Instructions:
1. Generate at least {num_branches} different possible approaches or solutions
2. For each approach, evaluate:
   - Feasibility
   - Advantages
   - Disadvantages
3. Compare all approaches and evaluate trade-offs
4. Choose the best approach with clear reasoning

Approach Exploration:"""
        )
        return tot_template.format(
            prompt=prompt,
            num_branches=effective_num_branches
        )
    
    def get_strategy_name(self) -> str:
        return "Tree of Thought"

