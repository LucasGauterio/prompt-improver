try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from langchain_core.prompts import PromptTemplate


class ChainOfThoughtStrategy(BaseStrategy):
    """Apply Chain of Thought reasoning by structuring prompts with step-by-step instructions."""
    
    def improve(self, prompt: str, **kwargs) -> str:
        """
        Improve prompt by adding Chain of Thought structure.
        
        Args:
            prompt: Original prompt
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with CoT structure
        """
        cot_template = PromptTemplate(
            input_variables=["prompt"],
            template="""Let's think step by step.

Task: {prompt}

Instructions:
1. Break down the problem into smaller, manageable parts
2. Think through each step carefully
3. Show your reasoning for each step
4. Provide a clear final answer after showing your reasoning

Step-by-step reasoning:"""
        )
        return cot_template.format(prompt=prompt)
    
    def get_strategy_name(self) -> str:
        return "Chain of Thought"

