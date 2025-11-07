try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import List, Dict, Optional
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


class FewShotStrategy(BaseStrategy):
    """Apply few-shot learning by structuring prompts with examples."""
    
    def __init__(self, examples: Optional[List[Dict[str, str]]] = None, llm_client=None):
        """
        Initialize Few-Shot Strategy.
        
        Args:
            examples: List of example dicts with 'input' and 'output' keys.
                      If None, examples will be generated using LLM.
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.examples = examples or []
    
    def improve(self, prompt: str, examples: Optional[List[Dict[str, str]]] = None, num_examples: int = 2, **kwargs) -> str:
        """
        Improve prompt by adding few-shot examples.
        
        Args:
            prompt: Original prompt
            examples: Optional list of examples (overrides self.examples)
            num_examples: Number of examples to include (if examples not provided)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with examples
        """
        effective_examples = examples or self.examples
        
        if effective_examples:
            example_prompt = PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}"
            )
            
            few_shot_prompt = FewShotPromptTemplate(
                examples=effective_examples[:num_examples],
                example_prompt=example_prompt,
                prefix="Here are some examples:\n",
                suffix="\nNow, following the pattern above:\n{prompt}",
                input_variables=["prompt"]
            )
            
            return few_shot_prompt.format(prompt=prompt)
        else:
            improved_prompt = f"""Here are some examples to guide the response:

{prompt}

Follow the pattern shown in the examples above."""
            return improved_prompt
    
    def get_strategy_name(self) -> str:
        return "Few-Shot Learning"

