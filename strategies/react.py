try:
    from .base import BaseStrategy
except ImportError:
    from strategies.base import BaseStrategy
from typing import Optional
from langchain_core.prompts import PromptTemplate


class ReActStrategy(BaseStrategy):
    """Apply ReAct framework by structuring prompts with Thought/Action/Observation format."""
    
    def __init__(self, domain: Optional[str] = None, llm_client=None):
        """
        Initialize ReAct Strategy.
        
        Args:
            domain: The domain/context (e.g., "software engineering", "debugging")
            llm_client: Optional LLMClient instance
        """
        super().__init__(llm_client)
        self.domain = domain
    
    def improve(self, prompt: str, domain: Optional[str] = None, **kwargs) -> str:
        """
        Improve prompt by adding ReAct framework structure.
        
        Args:
            prompt: Original prompt
            domain: Optional domain context (overrides self.domain)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Improved prompt with ReAct framework structure
        """
        effective_domain = domain or self.domain
        domain_context = f" in the domain of {effective_domain}" if effective_domain else ""
        react_template = PromptTemplate(
            input_variables=["prompt", "domain_context"],
            template="""Task{domain_context}: {prompt}

Instructions:
Use the ReAct framework to solve this task. Alternate between reasoning (Thought) and actions (Action).

Format your response as follows:
- Thought: [Your reasoning about the current situation]
- Action: [A concrete action or step to take]
- Observation: [The result or observation from the action]
- (Repeat Thought-Action-Observation cycle as needed)
- Final Answer: [Your final answer after reasoning through the steps]

Important: Do not fabricate information not provided in the context. Base your reasoning on available information.

Begin:"""
        )
        return react_template.format(
            prompt=prompt,
            domain_context=domain_context
        )
    
    def get_strategy_name(self) -> str:
        return "ReAct"

