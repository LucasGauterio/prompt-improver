from typing import Dict, Optional
try:
    from .llm_client import LLMClient
    from .strategies import (
        BaseStrategy,
        RoleStrategy,
        FewShotStrategy,
        ChainOfThoughtStrategy,
        SelfConsistencyStrategy,
        TreeOfThoughtStrategy,
        SkeletonOfThoughtStrategy,
        ReActStrategy
    )
except ImportError:
    # Fallback for when running as a script
    from llm_client import LLMClient
    from strategies import (
        BaseStrategy,
        RoleStrategy,
        FewShotStrategy,
        ChainOfThoughtStrategy,
        SelfConsistencyStrategy,
        TreeOfThoughtStrategy,
        SkeletonOfThoughtStrategy,
        ReActStrategy
    )


class PromptImprover:
    """Main class for improving prompts using various strategies.
    
    The improved prompts are generic and framework-agnostic, suitable for use with any LLM.
    """
    
    STRATEGY_MAP: Dict[str, type] = {
        'role': RoleStrategy,
        'few-shot': FewShotStrategy,
        'cot': ChainOfThoughtStrategy,
        'chain-of-thought': ChainOfThoughtStrategy,
        'self-consistency': SelfConsistencyStrategy,
        'tot': TreeOfThoughtStrategy,
        'tree-of-thought': TreeOfThoughtStrategy,
        'sot': SkeletonOfThoughtStrategy,
        'skeleton-of-thought': SkeletonOfThoughtStrategy,
        'react': ReActStrategy,
    }
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "openai",
        model_name: Optional[str] = None
    ):
        """
        Initialize the PromptImprover with default strategy instances.
        
        Args:
            llm_client: Optional LLMClient instance to share across strategies.
                       If None, creates a new client with the specified provider.
            provider: LLM provider to use if llm_client is None (default: "openai")
            model_name: Model name to use if llm_client is None (default: provider defaults)
        """
        # Share LLM client across all strategies for efficiency
        if llm_client is None:
            self.llm_client = LLMClient(provider=provider, model_name=model_name)
        else:
            self.llm_client = llm_client
        self.strategies: Dict[str, BaseStrategy] = {
            'role': RoleStrategy(llm_client=self.llm_client),
            'few-shot': FewShotStrategy(llm_client=self.llm_client),
            'cot': ChainOfThoughtStrategy(llm_client=self.llm_client),
            'chain-of-thought': ChainOfThoughtStrategy(llm_client=self.llm_client),
            'self-consistency': SelfConsistencyStrategy(llm_client=self.llm_client),
            'tot': TreeOfThoughtStrategy(llm_client=self.llm_client),
            'tree-of-thought': TreeOfThoughtStrategy(llm_client=self.llm_client),
            'sot': SkeletonOfThoughtStrategy(llm_client=self.llm_client),
            'skeleton-of-thought': SkeletonOfThoughtStrategy(llm_client=self.llm_client),
            'react': ReActStrategy(llm_client=self.llm_client),
        }
    
    def improve(self, prompt: str, strategy: str, **kwargs) -> str:
        """
        Improve a prompt using the specified strategy.
        
        Args:
            prompt: The original prompt to improve
            strategy: Strategy name (e.g., 'role', 'cot', 'react')
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            The improved prompt
            
        Raises:
            ValueError: If strategy is not recognized
        """
        strategy_lower = strategy.lower()
        
        if strategy_lower not in self.strategies:
            available = ', '.join(self.strategies.keys())
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Available strategies: {available}"
            )
        
        strategy_instance = self.strategies[strategy_lower]
        return strategy_instance.improve(prompt, **kwargs)
    
    def get_available_strategies(self) -> list:
        """Return list of available strategy names."""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, str]:
        """
        Get information about a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Dict with strategy name and description
        """
        strategy_lower = strategy.lower()
        
        if strategy_lower not in self.strategies:
            raise ValueError(f"Unknown strategy: '{strategy}'")
        
        strategy_instance = self.strategies[strategy_lower]
        return {
            'name': strategy_instance.get_strategy_name(),
            'key': strategy_lower
        }

