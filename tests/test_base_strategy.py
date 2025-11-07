"""
Unit tests for BaseStrategy abstract class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    
    def improve(self, prompt: str, **kwargs) -> str:
        """Improve prompt."""
        return f"Improved: {prompt}"
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Test Strategy"


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class."""
    
    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_concrete_strategy_initialization_without_llm_client(self):
        """Test concrete strategy initialization without LLM client."""
        strategy = ConcreteStrategy()
        
        assert strategy.llm_client is not None
        assert hasattr(strategy.llm_client, 'invoke')
    
    def test_concrete_strategy_initialization_with_llm_client(self):
        """Test concrete strategy initialization with LLM client."""
        mock_client = Mock()
        strategy = ConcreteStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_concrete_strategy_improve(self):
        """Test concrete strategy improve method."""
        strategy = ConcreteStrategy()
        result = strategy.improve("test prompt")
        
        assert result == "Improved: test prompt"
    
    def test_concrete_strategy_get_strategy_name(self):
        """Test concrete strategy get_strategy_name method."""
        strategy = ConcreteStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Test Strategy"
    
    def test_concrete_strategy_improve_with_kwargs(self):
        """Test concrete strategy improve method with kwargs."""
        strategy = ConcreteStrategy()
        result = strategy.improve("test prompt", extra_param="value")
        
        assert result == "Improved: test prompt"

