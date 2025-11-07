"""
Unit tests for ChainOfThoughtStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.chain_of_thought import ChainOfThoughtStrategy


class TestChainOfThoughtStrategy:
    """Tests for ChainOfThoughtStrategy class."""
    
    def test_init(self):
        """Test ChainOfThoughtStrategy initialization."""
        strategy = ChainOfThoughtStrategy()
        
        assert strategy is not None
    
    def test_init_with_llm_client(self):
        """Test ChainOfThoughtStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = ChainOfThoughtStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve(self):
        """Test improve method."""
        strategy = ChainOfThoughtStrategy()
        result = strategy.improve("Solve this math problem: 2 + 2")
        
        assert "Solve this math problem: 2 + 2" in result
        assert "step" in result.lower() or "Step" in result
        assert "Let's think step by step" in result or "step by step" in result.lower()
    
    def test_improve_contains_instructions(self):
        """Test that improve method contains step-by-step instructions."""
        strategy = ChainOfThoughtStrategy()
        result = strategy.improve("Test prompt")
        
        assert "Break down" in result or "break down" in result.lower()
        assert "reasoning" in result.lower() or "Reasoning" in result
        assert "final answer" in result.lower() or "Final Answer" in result
    
    def test_improve_with_kwargs(self):
        """Test improve method with kwargs (should be ignored)."""
        strategy = ChainOfThoughtStrategy()
        result = strategy.improve("Test prompt", extra_param="value")
        
        assert "Test prompt" in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = ChainOfThoughtStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Chain of Thought"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = ChainOfThoughtStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result
    
    def test_improve_structure(self):
        """Test that improve method returns properly structured prompt."""
        strategy = ChainOfThoughtStrategy()
        result = strategy.improve("Calculate 5 * 3")
        
        # Should have clear structure
        assert len(result) > len("Calculate 5 * 3")
        assert "Task:" in result or "task" in result.lower()

