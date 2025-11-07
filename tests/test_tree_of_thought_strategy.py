"""
Unit tests for TreeOfThoughtStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.tree_of_thought import TreeOfThoughtStrategy


class TestTreeOfThoughtStrategy:
    """Tests for TreeOfThoughtStrategy class."""
    
    def test_init_with_default_branches(self):
        """Test TreeOfThoughtStrategy initialization with default num_branches."""
        strategy = TreeOfThoughtStrategy()
        
        assert strategy.num_branches == 3
    
    def test_init_with_custom_branches(self):
        """Test TreeOfThoughtStrategy initialization with custom num_branches."""
        strategy = TreeOfThoughtStrategy(num_branches=5)
        
        assert strategy.num_branches == 5
    
    def test_init_with_llm_client(self):
        """Test TreeOfThoughtStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = TreeOfThoughtStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_with_default_branches(self):
        """Test improve method with default num_branches."""
        strategy = TreeOfThoughtStrategy()
        result = strategy.improve("Design a scalable system")
        
        assert "Design a scalable system" in result
        assert "3" in result
        assert "approach" in result.lower() or "solution" in result.lower()
    
    def test_improve_with_instance_branches(self):
        """Test improve method with num_branches set in instance."""
        strategy = TreeOfThoughtStrategy(num_branches=5)
        result = strategy.improve("Test prompt")
        
        assert "5" in result
        assert "Test prompt" in result
    
    def test_improve_with_kwargs_branches(self):
        """Test improve method with num_branches in kwargs."""
        strategy = TreeOfThoughtStrategy()
        result = strategy.improve("Test prompt", num_branches=4)
        
        assert "4" in result
        assert "Test prompt" in result
    
    def test_improve_kwargs_override_instance_branches(self):
        """Test that kwargs num_branches overrides instance num_branches."""
        strategy = TreeOfThoughtStrategy(num_branches=3)
        result = strategy.improve("Test prompt", num_branches=7)
        
        assert "7" in result
    
    def test_improve_contains_instructions(self):
        """Test that improve method contains tree of thought instructions."""
        strategy = TreeOfThoughtStrategy()
        result = strategy.improve("Test prompt")
        
        assert "approach" in result.lower() or "Approach" in result
        assert "feasibility" in result.lower() or "Feasibility" in result
        assert "advantage" in result.lower() or "Advantage" in result
        assert "disadvantage" in result.lower() or "Disadvantage" in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = TreeOfThoughtStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Tree of Thought"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = TreeOfThoughtStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

