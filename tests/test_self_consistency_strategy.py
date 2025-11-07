"""
Unit tests for SelfConsistencyStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.self_consistency import SelfConsistencyStrategy


class TestSelfConsistencyStrategy:
    """Tests for SelfConsistencyStrategy class."""
    
    def test_init_with_default_paths(self):
        """Test SelfConsistencyStrategy initialization with default num_paths."""
        strategy = SelfConsistencyStrategy()
        
        assert strategy.num_paths == 3
    
    def test_init_with_custom_paths(self):
        """Test SelfConsistencyStrategy initialization with custom num_paths."""
        strategy = SelfConsistencyStrategy(num_paths=5)
        
        assert strategy.num_paths == 5
    
    def test_init_with_llm_client(self):
        """Test SelfConsistencyStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = SelfConsistencyStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_with_default_paths(self):
        """Test improve method with default num_paths."""
        strategy = SelfConsistencyStrategy()
        result = strategy.improve("How many queries will this execute?")
        
        assert "How many queries will this execute?" in result
        assert "3" in result
        assert "reasoning" in result.lower() or "path" in result.lower()
    
    def test_improve_with_instance_paths(self):
        """Test improve method with num_paths set in instance."""
        strategy = SelfConsistencyStrategy(num_paths=5)
        result = strategy.improve("Test prompt")
        
        assert "5" in result
        assert "Test prompt" in result
    
    def test_improve_with_kwargs_paths(self):
        """Test improve method with num_paths in kwargs."""
        strategy = SelfConsistencyStrategy()
        result = strategy.improve("Test prompt", num_paths=4)
        
        assert "4" in result
        assert "Test prompt" in result
    
    def test_improve_kwargs_override_instance_paths(self):
        """Test that kwargs num_paths overrides instance num_paths."""
        strategy = SelfConsistencyStrategy(num_paths=3)
        result = strategy.improve("Test prompt", num_paths=7)
        
        assert "7" in result
        # Check that "3" appears only in instruction numbering, not as num_paths value
        # The num_paths value should be "7", not "3"
        assert "Generate 7 different" in result
        assert "Generate 3 different" not in result
    
    def test_improve_contains_instructions(self):
        """Test that improve method contains self-consistency instructions."""
        strategy = SelfConsistencyStrategy()
        result = strategy.improve("Test prompt")
        
        assert "different" in result.lower() or "independent" in result.lower()
        assert "consistent" in result.lower() or "Consistent" in result
        assert "path" in result.lower() or "Path" in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = SelfConsistencyStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Self-Consistency"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = SelfConsistencyStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

