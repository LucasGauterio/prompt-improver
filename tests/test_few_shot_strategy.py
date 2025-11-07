"""
Unit tests for FewShotStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.few_shot import FewShotStrategy


class TestFewShotStrategy:
    """Tests for FewShotStrategy class."""
    
    def test_init_without_examples(self):
        """Test FewShotStrategy initialization without examples."""
        strategy = FewShotStrategy()
        
        assert strategy.examples == []
    
    def test_init_with_examples(self):
        """Test FewShotStrategy initialization with examples."""
        examples = [
            {"input": "Error 404", "output": "Warning"},
            {"input": "Disk full", "output": "Critical"}
        ]
        strategy = FewShotStrategy(examples=examples)
        
        assert len(strategy.examples) == 2
        assert strategy.examples[0]["input"] == "Error 404"
    
    def test_init_with_llm_client(self):
        """Test FewShotStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = FewShotStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_without_examples(self):
        """Test improve method without examples."""
        strategy = FewShotStrategy()
        result = strategy.improve("Classify this log")
        
        assert "Classify this log" in result
        assert "examples" in result.lower()
        assert "pattern" in result.lower()
    
    def test_improve_with_instance_examples(self):
        """Test improve method with examples set in instance."""
        examples = [
            {"input": "Error 404", "output": "Warning"},
            {"input": "Disk full", "output": "Critical"}
        ]
        strategy = FewShotStrategy(examples=examples)
        result = strategy.improve("Classify this log")
        
        assert "Error 404" in result
        assert "Warning" in result
        assert "Disk full" in result
        assert "Critical" in result
        assert "Classify this log" in result
    
    def test_improve_with_kwargs_examples(self):
        """Test improve method with examples in kwargs."""
        strategy = FewShotStrategy()
        examples = [
            {"input": "Test input", "output": "Test output"}
        ]
        result = strategy.improve("Classify this log", examples=examples)
        
        assert "Test input" in result
        assert "Test output" in result
        assert "Classify this log" in result
    
    def test_improve_kwargs_override_instance_examples(self):
        """Test that kwargs examples override instance examples."""
        instance_examples = [{"input": "Instance", "output": "Example"}]
        kwargs_examples = [{"input": "Kwargs", "output": "Example"}]
        
        strategy = FewShotStrategy(examples=instance_examples)
        result = strategy.improve("Test", examples=kwargs_examples)
        
        assert "Kwargs" in result
        assert "Instance" not in result
    
    def test_improve_with_num_examples(self):
        """Test improve method with num_examples parameter."""
        examples = [
            {"input": "Input 1", "output": "Output 1"},
            {"input": "Input 2", "output": "Output 2"},
            {"input": "Input 3", "output": "Output 3"}
        ]
        strategy = FewShotStrategy(examples=examples)
        result = strategy.improve("Test", num_examples=2)
        
        # Should only include first 2 examples
        assert "Input 1" in result
        assert "Input 2" in result
        assert "Input 3" not in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = FewShotStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Few-Shot Learning"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = FewShotStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

