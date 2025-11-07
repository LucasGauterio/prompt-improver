"""
Unit tests for SkeletonOfThoughtStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.skeleton_of_thought import SkeletonOfThoughtStrategy


class TestSkeletonOfThoughtStrategy:
    """Tests for SkeletonOfThoughtStrategy class."""
    
    def test_init_with_default_points(self):
        """Test SkeletonOfThoughtStrategy initialization with default num_points."""
        strategy = SkeletonOfThoughtStrategy()
        
        assert strategy.num_points == 5
    
    def test_init_with_custom_points(self):
        """Test SkeletonOfThoughtStrategy initialization with custom num_points."""
        strategy = SkeletonOfThoughtStrategy(num_points=7)
        
        assert strategy.num_points == 7
    
    def test_init_with_llm_client(self):
        """Test SkeletonOfThoughtStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = SkeletonOfThoughtStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_with_default_points(self):
        """Test improve method with default num_points."""
        strategy = SkeletonOfThoughtStrategy()
        result = strategy.improve("How to optimize SQL queries?")
        
        assert "How to optimize SQL queries?" in result
        assert "5" in result
        assert "skeleton" in result.lower() or "Skeleton" in result
        assert "Step 1" in result or "Step 1:" in result
    
    def test_improve_with_instance_points(self):
        """Test improve method with num_points set in instance."""
        strategy = SkeletonOfThoughtStrategy(num_points=7)
        result = strategy.improve("Test prompt")
        
        assert "7" in result
        assert "Test prompt" in result
    
    def test_improve_with_kwargs_points(self):
        """Test improve method with num_points in kwargs."""
        strategy = SkeletonOfThoughtStrategy()
        result = strategy.improve("Test prompt", num_points=6)
        
        assert "6" in result
        assert "Test prompt" in result
    
    def test_improve_kwargs_override_instance_points(self):
        """Test that kwargs num_points overrides instance num_points."""
        strategy = SkeletonOfThoughtStrategy(num_points=5)
        result = strategy.improve("Test prompt", num_points=10)
        
        assert "10" in result
    
    def test_improve_contains_two_steps(self):
        """Test that improve method contains two-step structure."""
        strategy = SkeletonOfThoughtStrategy()
        result = strategy.improve("Test prompt")
        
        assert "Step 1" in result or "Step 1:" in result
        assert "Step 2" in result or "Step 2:" in result
        assert "Generate Skeleton" in result or "skeleton" in result.lower()
        assert "Expand" in result or "expand" in result.lower()
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = SkeletonOfThoughtStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Skeleton of Thought"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = SkeletonOfThoughtStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

