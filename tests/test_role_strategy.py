"""
Unit tests for RoleStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.role import RoleStrategy


class TestRoleStrategy:
    """Tests for RoleStrategy class."""
    
    def test_init_without_role(self):
        """Test RoleStrategy initialization without role."""
        strategy = RoleStrategy()
        
        assert strategy.role is None
    
    def test_init_with_role(self):
        """Test RoleStrategy initialization with role."""
        strategy = RoleStrategy(role="senior engineer")
        
        assert strategy.role == "senior engineer"
    
    def test_init_with_llm_client(self):
        """Test RoleStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = RoleStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_with_default_role(self):
        """Test improve method with default role."""
        strategy = RoleStrategy()
        result = strategy.improve("Explain recursion")
        
        assert "You are" in result
        assert "an expert in the relevant field" in result
        assert "Explain recursion" in result
        assert result.startswith("You are")
    
    def test_improve_with_instance_role(self):
        """Test improve method with role set in instance."""
        strategy = RoleStrategy(role="senior software engineer")
        result = strategy.improve("Explain recursion")
        
        assert "You are" in result
        assert "senior software engineer" in result
        assert "Explain recursion" in result
    
    def test_improve_with_kwargs_role(self):
        """Test improve method with role in kwargs."""
        strategy = RoleStrategy()
        result = strategy.improve("Explain recursion", role="university professor")
        
        assert "You are" in result
        assert "university professor" in result
        assert "Explain recursion" in result
    
    def test_improve_kwargs_override_instance_role(self):
        """Test that kwargs role overrides instance role."""
        strategy = RoleStrategy(role="instance role")
        result = strategy.improve("Test prompt", role="kwargs role")
        
        assert "kwargs role" in result
        assert "instance role" not in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = RoleStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "Role Prompting"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = RoleStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

