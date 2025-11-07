"""
Unit tests for ReActStrategy class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.react import ReActStrategy


class TestReActStrategy:
    """Tests for ReActStrategy class."""
    
    def test_init_without_domain(self):
        """Test ReActStrategy initialization without domain."""
        strategy = ReActStrategy()
        
        assert strategy.domain is None
    
    def test_init_with_domain(self):
        """Test ReActStrategy initialization with domain."""
        strategy = ReActStrategy(domain="software engineering")
        
        assert strategy.domain == "software engineering"
    
    def test_init_with_llm_client(self):
        """Test ReActStrategy initialization with LLM client."""
        mock_client = Mock()
        strategy = ReActStrategy(llm_client=mock_client)
        
        assert strategy.llm_client == mock_client
    
    def test_improve_without_domain(self):
        """Test improve method without domain."""
        strategy = ReActStrategy()
        result = strategy.improve("Debug this API endpoint")
        
        assert "Debug this API endpoint" in result
        assert "Thought:" in result or "thought" in result.lower()
        assert "Action:" in result or "action" in result.lower()
        assert "Observation:" in result or "observation" in result.lower()
        assert "Final Answer:" in result
    
    def test_improve_with_instance_domain(self):
        """Test improve method with domain set in instance."""
        strategy = ReActStrategy(domain="software engineering")
        result = strategy.improve("Debug this API endpoint")
        
        assert "software engineering" in result
        assert "Debug this API endpoint" in result
    
    def test_improve_with_kwargs_domain(self):
        """Test improve method with domain in kwargs."""
        strategy = ReActStrategy()
        result = strategy.improve("Debug this API endpoint", domain="backend engineering")
        
        assert "backend engineering" in result
        assert "Debug this API endpoint" in result
    
    def test_improve_kwargs_override_instance_domain(self):
        """Test that kwargs domain overrides instance domain."""
        strategy = ReActStrategy(domain="instance domain")
        result = strategy.improve("Test prompt", domain="kwargs domain")
        
        assert "kwargs domain" in result
        assert "instance domain" not in result
    
    def test_improve_contains_react_framework(self):
        """Test that improve method contains ReAct framework structure."""
        strategy = ReActStrategy()
        result = strategy.improve("Test prompt")
        
        assert "ReAct" in result or "react" in result.lower()
        assert "Thought" in result or "thought" in result.lower()
        assert "Action" in result or "action" in result.lower()
        assert "Observation" in result or "observation" in result.lower()
        assert "cycle" in result.lower() or "Cycle" in result
    
    def test_improve_contains_warning(self):
        """Test that improve method contains warning about not fabricating information."""
        strategy = ReActStrategy()
        result = strategy.improve("Test prompt")
        
        assert "fabricate" in result.lower() or "Fabricate" in result
        assert "information" in result.lower() or "Information" in result
    
    def test_get_strategy_name(self):
        """Test get_strategy_name method."""
        strategy = ReActStrategy()
        name = strategy.get_strategy_name()
        
        assert name == "ReAct"
    
    def test_improve_preserves_prompt(self):
        """Test that improve method preserves original prompt."""
        strategy = ReActStrategy()
        original_prompt = "Test prompt with special characters: @#$%"
        result = strategy.improve(original_prompt)
        
        assert original_prompt in result

