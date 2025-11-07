"""
Tests for the PromptImprover class.
"""

import sys
import unittest
from pathlib import Path

# Try to import pytest, fallback to unittest if not available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from improver import PromptImprover


class TestPromptImprover(unittest.TestCase):
    """Tests for PromptImprover class."""
    
    def test_initialization(self):
        """Test PromptImprover initialization."""
        improver = PromptImprover()
        assert improver is not None
        assert len(improver.strategies) > 0
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        improver = PromptImprover()
        strategies = improver.get_available_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert 'role' in strategies
        assert 'cot' in strategies
        assert 'react' in strategies
    
    def test_improve_with_role_strategy(self):
        """Test improving prompt with role strategy."""
        improver = PromptImprover()
        prompt = "Explain recursion"
        result = improver.improve(prompt, strategy='role', role='senior engineer')
        
        assert isinstance(result, str)
        assert prompt in result
        assert 'senior engineer' in result or 'expert' in result.lower()
    
    def test_improve_with_cot_strategy(self):
        """Test improving prompt with chain of thought strategy."""
        improver = PromptImprover()
        prompt = "Solve 2 + 2"
        result = improver.improve(prompt, strategy='cot')
        
        assert isinstance(result, str)
        assert prompt in result
        assert 'step' in result.lower() or 'Step' in result
    
    def test_improve_with_few_shot_strategy(self):
        """Test improving prompt with few-shot strategy."""
        improver = PromptImprover()
        prompt = "Classify this log"
        result = improver.improve(prompt, strategy='few-shot')
        
        assert isinstance(result, str)
        assert prompt in result
        assert 'example' in result.lower() or 'Example' in result
    
    def test_improve_with_self_consistency_strategy(self):
        """Test improving prompt with self-consistency strategy."""
        improver = PromptImprover()
        prompt = "How many queries?"
        result = improver.improve(prompt, strategy='self-consistency', num_paths=3)
        
        assert isinstance(result, str)
        assert prompt in result
        assert '3' in result
    
    def test_improve_with_tot_strategy(self):
        """Test improving prompt with tree of thought strategy."""
        improver = PromptImprover()
        prompt = "Design a system"
        result = improver.improve(prompt, strategy='tot', num_branches=3)
        
        assert isinstance(result, str)
        assert prompt in result
        assert '3' in result
    
    def test_improve_with_sot_strategy(self):
        """Test improving prompt with skeleton of thought strategy."""
        improver = PromptImprover()
        prompt = "How to optimize?"
        result = improver.improve(prompt, strategy='sot', num_points=5)
        
        assert isinstance(result, str)
        assert prompt in result
        assert '5' in result
        assert 'skeleton' in result.lower() or 'Skeleton' in result
    
    def test_improve_with_react_strategy(self):
        """Test improving prompt with ReAct strategy."""
        improver = PromptImprover()
        prompt = "Debug this API"
        result = improver.improve(prompt, strategy='react', domain='software engineering')
        
        assert isinstance(result, str)
        assert prompt in result
        assert 'Thought' in result or 'thought' in result.lower()
        assert 'Action' in result or 'action' in result.lower()
    
    def test_improve_with_invalid_strategy(self):
        """Test improving prompt with invalid strategy."""
        improver = PromptImprover()
        prompt = "Test prompt"
        
        if HAS_PYTEST:
            with pytest.raises(ValueError, match="Unknown strategy"):
                improver.improve(prompt, strategy='invalid-strategy')
        else:
            with self.assertRaises(ValueError) as context:
                improver.improve(prompt, strategy='invalid-strategy')
            self.assertIn("Unknown strategy", str(context.exception))
    
    def test_improve_with_strategy_aliases(self):
        """Test improving prompt with strategy aliases."""
        improver = PromptImprover()
        prompt = "Test prompt"
        
        # Test 'chain-of-thought' alias
        result1 = improver.improve(prompt, strategy='chain-of-thought')
        result2 = improver.improve(prompt, strategy='cot')
        assert result1 == result2
        
        # Test 'tree-of-thought' alias
        result3 = improver.improve(prompt, strategy='tree-of-thought')
        result4 = improver.improve(prompt, strategy='tot')
        assert result3 == result4
        
        # Test 'skeleton-of-thought' alias
        result5 = improver.improve(prompt, strategy='skeleton-of-thought')
        result6 = improver.improve(prompt, strategy='sot')
        assert result5 == result6
    
    def test_get_strategy_info(self):
        """Test getting strategy info."""
        improver = PromptImprover()
        info = improver.get_strategy_info('role')
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'key' in info
        assert info['name'] == 'Role Prompting'
        assert info['key'] == 'role'
    
    def test_get_strategy_info_invalid(self):
        """Test getting strategy info for invalid strategy."""
        improver = PromptImprover()
        
        if HAS_PYTEST:
            with pytest.raises(ValueError, match="Unknown strategy"):
                improver.get_strategy_info('invalid-strategy')
        else:
            with self.assertRaises(ValueError) as context:
                improver.get_strategy_info('invalid-strategy')
            self.assertIn("Unknown strategy", str(context.exception))
    
    def test_strategy_case_insensitive(self):
        """Test that strategy names are case-insensitive."""
        improver = PromptImprover()
        prompt = "Test prompt"
        
        result1 = improver.improve(prompt, strategy='ROLE')
        result2 = improver.improve(prompt, strategy='role')
        result3 = improver.improve(prompt, strategy='Role')
        
        # All should produce similar results (may differ slightly due to role handling)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)

