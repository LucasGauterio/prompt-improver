"""
Unit tests for PromptImprover class.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from improver import PromptImprover
from llm_client import LLMClient


class TestPromptImprover:
    """Tests for PromptImprover class."""
    
    def test_init_without_llm_client(self):
        """Test PromptImprover initialization without LLM client."""
        with patch('improver.LLMClient') as mock_llm_client_class:
            mock_client = MagicMock()
            mock_llm_client_class.return_value = mock_client
            
            improver = PromptImprover()
            
            assert improver is not None
            assert len(improver.strategies) > 0
            assert 'role' in improver.strategies
            assert 'cot' in improver.strategies
            assert 'react' in improver.strategies
    
    def test_init_with_llm_client(self):
        """Test PromptImprover initialization with LLM client."""
        mock_client = Mock(spec=LLMClient)
        improver = PromptImprover(llm_client=mock_client)
        
        assert improver.llm_client == mock_client
        assert len(improver.strategies) > 0
    
    def test_init_with_provider(self):
        """Test PromptImprover initialization with provider."""
        with patch('improver.LLMClient') as mock_llm_client_class:
            mock_client = MagicMock()
            mock_llm_client_class.return_value = mock_client
            
            improver = PromptImprover(provider="openai")
            
            mock_llm_client_class.assert_called_once_with(provider="openai", model_name=None)
    
    def test_init_with_model_name(self):
        """Test PromptImprover initialization with model name."""
        with patch('improver.LLMClient') as mock_llm_client_class:
            mock_client = MagicMock()
            mock_llm_client_class.return_value = mock_client
            
            improver = PromptImprover(provider="openai", model_name="gpt-4")
            
            mock_llm_client_class.assert_called_once_with(provider="openai", model_name="gpt-4")
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            strategies = improver.get_available_strategies()
            
            assert isinstance(strategies, list)
            assert len(strategies) > 0
            assert 'role' in strategies
            assert 'cot' in strategies
            assert 'react' in strategies
            assert 'few-shot' in strategies
    
    def test_improve_with_role_strategy(self):
        """Test improving prompt with role strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("Explain recursion", strategy='role', role='senior engineer')
            
            assert isinstance(result, str)
            assert "Explain recursion" in result
            assert 'senior engineer' in result or 'expert' in result.lower()
    
    def test_improve_with_cot_strategy(self):
        """Test improving prompt with chain of thought strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("Solve 2 + 2", strategy='cot')
            
            assert isinstance(result, str)
            assert "Solve 2 + 2" in result
            assert 'step' in result.lower() or 'Step' in result
    
    def test_improve_with_few_shot_strategy(self):
        """Test improving prompt with few-shot strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("Classify this log", strategy='few-shot')
            
            assert isinstance(result, str)
            assert "Classify this log" in result
    
    def test_improve_with_self_consistency_strategy(self):
        """Test improving prompt with self-consistency strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("How many queries?", strategy='self-consistency', num_paths=3)
            
            assert isinstance(result, str)
            assert "How many queries?" in result
            assert '3' in result
    
    def test_improve_with_tot_strategy(self):
        """Test improving prompt with tree of thought strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("Design a system", strategy='tot', num_branches=3)
            
            assert isinstance(result, str)
            assert "Design a system" in result
            assert '3' in result
    
    def test_improve_with_sot_strategy(self):
        """Test improving prompt with skeleton of thought strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("How to optimize?", strategy='sot', num_points=5)
            
            assert isinstance(result, str)
            assert "How to optimize?" in result
            assert '5' in result
            assert 'skeleton' in result.lower() or 'Skeleton' in result
    
    def test_improve_with_react_strategy(self):
        """Test improving prompt with ReAct strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            result = improver.improve("Debug this API", strategy='react', domain='software engineering')
            
            assert isinstance(result, str)
            assert "Debug this API" in result
            assert 'Thought' in result or 'thought' in result.lower()
            assert 'Action' in result or 'action' in result.lower()
    
    def test_improve_with_invalid_strategy(self):
        """Test improving prompt with invalid strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            
            with pytest.raises(ValueError, match="Unknown strategy"):
                improver.improve("Test prompt", strategy='invalid-strategy')
    
    def test_improve_with_strategy_aliases(self):
        """Test improving prompt with strategy aliases."""
        with patch('improver.LLMClient'):
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
    
    def test_improve_case_insensitive_strategy(self):
        """Test that strategy names are case-insensitive."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            prompt = "Test prompt"
            
            result1 = improver.improve(prompt, strategy='ROLE')
            result2 = improver.improve(prompt, strategy='role')
            result3 = improver.improve(prompt, strategy='Role')
            
            # All should produce similar results
            assert isinstance(result1, str)
            assert isinstance(result2, str)
            assert isinstance(result3, str)
    
    def test_get_strategy_info(self):
        """Test getting strategy info."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            info = improver.get_strategy_info('role')
            
            assert isinstance(info, dict)
            assert 'name' in info
            assert 'key' in info
            assert info['name'] == 'Role Prompting'
            assert info['key'] == 'role'
    
    def test_get_strategy_info_invalid(self):
        """Test getting strategy info for invalid strategy."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            
            with pytest.raises(ValueError, match="Unknown strategy"):
                improver.get_strategy_info('invalid-strategy')
    
    def test_get_strategy_info_case_insensitive(self):
        """Test that get_strategy_info is case-insensitive."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            
            info1 = improver.get_strategy_info('ROLE')
            info2 = improver.get_strategy_info('role')
            info3 = improver.get_strategy_info('Role')
            
            assert info1 == info2 == info3
    
    def test_all_strategies_initialized(self):
        """Test that all strategies are properly initialized."""
        with patch('improver.LLMClient'):
            improver = PromptImprover()
            
            expected_strategies = [
                'role', 'few-shot', 'cot', 'chain-of-thought',
                'self-consistency', 'tot', 'tree-of-thought',
                'sot', 'skeleton-of-thought', 'react'
            ]
            
            for strategy in expected_strategies:
                assert strategy in improver.strategies
                assert hasattr(improver.strategies[strategy], 'improve')
                assert hasattr(improver.strategies[strategy], 'get_strategy_name')

