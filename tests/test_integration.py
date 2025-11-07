"""
Integration tests for the prompt improver system.
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


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_all_strategies_work(self):
        """Test that all strategies can be used successfully."""
        improver = PromptImprover()
        test_prompt = "Explain how to optimize database queries"
        
        strategies = improver.get_available_strategies()
        
        for strategy in strategies:
            # Skip duplicates (aliases)
            if strategy in ['chain-of-thought', 'tree-of-thought', 'skeleton-of-thought']:
                continue
                
            try:
                result = improver.improve(test_prompt, strategy=strategy)
                assert isinstance(result, str)
                assert len(result) > len(test_prompt)  # Should be longer than original
                assert test_prompt in result  # Original prompt should be included
            except Exception as e:
                if HAS_PYTEST:
                    pytest.fail(f"Strategy '{strategy}' failed: {str(e)}")
                else:
                    self.fail(f"Strategy '{strategy}' failed: {str(e)}")
    
    def test_strategy_combinations(self):
        """Test that strategies can be chained conceptually."""
        improver = PromptImprover()
        prompt = "Design a microservices architecture"
        
        # Apply different strategies
        role_result = improver.improve(prompt, strategy='role', role='senior architect')
        cot_result = improver.improve(prompt, strategy='cot')
        tot_result = improver.improve(prompt, strategy='tot')
        
        # All should work independently
        assert isinstance(role_result, str)
        assert isinstance(cot_result, str)
        assert isinstance(tot_result, str)
        
        # All should contain the original prompt
        assert prompt in role_result
        assert prompt in cot_result
        assert prompt in tot_result
    
    def test_strategy_with_various_prompts(self):
        """Test strategies with various prompt types."""
        improver = PromptImprover()
        
        test_prompts = [
            "Explain recursion",
            "Debug this code: print('hello')",
            "Classify this log entry",
            "Design a REST API",
            "Optimize this SQL query",
            "What is machine learning?",
        ]
        
        for prompt in test_prompts:
            # Test with a few key strategies
            for strategy in ['role', 'cot', 'react']:
                result = improver.improve(prompt, strategy=strategy)
                assert isinstance(result, str)
                assert prompt in result
                assert len(result) > len(prompt)
    
    def test_strategy_parameters(self):
        """Test that strategy parameters work correctly."""
        improver = PromptImprover()
        prompt = "Test prompt"
        
        # Test role with custom role
        result1 = improver.improve(prompt, strategy='role', role='custom role')
        assert 'custom role' in result1
        
        # Test self-consistency with custom num_paths
        result2 = improver.improve(prompt, strategy='self-consistency', num_paths=5)
        assert '5' in result2
        
        # Test tot with custom num_branches
        result3 = improver.improve(prompt, strategy='tot', num_branches=7)
        assert '7' in result3
        
        # Test sot with custom num_points
        result4 = improver.improve(prompt, strategy='sot', num_points=10)
        assert '10' in result4
        
        # Test react with custom domain
        result5 = improver.improve(prompt, strategy='react', domain='custom domain')
        assert 'custom domain' in result5

