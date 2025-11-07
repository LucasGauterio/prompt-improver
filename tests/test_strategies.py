"""
Tests for all prompt improvement strategies.
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

from strategies import (
    RoleStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
    SelfConsistencyStrategy,
    TreeOfThoughtStrategy,
    SkeletonOfThoughtStrategy,
    ReActStrategy
)


class TestRoleStrategy(unittest.TestCase):
    """Tests for RoleStrategy."""
    
    def test_role_strategy_with_role(self):
        """Test role strategy with explicit role."""
        strategy = RoleStrategy(role="senior software engineer")
        prompt = "Explain recursion"
        result = strategy.improve(prompt)
        
        assert "You are" in result
        assert "senior software engineer" in result
        assert prompt in result
        assert result.startswith("You are")
    
    def test_role_strategy_without_role(self):
        """Test role strategy without explicit role."""
        strategy = RoleStrategy()
        prompt = "Explain recursion"
        result = strategy.improve(prompt)
        
        assert "You are" in result
        assert "expert" in result.lower()
        assert prompt in result
    
    def test_role_strategy_with_kwargs(self):
        """Test role strategy with role in kwargs."""
        strategy = RoleStrategy()
        prompt = "Explain recursion"
        result = strategy.improve(prompt, role="university professor")
        
        assert "university professor" in result
        assert prompt in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = RoleStrategy()
        assert strategy.get_strategy_name() == "Role Prompting"


class TestFewShotStrategy(unittest.TestCase):
    """Tests for FewShotStrategy."""
    
    def test_few_shot_strategy_without_examples(self):
        """Test few-shot strategy without examples."""
        strategy = FewShotStrategy()
        prompt = "Classify this log"
        result = strategy.improve(prompt)
        
        assert "examples" in result.lower()
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert prompt in result
    
    def test_few_shot_strategy_with_examples(self):
        """Test few-shot strategy with examples."""
        examples = [
            {"input": "Error 404", "output": "Warning"},
            {"input": "Disk full", "output": "Critical"}
        ]
        strategy = FewShotStrategy(examples=examples)
        prompt = "Classify this log"
        result = strategy.improve(prompt)
        
        assert "Error 404" in result
        assert "Warning" in result
        assert "Disk full" in result
        assert "Critical" in result
        assert prompt in result
    
    def test_few_shot_strategy_with_kwargs(self):
        """Test few-shot strategy with examples in kwargs."""
        strategy = FewShotStrategy()
        examples = [
            {"input": "Test input", "output": "Test output"}
        ]
        prompt = "Classify this log"
        result = strategy.improve(prompt, examples=examples)
        
        assert "Test input" in result
        assert "Test output" in result
    
    def test_few_shot_strategy_num_examples(self):
        """Test few-shot strategy with num_examples parameter."""
        strategy = FewShotStrategy()
        prompt = "Classify this log"
        result = strategy.improve(prompt, num_examples=3)
        
        # Should still show template with 2 examples by default
        assert "Example" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = FewShotStrategy()
        assert strategy.get_strategy_name() == "Few-Shot Learning"


class TestChainOfThoughtStrategy(unittest.TestCase):
    """Tests for ChainOfThoughtStrategy."""
    
    def test_chain_of_thought_strategy(self):
        """Test chain of thought strategy."""
        strategy = ChainOfThoughtStrategy()
        prompt = "Solve this math problem: 2 + 2"
        result = strategy.improve(prompt)
        
        assert prompt in result
        assert "step" in result.lower() or "Step" in result
        assert "Answer:" in result or "Final Answer:" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = ChainOfThoughtStrategy()
        assert strategy.get_strategy_name() == "Chain of Thought"


class TestSelfConsistencyStrategy(unittest.TestCase):
    """Tests for SelfConsistencyStrategy."""
    
    def test_self_consistency_strategy_default(self):
        """Test self-consistency strategy with default paths."""
        strategy = SelfConsistencyStrategy()
        prompt = "How many queries will this execute?"
        result = strategy.improve(prompt)
        
        assert prompt in result
        assert "3" in result  # Default num_paths
        assert "reasoning" in result.lower() or "path" in result.lower()
    
    def test_self_consistency_strategy_custom_paths(self):
        """Test self-consistency strategy with custom num_paths."""
        strategy = SelfConsistencyStrategy(num_paths=5)
        prompt = "How many queries will this execute?"
        result = strategy.improve(prompt)
        
        assert "5" in result
    
    def test_self_consistency_strategy_with_kwargs(self):
        """Test self-consistency strategy with num_paths in kwargs."""
        strategy = SelfConsistencyStrategy()
        prompt = "How many queries will this execute?"
        result = strategy.improve(prompt, num_paths=4)
        
        assert "4" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = SelfConsistencyStrategy()
        assert strategy.get_strategy_name() == "Self-Consistency"


class TestTreeOfThoughtStrategy(unittest.TestCase):
    """Tests for TreeOfThoughtStrategy."""
    
    def test_tree_of_thought_strategy_default(self):
        """Test tree of thought strategy with default branches."""
        strategy = TreeOfThoughtStrategy()
        prompt = "Design a scalable system"
        result = strategy.improve(prompt)
        
        assert prompt in result
        assert "3" in result  # Default num_branches
        assert "approach" in result.lower() or "solution" in result.lower()
        assert "Final Answer:" in result
    
    def test_tree_of_thought_strategy_custom_branches(self):
        """Test tree of thought strategy with custom num_branches."""
        strategy = TreeOfThoughtStrategy(num_branches=5)
        prompt = "Design a scalable system"
        result = strategy.improve(prompt)
        
        assert "5" in result
    
    def test_tree_of_thought_strategy_with_kwargs(self):
        """Test tree of thought strategy with num_branches in kwargs."""
        strategy = TreeOfThoughtStrategy()
        prompt = "Design a scalable system"
        result = strategy.improve(prompt, num_branches=4)
        
        assert "4" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = TreeOfThoughtStrategy()
        assert strategy.get_strategy_name() == "Tree of Thought"


class TestSkeletonOfThoughtStrategy(unittest.TestCase):
    """Tests for SkeletonOfThoughtStrategy."""
    
    def test_skeleton_of_thought_strategy_default(self):
        """Test skeleton of thought strategy with default points."""
        strategy = SkeletonOfThoughtStrategy()
        prompt = "How to optimize SQL queries?"
        result = strategy.improve(prompt)
        
        assert prompt in result
        assert "5" in result  # Default num_points
        assert "skeleton" in result.lower() or "Skeleton" in result
        assert "Step 1:" in result
        assert "Step 2:" in result
    
    def test_skeleton_of_thought_strategy_custom_points(self):
        """Test skeleton of thought strategy with custom num_points."""
        strategy = SkeletonOfThoughtStrategy(num_points=7)
        prompt = "How to optimize SQL queries?"
        result = strategy.improve(prompt)
        
        assert "7" in result
    
    def test_skeleton_of_thought_strategy_with_kwargs(self):
        """Test skeleton of thought strategy with num_points in kwargs."""
        strategy = SkeletonOfThoughtStrategy()
        prompt = "How to optimize SQL queries?"
        result = strategy.improve(prompt, num_points=6)
        
        assert "6" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = SkeletonOfThoughtStrategy()
        assert strategy.get_strategy_name() == "Skeleton of Thought"


class TestReActStrategy(unittest.TestCase):
    """Tests for ReActStrategy."""
    
    def test_react_strategy_without_domain(self):
        """Test ReAct strategy without domain."""
        strategy = ReActStrategy()
        prompt = "Debug this API endpoint"
        result = strategy.improve(prompt)
        
        assert prompt in result
        assert "Thought:" in result or "thought" in result.lower()
        assert "Action:" in result or "action" in result.lower()
        assert "Observation:" in result or "observation" in result.lower()
        assert "Final Answer:" in result
    
    def test_react_strategy_with_domain(self):
        """Test ReAct strategy with domain."""
        strategy = ReActStrategy(domain="software engineering")
        prompt = "Debug this API endpoint"
        result = strategy.improve(prompt)
        
        assert "software engineering" in result
        assert prompt in result
    
    def test_react_strategy_with_kwargs(self):
        """Test ReAct strategy with domain in kwargs."""
        strategy = ReActStrategy()
        prompt = "Debug this API endpoint"
        result = strategy.improve(prompt, domain="backend engineering")
        
        assert "backend engineering" in result
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = ReActStrategy()
        assert strategy.get_strategy_name() == "ReAct"

