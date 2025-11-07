#!/usr/bin/env python3
"""
Simple test runner for prompt-improver tests.
Can be used as an alternative to pytest.
"""

import sys
import unittest
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
from tests.test_llm_client import TestLLMClient
from tests.test_utils import TestUtils
from tests.test_base_strategy import TestBaseStrategy
from tests.test_role_strategy import TestRoleStrategy
from tests.test_few_shot_strategy import TestFewShotStrategy
from tests.test_chain_of_thought_strategy import TestChainOfThoughtStrategy
from tests.test_self_consistency_strategy import TestSelfConsistencyStrategy
from tests.test_tree_of_thought_strategy import TestTreeOfThoughtStrategy
from tests.test_skeleton_of_thought_strategy import TestSkeletonOfThoughtStrategy
from tests.test_react_strategy import TestReActStrategy
from tests.test_improver import TestPromptImprover


def main():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLLMClient,
        TestUtils,
        TestBaseStrategy,
        TestRoleStrategy,
        TestFewShotStrategy,
        TestChainOfThoughtStrategy,
        TestSelfConsistencyStrategy,
        TestTreeOfThoughtStrategy,
        TestSkeletonOfThoughtStrategy,
        TestReActStrategy,
        TestPromptImprover,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()


