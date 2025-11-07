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
from tests.test_strategies import (
    TestRoleStrategy,
    TestFewShotStrategy,
    TestChainOfThoughtStrategy,
    TestSelfConsistencyStrategy,
    TestTreeOfThoughtStrategy,
    TestSkeletonOfThoughtStrategy,
    TestReActStrategy
)
from tests.test_improver import TestPromptImprover
from tests.test_integration import TestIntegration


def main():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestRoleStrategy,
        TestFewShotStrategy,
        TestChainOfThoughtStrategy,
        TestSelfConsistencyStrategy,
        TestTreeOfThoughtStrategy,
        TestSkeletonOfThoughtStrategy,
        TestReActStrategy,
        TestPromptImprover,
        TestIntegration,
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

