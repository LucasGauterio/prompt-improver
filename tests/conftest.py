"""
Pytest configuration and fixtures for prompt-improver tests.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked LLM response")
    return mock


@pytest.fixture
def mock_llm_client(mock_llm):
    """Create a mock LLMClient for testing."""
    from llm_client import LLMClient
    
    client = Mock(spec=LLMClient)
    client.invoke.return_value = "Mocked response"
    client.invoke_direct.return_value = "Mocked direct response"
    client.llm = mock_llm
    client.provider = "openai"
    client.model_name = "gpt-4o-mini"
    return client

