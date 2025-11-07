"""
Unit tests for utility functions.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import print_improved_prompt, print_error, print_info


class TestUtils:
    """Tests for utility functions."""
    
    @patch('utils.Console')
    def test_print_improved_prompt(self, mock_console_class):
        """Test print_improved_prompt function."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        print_improved_prompt(
            original="Original prompt",
            improved="Improved prompt",
            strategy="role"
        )
        
        # Verify console was created
        mock_console_class.assert_called_once()
        
        # Verify print was called (at least once for panels and text)
        assert mock_console.print.call_count >= 3
    
    @patch('utils.Console')
    def test_print_error(self, mock_console_class):
        """Test print_error function."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        print_error("Test error message")
        
        # Verify console was created
        mock_console_class.assert_called_once()
        
        # Verify print was called with error message
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Error:" in call_args
        assert "Test error message" in call_args
    
    @patch('utils.Console')
    def test_print_info(self, mock_console_class):
        """Test print_info function."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        print_info("Test info message")
        
        # Verify console was created
        mock_console_class.assert_called_once()
        
        # Verify print was called with info message
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Info:" in call_args
        assert "Test info message" in call_args

