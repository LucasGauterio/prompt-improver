"""
Unit tests for LLMClient class.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import LLMClient


class TestLLMClient:
    """Tests for LLMClient class."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_client.ChatOpenAI')
    def test_init_with_openai(self, mock_chat_openai):
        """Test LLMClient initialization with OpenAI provider."""
        client = LLMClient(provider="openai", model_name="gpt-4o-mini")
        
        assert client.provider == "openai"
        assert client.model_name == "gpt-4o-mini"
        mock_chat_openai.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_client.ChatOpenAI')
    def test_init_with_openai_default_model(self, mock_chat_openai):
        """Test LLMClient initialization with OpenAI default model."""
        client = LLMClient(provider="openai")
        
        assert client.provider == "openai"
        assert client.model_name == "gpt-4o-mini"
        mock_chat_openai.assert_called_once()
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'})
    @patch('llm_client.ChatGoogleGenerativeAI')
    def test_init_with_gemini(self, mock_chat_gemini):
        """Test LLMClient initialization with Gemini provider."""
        with patch('llm_client.GEMINI_AVAILABLE', True):
            client = LLMClient(provider="gemini", model_name="gemini-2.0-flash-exp")
            
            assert client.provider == "gemini"
            assert client.model_name == "gemini-2.0-flash-exp"
            mock_chat_gemini.assert_called_once()
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'})
    def test_init_with_gemini_not_available(self):
        """Test LLMClient initialization with Gemini when not available."""
        with patch('llm_client.GEMINI_AVAILABLE', False):
            with pytest.raises(ImportError, match="langchain-google-genai is not installed"):
                LLMClient(provider="gemini")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_openai(self):
        """Test LLMClient initialization without OpenAI API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            LLMClient(provider="openai")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_gemini(self):
        """Test LLMClient initialization without Gemini API key."""
        with patch('llm_client.GEMINI_AVAILABLE', True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
                LLMClient(provider="gemini")
    
    def test_init_with_invalid_provider(self):
        """Test LLMClient initialization with invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="invalid")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_client.ChatOpenAI')
    def test_init_with_custom_temperature(self, mock_chat_openai):
        """Test LLMClient initialization with custom temperature."""
        client = LLMClient(provider="openai", temperature=0.5)
        
        assert client.temperature == 0.5
        # Verify temperature was passed to ChatOpenAI
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['temperature'] == 0.5
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_client.ChatOpenAI')
    @patch('llm_client.ChatPromptTemplate')
    def test_invoke(self, mock_prompt_template_class, mock_chat_openai):
        """Test LLMClient invoke method."""
        # Create a mock chain that will be returned
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Test response"
        
        # Create mocks for the pipe chain: prompt | llm | output_parser
        mock_prompt = MagicMock()
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
        
        # Patch ChatPromptTemplate.from_template to return our mock
        mock_prompt_template_class.from_template.return_value = mock_prompt
        
        client = LLMClient(provider="openai")
        
        # Patch the llm and output_parser to work with our mock chain
        with patch.object(client, 'llm', MagicMock()):
            with patch.object(client, 'output_parser', MagicMock()):
                result = client.invoke("Test {variable}", variable="value")
                
                assert result == "Test response"
                mock_chain.invoke.assert_called_once_with({"variable": "value"})
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('llm_client.ChatOpenAI')
    @patch('llm_client.ChatPromptTemplate')
    def test_invoke_direct(self, mock_prompt_template_class, mock_chat_openai):
        """Test LLMClient invoke_direct method."""
        # Create a mock chain that will be returned
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Direct response"
        
        # Create mocks for the pipe chain: prompt | llm | output_parser
        mock_prompt = MagicMock()
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
        
        # Patch ChatPromptTemplate.from_messages to return our mock
        mock_prompt_template_class.from_messages.return_value = mock_prompt
        
        client = LLMClient(provider="openai")
        
        # Patch the llm and output_parser to work with our mock chain
        with patch.object(client, 'llm', MagicMock()):
            with patch.object(client, 'output_parser', MagicMock()):
                result = client.invoke_direct("Direct message")
                
                assert result == "Direct response"
                mock_prompt_template_class.from_messages.assert_called_once_with([("human", "Direct message")])
                mock_chain.invoke.assert_called_once_with({})

