"""LangChain LLM client setup for prompt improvement."""
import os
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Try to import Google Gemini support (optional dependency)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None


class LLMClient:
    """LangChain-based LLM client for prompt improvement with OpenAI and Gemini support."""
    
    def __init__(
        self,
        provider: Literal["openai", "gemini"] = "gemini",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use - "openai" or "gemini" (default: "openai")
            model_name: Model name to use. If None, uses defaults:
                       - OpenAI: "gpt-4o-mini"
                       - Gemini: "gemini-2.0-flash-exp"
            temperature: Temperature for generation (default: 0.7)
            api_key: API key (default: from environment variables)
            
        Raises:
            ValueError: If provider is not supported or API key is missing
            ImportError: If Gemini provider is requested but langchain-google-genai is not installed
        """
        self.provider = provider.lower()
        
        # Set default model names if not provided
        if model_name is None:
            if self.provider == "openai":
                model_name = "gpt-4o-mini"
            elif self.provider == "gemini":
                model_name = "gemini-2.0-flash-exp"
            else:
                raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'.")
        
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize provider-specific LLM
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Please set it in your .env file or pass it as a parameter."
                )
            
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=self.api_key
            )
            
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "langchain-google-genai is not installed. "
                    "Install it with: pip install langchain-google-genai"
                )
            
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Please set it in your .env file or pass it as a parameter."
                )
            
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'.")
        
        self.output_parser = StrOutputParser()
    
    def invoke(self, prompt_template: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt template.
        
        Args:
            prompt_template: Prompt template string
            **kwargs: Variables to fill in the template
            
        Returns:
            LLM response as string
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.output_parser
        return chain.invoke(kwargs)
    
    def invoke_direct(self, message: str) -> str:
        """
        Invoke the LLM with a direct message (no template).
        
        Args:
            message: Direct message to send to LLM
            
        Returns:
            LLM response as string
        """
        prompt = ChatPromptTemplate.from_messages([("human", message)])
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({})

