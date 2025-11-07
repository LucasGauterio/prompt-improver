# Prompt Improver

A Python tool to intelligently improve prompts by applying various prompt engineering strategies. The tool uses LLM-powered prompt improvement, not just template-based instructions.

**Note**: LangChain is used internally for implementation purposes only. The improved prompts themselves are generic and framework-agnostic, suitable for use with any LLM.

## Features

- **Multi-Provider Support**: Works with both **OpenAI** and **Google Gemini** models
- **Framework-Agnostic Output**: Improved prompts are generic and contain no framework references
- **Multiple Strategies**: Apply various prompt engineering techniques:
  - **Role Prompting**: Use LLM to improve prompts with role/identity context
  - **One/Few-shot**: Use LLM to generate and integrate examples into prompts
  - **Chain of Thought (CoT)**: Use LLM to enhance prompts with step-by-step reasoning instructions
  - **Self-Consistency**: Use LLM to improve prompts with multiple reasoning paths
  - **Tree of Thought (ToT)**: Use LLM to enhance prompts with multiple solution branches
  - **Skeleton of Thought (SoT)**: Use LLM with SoT approach to structure and improve prompts
  - **ReAct**: Use LLM to improve prompts with reasoning and action frameworks

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install pytest separately if needed
pip install pytest==8.3.4

# Configure environment
cp .env.example .env
# Edit .env and add your API keys:
# For OpenAI: OPENAI_API_KEY=your-api-key-here
# For Gemini: GOOGLE_API_KEY=your-api-key-here
# Get Gemini API key from: https://makersuite.google.com/app/apikey
```

## Usage

### Command Line

```bash
# Using OpenAI (default)
python main.py "Your original prompt" --strategy role

# Using Gemini
python main.py "Your original prompt" --strategy role --provider gemini

# Specify a custom model
python main.py "Your original prompt" --strategy sot --provider openai --model gpt-4o
```

**Available strategies:**
- `role` - Role Prompting
- `few-shot` - One/Few-shot examples
- `cot` - Chain of Thought
- `self-consistency` - Self-Consistency
- `tot` - Tree of Thought
- `sot` - Skeleton of Thought
- `react` - ReAct

**Provider options:**
- `--provider openai` - Use OpenAI models (default: gpt-4o-mini)
- `--provider gemini` - Use Google Gemini models (default: gemini-2.0-flash-exp)
- `--model MODEL_NAME` - Specify a custom model name

### Python API

```python
from prompt_improver import PromptImprover
from prompt_improver.llm_client import LLMClient

# Initialize with default OpenAI client (uses OPENAI_API_KEY from .env)
improver = PromptImprover()

# Or use Gemini provider
improver = PromptImprover(provider="gemini")

# Or provide a custom LLM client
llm_client = LLMClient(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.7
)
improver = PromptImprover(llm_client=llm_client)

# Or use Gemini with custom model
gemini_client = LLMClient(
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    temperature=0.7
)
improver = PromptImprover(llm_client=gemini_client)

# Improve a prompt
improved_prompt = improver.improve(
    prompt="Your original prompt",
    strategy="sot"  # Skeleton of Thought
)
print(improved_prompt)
```

## Examples

```bash
# Apply role prompting with OpenAI
python main.py "Explain recursion" --strategy role

# Apply role prompting with Gemini
python main.py "Explain recursion" --strategy role --provider gemini

# Apply chain of thought
python main.py "Classify this log: Disk usage at 85%" --strategy cot

# Apply Skeleton of Thought
python main.py "Explain machine learning" --strategy sot --num-points 5

# Apply Skeleton of Thought with Gemini
python main.py "Explain machine learning" --strategy sot --num-points 5 --provider gemini

# Apply ReAct with custom model
python main.py "Debug this API endpoint" --strategy react --provider openai --model gpt-4o
```

## Testing

The project includes comprehensive tests for all strategies and the main PromptImprover class.

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_strategies.py -v

# Run specific test class
pytest tests/test_strategies.py::TestRoleStrategy -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Structure

- `tests/test_strategies.py` - Tests for all 7 strategies
- `tests/test_improver.py` - Tests for PromptImprover class
- `tests/test_integration.py` - Integration tests

All strategies are tested for:
- Basic functionality
- Parameter handling
- Strategy name retrieval
- Edge cases

