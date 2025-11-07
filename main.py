#!/usr/bin/env python3
"""
Command-line interface for the Prompt Improver tool.
"""

import argparse
import sys
from improver import PromptImprover
from utils import print_improved_prompt, print_error, print_info


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Improve prompts using various prompt engineering strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available strategies:
  role              - Role Prompting (add role/identity context)
  few-shot          - Few-Shot Learning (add examples)
  cot               - Chain of Thought (step-by-step reasoning)
  self-consistency  - Self-Consistency (multiple reasoning paths)
  tot               - Tree of Thought (explore multiple branches)
  sot               - Skeleton of Thought (skeleton then expand)
  react             - ReAct (alternate Thought and Action)

Examples:
  python main.py "Explain recursion" --strategy role
  python main.py "Classify this log" --strategy cot
  python main.py "Debug this API" --strategy react
        """
    )
    
    parser.add_argument(
        'prompt',
        type=str,
        help='The original prompt to improve'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        required=True,
        help='Strategy to apply (role, few-shot, cot, self-consistency, tot, sot, react)'
    )
    
    parser.add_argument(
        '--role',
        type=str,
        help='Role for role strategy (e.g., "senior software engineer")'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        help='Domain for react strategy (e.g., "software engineering")'
    )
    
    parser.add_argument(
        '--num-examples',
        type=int,
        default=2,
        help='Number of examples for few-shot strategy (default: 2)'
    )
    
    parser.add_argument(
        '--num-paths',
        type=int,
        default=3,
        help='Number of paths for self-consistency strategy (default: 3)'
    )
    
    parser.add_argument(
        '--num-branches',
        type=int,
        default=3,
        help='Number of branches for tree-of-thought strategy (default: 3)'
    )
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=5,
        help='Number of points for skeleton-of-thought strategy (default: 5)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'gemini'],
        default='openai',
        help='LLM provider to use: openai or gemini (default: openai)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to use (default: gpt-4o-mini for OpenAI, gemini-2.0-flash-exp for Gemini)'
    )
    
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List all available strategies and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize improver with provider
    from llm_client import LLMClient
    llm_client = LLMClient(
        provider=args.provider,
        model_name=args.model
    )
    improver = PromptImprover(llm_client=llm_client)
    
    # List strategies if requested
    if args.list_strategies:
        print_info("Available strategies:")
        for strategy_key in improver.get_available_strategies():
            info = improver.get_strategy_info(strategy_key)
            print(f"  - {strategy_key:20} : {info['name']}")
        sys.exit(0)
    
    # Prepare kwargs based on strategy
    kwargs = {}
    if args.strategy.lower() == 'role' and args.role:
        kwargs['role'] = args.role
    elif args.strategy.lower() == 'react' and args.domain:
        kwargs['domain'] = args.domain
    elif args.strategy.lower() == 'few-shot':
        kwargs['num_examples'] = args.num_examples
    elif args.strategy.lower() == 'self-consistency':
        kwargs['num_paths'] = args.num_paths
    elif args.strategy.lower() in ['tot', 'tree-of-thought']:
        kwargs['num_branches'] = args.num_branches
    elif args.strategy.lower() in ['sot', 'skeleton-of-thought']:
        kwargs['num_points'] = args.num_points
    
    # Improve the prompt
    try:
        improved = improver.improve(args.prompt, args.strategy, **kwargs)
        strategy_info = improver.get_strategy_info(args.strategy)
        print_improved_prompt(args.prompt, improved, strategy_info['name'])
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

