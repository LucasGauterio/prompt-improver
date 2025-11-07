#!/usr/bin/env python3
"""
Example usage of the Prompt Improver tool.
"""

from improver import PromptImprover
from utils import print_improved_prompt


def main():
    """Demonstrate different strategies."""
    improver = PromptImprover()
    
    # Example prompt
    original_prompt = "Explain recursion in programming"
    
    print("=" * 70)
    print("PROMPT IMPROVER EXAMPLES")
    print("=" * 70)
    print()
    
    # Example 1: Role Prompting
    print("Example 1: Role Prompting")
    improved = improver.improve(
        original_prompt,
        strategy="role",
        role="a university professor of computer science"
    )
    print_improved_prompt(original_prompt, improved, "role")
    
    # Example 2: Chain of Thought
    print("\nExample 2: Chain of Thought")
    prompt2 = "Classify the log severity: 'Disk usage at 85%'"
    improved2 = improver.improve(prompt2, strategy="cot")
    print_improved_prompt(prompt2, improved2, "cot")
    
    # Example 3: ReAct
    print("\nExample 3: ReAct")
    prompt3 = "Debug why the API endpoint POST /products returns HTTP 500"
    improved3 = improver.improve(
        prompt3,
        strategy="react",
        domain="a Go backend engineer"
    )
    print_improved_prompt(prompt3, improved3, "react")
    
    # Example 4: Tree of Thought
    print("\nExample 4: Tree of Thought")
    prompt4 = "Design a service that processes millions of images daily"
    improved4 = improver.improve(prompt4, strategy="tot", num_branches=3)
    print_improved_prompt(prompt4, improved4, "tot")
    
    # Example 5: Skeleton of Thought
    print("\nExample 5: Skeleton of Thought")
    prompt5 = "How to optimize SQL queries for better performance?"
    improved5 = improver.improve(prompt5, strategy="sot", num_points=5)
    print_improved_prompt(prompt5, improved5, "sot")
    
    # Example 6: Self-Consistency
    print("\nExample 6: Self-Consistency")
    prompt6 = "How many database queries will this code execute if there are N users?"
    improved6 = improver.improve(prompt6, strategy="self-consistency", num_paths=3)
    print_improved_prompt(prompt6, improved6, "self-consistency")
    
    # Example 7: Few-Shot
    print("\nExample 7: Few-Shot")
    prompt7 = "Classify: 'API response time is above threshold'"
    improved7 = improver.improve(prompt7, strategy="few-shot", num_examples=2)
    print_improved_prompt(prompt7, improved7, "few-shot")


if __name__ == '__main__':
    main()

