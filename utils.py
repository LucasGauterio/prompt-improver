from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from typing import Optional


def print_improved_prompt(original: str, improved: str, strategy: str):
    """
    Print the original and improved prompts with colored formatting.
    
    Args:
        original: Original prompt
        improved: Improved prompt
        strategy: Strategy name used
    """
    console = Console()
    
    # Print original prompt
    console.print(Panel(
        Text(original, style="blue"),
        title="[bold green]Original Prompt[/bold green]",
        border_style="green"
    ))
    
    console.print()
    
    # Print strategy info
    console.print(f"[bold yellow]Strategy:[/bold yellow] [cyan]{strategy}[/cyan]")
    console.print()
    
    # Print improved prompt
    console.print(Panel(
        Text(improved, style="bright_blue"),
        title="[bold green]Improved Prompt[/bold green]",
        border_style="bright_green"
    ))
    
    console.print()
    console.print(f"[yellow]{'='*70}[/yellow]")


def print_error(message: str):
    """Print an error message with colored formatting."""
    console = Console()
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str):
    """Print an info message with colored formatting."""
    console = Console()
    console.print(f"[bold cyan]Info:[/bold cyan] {message}")

