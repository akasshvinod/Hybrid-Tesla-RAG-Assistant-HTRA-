from __future__ import annotations
import sys
import logging
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

from src.pipeline.rag_pipeline import answer_query, chat_history
from src.utils.logger import get_logger

# =========================================================
# SILENT MODE (hide logs in CLI)
# =========================================================

logging.getLogger().setLevel(logging.ERROR)   # hide INFO logs globally

console = Console()
logger = get_logger(__name__)


# =========================================================
# UI Components
# =========================================================

def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]TESLA MODEL 3 — HYBRID RAG ASSISTANT[/bold cyan]\n"
            "Ask anything about your vehicle.\n"
            "Commands: [yellow]exit[/yellow], [yellow]clear[/yellow], [yellow]history[/yellow], [yellow]logs[/yellow]",
            border_style="cyan",
        )
    )


def print_answer(answer: str):
    console.print(
        Panel(
            answer,
            title="[bold green]Assistant[/bold green]",
            border_style="green",
            expand=True
        )
    )


def print_metrics(meta: dict):
    console.print(
        Panel(
            f"[cyan]Retrieval:[/cyan] {meta.get('retrieval_latency', '-')} ms\n"
            f"[cyan]LLM:[/cyan] {meta.get('llm_latency', '-')} ms\n"
            f"[cyan]Total:[/cyan] {meta.get('total_latency', '-')} ms\n"
            f"[cyan]Docs Used:[/cyan] {meta.get('docs_used', '-')}\n"
            f"[cyan]Chapter Auto-Detected:[/cyan] {meta.get('chapter_used', '-')}",
            title="[yellow]Latency Metrics[/yellow]",
            border_style="yellow",
            expand=True
        )
    )


def print_history():
    if not chat_history.messages:
        history_str = "[None]"
    else:
        history_str = "\n".join(
            f"[User] {m.content}" if m.type == "human" else f"[AI] {m.content}"
            for m in chat_history.messages
        )

    console.print(
        Panel(
            history_str,
            title="[bold cyan]Conversation History[/bold cyan]",
            border_style="cyan",
            expand=True
        )
    )


# =========================================================
# MAIN INTERACTIVE LOOP
# =========================================================

def main():
    print_banner()

    logs_visible = False

    while True:
        try:
            user_input = Prompt.ask("[bold white]You[/bold white]").strip()
            command = user_input.lower()

            # ---------------------
            # Exit
            # ---------------------
            if command in ("exit", "quit"):
                console.print("\n[bold red]Exiting Tesla RAG Assistant...[/bold red]")
                sys.exit(0)

            # ---------------------
            # Clear screen
            # ---------------------
            if command == "clear":
                console.clear()
                print_banner()
                continue

            # ---------------------
            # Show chat memory
            # ---------------------
            if command == "history":
                print_history()
                continue

            # ---------------------
            # Toggle logs
            # ---------------------
            if command == "logs":
                logs_visible = not logs_visible
                level = logging.INFO if logs_visible else logging.ERROR
                logging.getLogger().setLevel(level)

                msg = "Logs are now VISIBLE." if logs_visible else "Logs are now HIDDEN."
                console.print(f"[yellow]{msg}[/yellow]")
                continue

            # ---------------------
            # RUN RAG PIPELINE
            # ---------------------
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                result = answer_query(user_input)

            print_answer(result["answer"])
            print_metrics(result)

        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted. Exiting...[/bold red]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
