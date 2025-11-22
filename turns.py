import re
from config import TURNS_TO_KEEP, AUTO_SUMMARIZE, DEFAULT_MODEL_TOKEN_LIMIT, CONTEXT_THRESHOLD, prompts_dir
from rich.console import Console
from pathlib import Path

console = Console()

def ensure_current_turn(scene_path):
    """
    Ensures the scene has a # Description section
    and that the last turn header is present at the end.
    Returns the current turn number (last turn found).
    Does NOT create a new turn — that is done in advance_turn().
    """
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return 0  # fallback

    lines = scene_path.read_text(encoding="utf-8").splitlines()
    turn_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)

    # --- Ensure Description exists ---
    has_description = any(line.strip().lower().startswith("# description") for line in lines)
    first_turn_index = next((i for i, l in enumerate(lines) if turn_pattern.match(l)), len(lines))

    if not has_description:
        console.print("[cyan]No # Description section found — creating one…[/cyan]")
        description_content = lines[:first_turn_index]
        remaining = lines[first_turn_index:]
        lines = ["# Description", ""] + description_content + [""] + remaining

    # --- Determine last turn ---
    turn_numbers = [int(turn_pattern.match(l).group(1)) for l in lines if turn_pattern.match(l)]
    if turn_numbers:
        last_turn = max(turn_numbers)
    else:
        last_turn = 0  # no turn exists yet

    # --- Save back ---
    scene_path.write_text("\n".join(lines), encoding="utf-8")

    return last_turn

def advance_turn(scene_path, agent, current_turn=None):
    if not scene_path or not scene_path.exists():
        console.print("[red]No active scene file found.[/red]")
        return

    lines = scene_path.read_text(encoding="utf-8").splitlines()
    turn_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)

    # Determine last turn if current_turn not provided
    turn_numbers = [int(turn_pattern.match(l).group(1)) for l in lines if turn_pattern.match(l)]
    last_turn = max(turn_numbers) if turn_numbers else 0

    if current_turn is None:
        current_turn = last_turn

    new_turn = last_turn + 1

    # --- Auto-summary based on token usage ---
    if AUTO_SUMMARIZE and hasattr(agent, "_last_token_usage") and DEFAULT_MODEL_TOKEN_LIMIT:
        usage_ratio = agent._last_token_usage / DEFAULT_MODEL_TOKEN_LIMIT
        if usage_ratio >= CONTEXT_THRESHOLD:
            console.print(f"[yellow]Token usage {usage_ratio*100:.1f}% — auto-summarizing previous turns[/yellow]")
            summarize_scene_turns(scene_path, agent)
            # Refresh lines after summarization
            lines = scene_path.read_text(encoding="utf-8").splitlines()

    # --- Append new turn ---
    console.print(f"[cyan]Advancing to Turn {new_turn}…[/cyan]")
    lines.append(f"# Turn {new_turn}")
    scene_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[bold green]Turn {new_turn} created.[/bold green]")

    return new_turn



def summarize_turn(turn_text, agent, turn_num):
    """
    Sends a single turn text to the LLM for summarization.
    Returns the summary string.
    """
    from Prompt_Manager2000 import PromptManager
    pm = PromptManager(prompts_dir)
    summary_messages = pm.build_turn_summary_messages(turn_text, turn_num)
    summary = agent.chat(summary_messages).strip()
    return summary

turn_header_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)
summary_header = "## Summary:"

def summarize_scene_turns(scene_path: Path, agent, turns_to_keep: int = None):
    from rich.console import Console
    from config import TURNS_TO_KEEP
    import re

    console = Console()

    if not scene_path or not scene_path.exists():
        console.print("[red]Scene file not found.[/red]")
        return

    if turns_to_keep is None:
        turns_to_keep = TURNS_TO_KEEP

    turn_pattern = re.compile(r"^#\s*Turn\s+(\d+)", re.I)
    summary_marker = "## Summary"

    console.print("\n[cyan]Starting turn summarization…[/cyan]\n")

    while True:
        lines = scene_path.read_text(encoding="utf-8").splitlines()
        # Find all turns
        turn_positions = [(i, int(m.group(1))) for i, line in enumerate(lines) if (m := turn_pattern.match(line))]

        if not turn_positions:
            console.print("[yellow]No turns found to summarize.[/yellow]")
            return

        # Determine which turns to summarize (skip last N)
        if turns_to_keep == 0 or len(turn_positions) <= turns_to_keep:
            turns_to_summarize = turn_positions
        else:
            turns_to_summarize = turn_positions[:-turns_to_keep]

        # Find the first unsummarized turn
        next_turn = None
        for idx, (line_idx, turn_num) in enumerate(turns_to_summarize):
            # Find end of turn
            next_idx = turn_positions[idx + 1][0] if idx + 1 < len(turn_positions) else len(lines)
            turn_block = lines[line_idx:next_idx]
            turn_text = "\n".join(turn_block)

            if summary_marker not in turn_text:
                next_turn = (line_idx, turn_num, next_idx, turn_block)
                break

        if not next_turn:
            # No more turns to summarize
            break

        line_idx, turn_num, next_idx, turn_block = next_turn

        console.print(f"[cyan]→ Turn {turn_num}: generating summary…[/cyan]")

        summary = summarize_turn("\n".join(turn_block), agent, turn_num).strip()

        new_block = [
            f"# Turn {turn_num}",
            "## Summary",
            summary,
            "",
            "## Full Turn",
        ] + turn_block[1:]  # preserve original lines except header

        # Insert summary immediately
        lines[line_idx:next_idx] = new_block
        scene_path.write_text("\n".join(lines), encoding="utf-8")

        console.print(f"[green]✓ Turn {turn_num} summarized.[/green]\n")

    console.print("[bold green]All missing turn summaries completed.[/bold green]\n")
